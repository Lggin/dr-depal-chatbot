"""Ingestion pipeline: corpus.yaml → chunks JSON (+manifest) → embeddings → Qdrant."""

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ingestion.chunker import Chunker
from ingestion.markdown_parser import parse_markdown
from ingestion.metadata_generator import build_metadata

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for key in ("collection", "sources"):
        if key not in cfg:
            raise ValueError(f"corpus config missing '{key}'")
    return cfg


# (path marker, prefix to keep) — labels match the legacy corpus and the held-out QA set:
#   ~/scratch_github_check/dr-depal-knowhow/docs/03-x.md -> docs/03-x.md
#   ~/depal-al/README.md                                  -> depal-al/README.md
#   corpus/raw/notes/user_notes.md                        -> notes/user_notes.md
_LABEL_RULES = (("dr-depal-knowhow/", ""), ("depal-al/", "depal-al/"), ("corpus/raw/", ""))


def source_label(cfg_path: str) -> str:
    """Stable, human-readable source id: path relative to its repo/project root."""
    p = os.path.expanduser(cfg_path)
    for marker, prefix in _LABEL_RULES:
        if marker in p:
            return prefix + p.split(marker, 1)[1]
    return Path(p).name


class IngestionPipeline:
    def __init__(self, config_path: Path, count_tokens, out_dir: Optional[Path] = None):
        self.cfg = load_config(config_path)
        self.count_tokens = count_tokens
        ch = self.cfg.get("chunking", {})
        self.chunker = Chunker(count_tokens, ch.get("max_tokens", 800), ch.get("min_tokens", 60), ch.get("overlap_tokens", 80))
        self.out_dir = out_dir or (PROJECT_ROOT / "corpus" / "processed")

    # ---------- build ----------

    def build(self) -> Tuple[List[str], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        t0 = time.perf_counter()
        texts, metas, ids = [], [], []
        per_source: List[Dict[str, Any]] = []
        missing: List[str] = []
        for src in self.cfg["sources"]:
            path = resolve_path(src["path"])
            if not path.exists():
                missing.append(str(path)); logger.warning(f"missing source: {path}"); continue
            label = source_label(src["path"])
            doc = parse_markdown(path.read_text(encoding="utf-8"), default_title=src.get("document_name"))
            chunks = self.chunker.chunk_document(doc, label)
            for c in chunks:
                texts.append(c.text); ids.append(c.chunk_id)
                metas.append(build_metadata(c, src, self.cfg))
            per_source.append({"source": label, "document": src.get("document_name") or doc.title, "sections": len(doc.sections),
                               "chunks": len(chunks), "split": sum(1 for c in chunks if c.n_parts > 1),
                               "merged": sum(1 for c in chunks if c.merged_from)})

        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            raise RuntimeError(f"duplicate chunk_ids after chunking: {dupes[:5]}")

        toks = sorted(m["tokens"] for m in metas)
        manifest = {
            "collection": self.cfg["collection"], "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": len(ids), "sources": len(per_source), "missing_sources": missing,
            "tokens": {"p50": toks[len(toks) // 2] if toks else 0, "p90": toks[int(len(toks) * 0.9)] if toks else 0,
                       "max": toks[-1] if toks else 0, "over_max": sum(1 for t in toks if t > self.chunker.max_tokens)},
            "by_document_type": dict(Counter(m["document_type"] for m in metas)),
            "by_system": dict(Counter(str(m["system"]) for m in metas)),
            "by_trust": dict(Counter(m["trust_level"] for m in metas)),
            "with_version": sum(1 for m in metas if m["software_version"]), "with_date": sum(1 for m in metas if m["as_of_date"]),
            "per_source": per_source, "build_seconds": round(time.perf_counter() - t0, 1),
        }
        return texts, metas, ids, manifest

    def write(self, texts, metas, ids, manifest) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"chunk_id": cid, "text": t, **m} for cid, t, m in zip(ids, texts, metas)]
        out = self.out_dir / f"{self.cfg['collection']}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        (self.out_dir / f"{self.cfg['collection']}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return out

    # ---------- index ----------

    @staticmethod
    def token_batches(lengths: List[int], max_items: int = 16, token_budget: int = 6000) -> List[slice]:
        """Contiguous batches capped by item count AND summed tokens.

        Attention memory scales with batch × longest² — a fixed batch of 16 still OOMed
        a 16GB GPU on ~2k-token legacy chunks, so long items get batches of 1-3.
        """
        out, start, total = [], 0, 0
        for i, n in enumerate(lengths):
            if i > start and (total + n > token_budget or i - start >= max_items):
                out.append(slice(start, i)); start, total = i, 0
            total += n
        if start < len(lengths):
            out.append(slice(start, len(lengths)))
        return out

    @staticmethod
    def index(texts, metas, ids, embedder, store, max_items: int = 16, token_budget: int = 6000) -> int:
        """Embed search_text, store body text; recreate the collection; verify the count."""
        store.create_collection(force=True)
        search_texts = [m["search_text"] for m in metas]
        lengths = [m.get("tokens") or embedder.count_tokens(s) for m, s in zip(metas, search_texts)]
        batches = IngestionPipeline.token_batches(lengths, max_items, token_budget)
        for sl in batches:
            vecs = embedder.embed(search_texts[sl], batch_size=sl.stop - sl.start)
            store.upsert_documents(texts=texts[sl], embeddings=vecs, metadata=metas[sl], chunk_ids=ids[sl])
        count = store.count()
        if count != len(ids):
            raise RuntimeError(f"index verification failed: {count} points in {store.collection_name}, expected {len(ids)}")
        logger.info(f"indexed {len(ids)} chunks into {store.collection_name} in {len(batches)} batches (verified)")
        return count
