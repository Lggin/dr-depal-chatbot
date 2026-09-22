#!/usr/bin/env python3
"""Build and index the corpus.

  ./run.sh python3 scripts/ingest.py --dry-run                 # chunk + manifest only, no models
  ./run.sh python3 scripts/ingest.py                           # chunk → embed → Qdrant (collection from corpus.yaml)
  ./run.sh python3 scripts/ingest.py --kind v2json --collection knowhow_v2   # legacy 574-chunk import for A/B
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from ingestion.pipeline import IngestionPipeline, PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("ingest")

V2_PATH = Path("/home/dr/depal_knowhow_rag/corpus/v2/chunks_v2_phase2_merged.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / settings.corpus_config))
    ap.add_argument("--kind", choices=["markdown", "v2json"], default="markdown")
    ap.add_argument("--collection", default=None, help="override collection name")
    ap.add_argument("--dry-run", action="store_true", help="chunk + write JSON/manifest, skip embedding/Qdrant")
    ap.add_argument("--v2-path", default=str(V2_PATH))
    ap.add_argument("--device", default=None, help="embedding device (default: cuda if available — the LLM is not loaded during ingest)")
    args = ap.parse_args()

    embedder = None
    if args.dry_run and args.kind == "markdown":
        # token counting without the embedding model: BGE-M3 tokenizer only
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(settings.embedding_model)
        count_tokens = lambda s: len(tok(s, add_special_tokens=True)["input_ids"])
    else:
        import torch
        from app.llm.embedding_client import EmbeddingClient
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        embedder = EmbeddingClient(settings.embedding_model, device=device); embedder.load()
        count_tokens = embedder.count_tokens

    if args.kind == "markdown":
        pipe = IngestionPipeline(Path(args.config), count_tokens)
        if args.collection:
            pipe.cfg["collection"] = args.collection
        texts, metas, ids, manifest = pipe.build()
        out = pipe.write(texts, metas, ids, manifest)
        collection = pipe.cfg["collection"]
    else:
        from ingestion.v2_import import load_v2
        cfg = load_config(Path(args.config))
        collection = args.collection or "knowhow_v2"
        texts, metas, ids = load_v2(Path(args.v2_path), cfg["sources"], cfg)
        manifest = {"collection": collection, "chunks": len(ids), "kind": "v2json", "source_file": args.v2_path}
        out = PROJECT_ROOT / "corpus/processed" / f"{collection}.json"
        out.write_text(json.dumps([{"chunk_id": c, "text": t, **m} for c, t, m in zip(ids, texts, metas)], ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "per_source"}, ensure_ascii=False, indent=1))
    if "per_source" in manifest:
        print(f"\n{'chunks':>6} {'split':>5} {'merged':>6}  source")
        for s in manifest["per_source"]:
            print(f"{s['chunks']:>6} {s['split']:>5} {s['merged']:>6}  {s['source']}")
    print(f"\nwrote {out}")

    if args.dry_run:
        return 0

    from app.rag.qdrant_client import QdrantClient
    store = QdrantClient(settings.qdrant_url, collection, embedder.embedding_dim); store.connect()
    IngestionPipeline.index(texts, metas, ids, embedder, store)
    info = store.get_collection_info()
    print(f"\n✅ {info['name']}: {info['points_count']} points ({info['vector_size']}d)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
