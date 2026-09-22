"""Import the legacy depal_knowhow_rag Corpus V2 chunks (574) into the new schema.

Kept for A/B comparison against the fresh markdown chunker; the same per-source
config (corpus.yaml) supplies document_type / trust so both corpora are comparable.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestion.metadata_generator import detect_components, infer_system, latest_date

# old document_type -> (new document_type, extra tag)
_TYPE_MAP = {
    "reference": ("manual", None), "architecture": ("manual", None), "configuration": ("manual", None),
    "methodology": ("sop", None), "release_note": ("manual", "release_note"), "user_note": ("engineer_note", None),
}


def _match_source_cfg(source: str, sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    tail = source.split("/")[-1]
    for s in sources:
        p = str(s["path"])
        if p.endswith(source) or p.endswith(tail):
            return s
    return None


def load_v2(path: Path, sources_cfg: List[Dict[str, Any]], globals_cfg: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Returns (texts, metadata, chunk_ids) in the shape QdrantClient.upsert_documents expects."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    texts, metas, ids = [], [], []
    for r in rows:
        cfg = _match_source_cfg(r["source"], sources_cfg) or {}
        new_type, extra_tag = _TYPE_MAP.get(r.get("document_type", "reference"), ("manual", None))
        components = r.get("components") or detect_components(r.get("search_text") or r["text"])
        tags = list(dict.fromkeys((cfg.get("tags") or []) + (r.get("tags") or []) + ([extra_tag] if extra_tag else [])))
        metas.append({
            "machine": globals_cfg.get("machine"), "manufacturer": globals_cfg.get("manufacturer"),
            "document_type": cfg.get("document_type", new_type),
            "document_name": cfg.get("document_name") or r.get("doc_title") or r["source"],
            "source": r["source"], "section": r.get("section"), "heading_path": [r["section"]] if r.get("section") else [],
            "system": infer_system(components, cfg.get("system")),
            "component": components[0] if components else None, "components": components,
            "alarm_code": None, "symptoms": [],
            "software_version": r.get("version_from"), "language": globals_cfg.get("language", "ko"),
            "trust_level": int(cfg.get("trust_level", r.get("trust_level", 80))),
            "tags": tags, "as_of_date": r.get("as_of_date") or latest_date(r["text"]),
            "contains_numeric_spec": bool(r.get("contains_numeric_spec")),
            "status": r.get("status"), "audience": r.get("audience"),
            "search_text": r.get("search_text") or r["text"],
            "legacy_v2": True,
        })
        texts.append(r["text"])
        ids.append(r["chunk_id"])
    return texts, metas, ids
