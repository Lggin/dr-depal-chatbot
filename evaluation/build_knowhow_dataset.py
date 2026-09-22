#!/usr/bin/env python3
"""Convert the legacy held-out QA sets (question / reference_answer / source) into
the §26 evaluation schema against the knowhow_docs chunks.

`source` looks like "docs/03-grasp-estimation.md > 1. 윗면 평면 적합 > 두 경로의 역사".
We resolve it to chunk_ids by matching the source label and the deepest heading
that appears in a chunk's section path (all parts of a split section count).
must_include is seeded heuristically from the reference answer (numbers with
units, quoted/backticked terms) — review and tighten by hand over time.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "corpus/processed/knowhow_docs.json"
OUT = ROOT / "evaluation/datasets/knowhow_questions.json"
LEGACY = [
    (Path("/home/dr/depal_knowhow_rag/data/heldout_qa.jsonl"), "developer"),
    (Path("/home/dr/depal_knowhow_rag/data/heldout_qa_operator.jsonl"), "operator"),
]

_QUOTED = re.compile(r"[`'\"‘’“”]([^`'\"‘’“”]{1,40})[`'\"‘’“”]")
_NUM_UNIT = re.compile(r"\b\d[\d,.]*\s*(?:개|줄|장|초|ms|mm|cm|kg|kPa|%|GB|MB|fps|FPS|배|회|번|일)\b")
_VERSION = re.compile(r"v0\.0\.\d+")


def _norm(s: str) -> str:
    return re.sub(r"[\s`*_~#()\[\]\-—–·]+", "", s.lower())


def _fragments(heading: str) -> List[str]:
    """Deepest-first candidates: the whole heading, then the pieces of an 'h2 — h3' compound."""
    pieces = [p.strip() for p in re.split(r"\s+[—–-]{1,2}\s+", heading) if p.strip()]
    cands = [heading] + list(reversed(pieces)) if len(pieces) > 1 else [heading]
    return [c for c in cands if len(_norm(c)) >= 4]


def resolve_chunks(source: str, chunks: List[dict]) -> List[str]:
    parts = [p.strip() for p in source.split(">")]
    label, headings = parts[0], parts[1:]
    in_doc = [c for c in chunks if c["source"] == label or c["source"].endswith(label.split("/")[-1])]
    if not in_doc:
        return []
    for depth in range(len(headings), 0, -1):          # deepest heading first, then fall back
        for frag in _fragments(headings[depth - 1]):
            h = _norm(frag)
            hits = [c["chunk_id"] for c in in_doc if h in _norm(c["section"])]
            if hits:
                return hits
    return []


def seed_must_include(answer: str) -> List[str]:
    toks: List[str] = []
    toks += [m.strip() for m in _VERSION.findall(answer)]
    toks += [m.group(1).strip() for m in _QUOTED.finditer(answer)]
    toks += [re.sub(r"\s+", "", m.group(0)) for m in _NUM_UNIT.finditer(answer)]
    seen, out = set(), []
    for t in toks:
        if len(t) >= 2 and t not in seen:
            seen.add(t); out.append(t)
    return out[:3]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(CHUNKS), help="processed chunks JSON to resolve expected_source against")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    cases, unresolved = [], []
    for path, audience in LEGACY:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, r in enumerate(rows, start=1):
            ids = resolve_chunks(r.get("source", ""), chunks)
            if not ids:
                unresolved.append((r.get("source"), r["question"][:50]))
            cases.append({
                "id": f"K{audience[0].upper()}{i:03d}",
                "category": audience,
                "question": r["question"],
                "expected_intent": "knowledge",
                "expected_alarm_code": None,
                "expected_source": ids or None,
                "must_include": seed_must_include(r["reference_answer"]),
                "must_not_include": ["확인할 수 없습니다"],
                "expect_refusal": False,
                "reference_answer": r["reference_answer"],
                "legacy_source": r.get("source"),
                "auto_generated": True,
            })
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    resolved = sum(1 for c in cases if c["expected_source"])
    print(f"wrote {out_path}: {len(cases)} cases, {resolved} with resolved expected_source, "
          f"{sum(1 for c in cases if c['must_include'])} with seeded must_include")
    for src, q in unresolved:
        print(f"  unresolved: {src}  ← {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
