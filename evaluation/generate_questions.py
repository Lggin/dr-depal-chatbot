#!/usr/bin/env python3
"""Semi-automatic evaluation-question generation (spec §26 target: 300+ questions).

For each selected chunk, the local LLM writes one realistic operator/engineer question
that the chunk answers, plus a reference answer quoting the chunk's facts. Output is
§26-schema JSON with `auto_generated: true, reviewed: false`; expected_source is the
chunk itself, so these measure retrieval recall and answer groundedness — they are
NOT independent of the corpus wording the way the 40 legacy held-out questions are.

  ./run.sh python3 evaluation/generate_questions.py --n 300 --out evaluation/datasets/knowhow_generated.json
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judge import parse_json_block

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.WARNING)

GEN_SYSTEM = """You write evaluation questions for a Korean technical-support chatbot about a depalletizing robot system.
Given one document chunk, produce ONE question a field operator or engineer would realistically ask that this chunk answers, and a short reference answer that states the chunk's specific facts (numbers, names, versions) verbatim.
Rules: Korean, natural spoken register (e.g. "…얼마야?", "…어떻게 해?"), no yes/no questions, do not mention "문서" or "이 청크", the question must be answerable from the chunk alone.
Output JSON only: {"question": "...", "reference_answer": "...", "key_facts": ["...", "..."]}"""

_NUM_UNIT = re.compile(r"\b\d[\d,.]*\s*(?:개|줄|장|초|ms|mm|cm|kg|kPa|MPa|%|GB|MB|fps|FPS|배|회|번|일|토큰|px)\b")
_VERSION = re.compile(r"v0\.0\.\d+")
_CODE = re.compile(r"`([^`]{2,40})`")


def seed_must_include(answer: str, key_facts: List[str]) -> List[str]:
    toks: List[str] = []
    toks += _VERSION.findall(answer)
    toks += [m.group(1) for m in _CODE.finditer(answer)]
    toks += [re.sub(r"\s+", "", m.group(0)) for m in _NUM_UNIT.finditer(answer)]
    for f in key_facts:
        f = str(f).strip()
        if 2 <= len(f) <= 30 and f in answer:
            toks.append(f)
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:3]


def select_chunks(chunks: List[Dict[str, Any]], n: int, min_tokens: int, per_source_cap: int, seed: int) -> List[Dict[str, Any]]:
    """Informative chunks (enough tokens, not pure tables), spread across sources."""
    rng = random.Random(seed)
    pool = [c for c in chunks if c.get("tokens", 0) >= min_tokens and c["text"].count("|") < len(c["text"]) / 25]
    rng.shuffle(pool)
    pool.sort(key=lambda c: -c.get("tokens", 0))
    taken: Counter = Counter()
    out = []
    for c in pool:
        if taken[c["source"]] >= per_source_cap:
            continue
        out.append(c); taken[c["source"]] += 1
        if len(out) >= n:
            break
    return out


def generate_one(llm, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    head = f"{chunk.get('document_name')} > {chunk.get('section')}"
    user = f"CHUNK ({head}):\n{chunk['text'][:3500]}\n\nJSON:"
    gen = llm.generate([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
                       temperature=0.3, max_tokens=400)
    data = parse_json_block(gen.text)
    if not data or not data.get("question") or not data.get("reference_answer"):
        return None
    q = str(data["question"]).strip()
    if len(q) < 8 or len(q) > 120 or "문서" in q or "청크" in q:
        return None
    return {"question": q, "reference_answer": str(data["reference_answer"]).strip(), "key_facts": list(data.get("key_facts") or [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(ROOT / "corpus/processed/knowhow_docs.json"))
    ap.add_argument("--out", default=str(ROOT / "evaluation/datasets/knowhow_generated.json"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-tokens", type=int, default=120)
    ap.add_argument("--per-source-cap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true", help="keep existing cases in --out and only add new chunks")
    args = ap.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if args.resume and out_path.exists() else []
    done = {c["expected_source"][0] for c in existing if c.get("expected_source")}

    selected = [c for c in select_chunks(chunks, args.n + len(done), args.min_tokens, args.per_source_cap, args.seed) if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks → {len(selected)} selected (min_tokens={args.min_tokens}, cap/source={args.per_source_cap}), {len(existing)} existing")

    from app.config import settings
    from app.llm.client import LlamaCppClient
    llm = LlamaCppClient(settings.llm_model_path, n_ctx=settings.llm_context_length, n_gpu_layers=settings.llm_n_gpu_layers,
                         enable_thinking=False).load()

    cases = list(existing)
    failed = 0
    start = len(existing) + 1
    for i, ch in enumerate(selected, start=start):
        g = generate_one(llm, ch)
        if g is None:
            failed += 1; continue
        cases.append({
            "id": f"KG{i:03d}", "category": "generated", "question": g["question"], "expected_intent": "knowledge",
            "expected_alarm_code": None, "expected_source": [ch["chunk_id"]],
            "must_include": seed_must_include(g["reference_answer"], g["key_facts"]),
            "must_not_include": ["확인할 수 없습니다"], "expect_refusal": False,
            "reference_answer": g["reference_answer"], "source_section": f"{ch.get('document_name')} > {ch.get('section')}",
            "auto_generated": True, "reviewed": False,
        })
        if len(cases) % 25 == 0:
            out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {len(cases)} cases written…")
        if len(cases) - len(existing) >= args.n:
            break
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    llm.unload()
    print(f"wrote {out_path}: {len(cases)} cases ({failed} chunks skipped: unparseable/invalid)")
    print("by source:", dict(Counter(c["expected_source"][0].split("-")[0] for c in cases).most_common(8)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
