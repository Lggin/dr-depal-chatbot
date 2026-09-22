#!/usr/bin/env python3
"""Turn real chat logs (spec §25) into evaluation-case candidates for human review.

  ./run.sh python3 scripts/harvest_logs.py --since-days 7 --out evaluation/datasets/candidates_from_logs.json
Candidates keep the logged answer, sources, validation flags and user feedback; a reviewer
fills must_include / expected_source and flips reviewed=true before the case joins a dataset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import ChatLogRepository, init_db


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=float, default=30)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--feedback", choices=["any", "down", "up", "none"], default="any")
    ap.add_argument("--out", default="evaluation/datasets/candidates_from_logs.json")
    args = ap.parse_args()

    init_db(settings.database_url)
    rows = ChatLogRepository().recent(limit=args.limit)
    cutoff = time.time() - args.since_days * 86400
    seen, cands = set(), []
    for r in rows:
        if r.timestamp < cutoff or r.question in seen:
            continue
        if args.feedback == "down" and r.user_feedback != "down": continue
        if args.feedback == "up" and r.user_feedback != "up": continue
        if args.feedback == "none" and r.user_feedback: continue
        seen.add(r.question)
        cands.append({
            "id": f"LOG{r.id:05d}", "category": "from_logs", "question": r.question,
            "expected_intent": r.intent, "expected_alarm_code": (r.alarm_codes or [None])[0],
            "expected_source": None, "must_include": [], "must_not_include": [], "expect_refusal": bool((r.validation or {}).get("is_refusal")),
            "logged": {"answer": r.answer, "sources": r.sources, "retrieved_chunk_ids": r.retrieved_chunk_ids,
                       "validation": r.validation, "user_feedback": r.user_feedback, "feedback_note": r.feedback_note,
                       "timestamp": r.timestamp},
            "auto_generated": True, "reviewed": False,
        })
    Path(args.out).write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(cands)} candidates from {len(rows)} logs "
          f"(feedback down={sum(1 for c in cands if c['logged']['user_feedback']=='down')}, up={sum(1 for c in cands if c['logged']['user_feedback']=='up')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
