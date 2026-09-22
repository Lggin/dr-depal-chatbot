#!/usr/bin/env python3
"""Phase 3 E2E on the sample alarm corpus: intent routes + the spec §20 dialogue with the real LLM.

  QDRANT_COLLECTION=depalletizer_docs ./run.sh python3 scripts/test_troubleshooting.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dependencies import build_chat_pipeline

logging.basicConfig(level=logging.WARNING)

ROUTES = [
    "안녕",
    "지금 진공압 얼마야?",
    "지금 박스를 왜 못 집어?",
    "VAC-203 지난번에도 있었어?",
    "VIS-301 알람이 떴어. 뭘 확인해?",
]
DIALOGUE = ["박스를 못 집어.", "응", "응", "아니, -45 정도에서 안 올라가", "해결됐어"]


def show(r):
    print(f"   intent={r.intent} route={r.route} docs={r.retrieval.top_k} {r.latency_ms:.0f}ms"
          + (f" ts={ {k: r.troubleshooting[k] for k in ('issue','facts','current_step')} }" if r.troubleshooting else ""))
    print("   " + r.answer.replace("\n", "\n   ")[:700])


def main() -> int:
    pipe = build_chat_pipeline()
    print("\n=== routes ===")
    for i, q in enumerate(ROUTES):
        print(f"\nQ: {q}"); show(pipe.answer(f"route-{i}", q))
    print("\n=== spec §20 dialogue (one session) ===")
    for q in DIALOGUE:
        print(f"\nU: {q}"); show(pipe.answer("dialog", q))
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
