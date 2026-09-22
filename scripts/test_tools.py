#!/usr/bin/env python3
"""Phase 4 E2E on the sample corpus with mock tools (spec §17-§18).

  QDRANT_COLLECTION=depalletizer_docs ./run.sh python3 scripts/test_tools.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dependencies import build_chat_pipeline

logging.basicConfig(level=logging.WARNING)


def show(r):
    print(f"   intent={r.intent} route={r.route} tools={r.tools_used} docs={r.retrieval.top_k} {r.latency_ms:.0f}ms")
    if r.machine_state:
        print(f"   machine_state={json.dumps({k: r.machine_state[k] for k in ('machine_state','alarm','vacuum_pressure','vision_detection','robot_state') if k in r.machine_state}, ensure_ascii=False)}")
    print("   " + r.answer.replace("\n", "\n   ")[:900])


def main() -> int:
    pipe = build_chat_pipeline()
    print(f"\nscenario = {pipe.mock.scenario}")
    print("\n=== §17 current value → tool ===");            print("\nQ: 지금 진공압 얼마야?"); show(pipe.answer("a", "지금 진공압 얼마야?"))
    print("\n=== §17 reference value → RAG ===");           print("\nQ: 진공압 정상 범위가 얼마야?"); show(pipe.answer("b", "진공압 정상 범위가 얼마야?"))
    print("\n=== §17 both ===");                           print("\nQ: 지금 진공압이 정상인지 확인해줘."); show(pipe.answer("c", "지금 진공압이 정상인지 확인해줘."))
    print("\n=== §18 live diagnosis ===");                 print("\nQ: 지금 박스를 왜 못 집어?"); show(pipe.answer("d", "지금 박스를 왜 못 집어?"))
    print("\n=== §20 dialogue with tools prefilling facts ===")
    for q in ["박스를 못 집어.", "응", "해결됐어"]:
        print(f"\nU: {q}"); show(pipe.answer("e", q))
    print("\n=== §21 history with maintenance DB ===");    print("\nQ: VAC-203 지난번에도 있었어?"); show(pipe.answer("f", "VAC-203 지난번에도 있었어?"))
    print("\n=== scenario switch: camera_timeout ===")
    pipe.mock.set_scenario("camera_timeout")
    print("\nQ: 지금 알람 뭐 떠 있어?"); show(pipe.answer("g", "지금 알람 뭐 떠 있어?"))
    print("\nU: 박스가 있는데 검출이 안 돼"); show(pipe.answer("h", "박스가 있는데 검출이 안 돼"))
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
