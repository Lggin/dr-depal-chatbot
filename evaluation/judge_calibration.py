#!/usr/bin/env python3
"""Judge calibration: known-good and known-bad answers the judge MUST get right before
its metrics are trusted (lesson from the previous project: test both the false-positive
and the false-negative direction, or the judge silently fails one of them).

  ./run.sh python3 evaluation/judge_calibration.py
Exit 1 if any expectation fails.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)

CTX = [
    {"chunk_id": "VAC_203_001", "metadata": {"document_name": "Vacuum System Manual", "section": "6.3"},
     "text": "[Alarm] VAC-203 [Title] Vacuum Pressure Low [Symptom] 진공압이 정상 범위에 도달하지 못함. 박스 Pickup 실패 가능. "
             "[Cause] 1. Vacuum hose leakage 2. Suction cup damage 3. Vacuum generator malfunction "
             "[Inspection] 1. Vacuum pressure 확인 2. Hose leakage 확인 3. Suction cup 상태 확인 [Normal Range] -60 ~ -80 kPa"},
    {"chunk_id": "MAINT_20260910_001", "metadata": {"document_name": "Maintenance Log"},
     "text": "2026-09-10 DP-01 VAC-203 발생. 증상: 진공압 -45 kPa에서 정체. 조치: 3번 suction cup 측 vacuum hose에 균열 발견, 호스 교체. 결과: 진공압 -72 kPa 복귀, 정상 가동."},
]
Q = "VAC-203이 뭐야? 정상 범위도 알려줘."
REF = "VAC-203은 진공압 저하(Vacuum Pressure Low) 알람이다. 정상 범위는 -60 ~ -80 kPa. 원인: 호스 누설, 패드 손상, 발생기 고장."

# (name, answer, expectations) — expectations are predicates on the JudgeResult
CASES = [
    ("fully grounded",
     "VAC-203은 진공압이 정상 범위에 도달하지 못하는 'Vacuum Pressure Low' 알람입니다. 정상 범위는 -60 ~ -80 kPa이며, 원인은 호스 누설, suction cup 손상, 진공 발생기 고장입니다.\n출처: Vacuum System Manual §6.3",
     {"groundedness == 1.0": lambda j: j.groundedness == 1.0, "no contradiction": lambda j: j.contradicts_reference is False,
      "uses source 1 only": lambda j: 1 in j.used_sources and 2 not in j.used_sources}),
    ("invented number",
     "VAC-203은 진공압 저하 알람입니다. 정상 범위는 -50 ~ -80 kPa이고, 원인은 호스 누설, 패드 손상, 발생기 고장입니다.",
     {"groundedness < 1.0": lambda j: j.groundedness is not None and j.groundedness < 1.0,
      "flags the -50 sentence": lambda j: any("-50" in s for s in j.unsupported)}),
    ("derived arithmetic",
     "VAC-203은 진공압 저하 알람입니다. 정상 범위는 -60 ~ -80 kPa입니다. 2026-09-10 사례에서는 -45 kPa로, 정상 하한보다 15 kPa 부족했습니다.",
     {"groundedness < 1.0": lambda j: j.groundedness is not None and j.groundedness < 1.0,
      "flags the 15 kPa sentence": lambda j: any("15" in s for s in j.unsupported)}),
    ("omission",
     "VAC-203은 진공압 저하 알람입니다.\n출처: Vacuum System Manual §6.3",
     {"grounded": lambda j: j.groundedness == 1.0, "missing normal range": lambda j: len(j.missing_from_reference) >= 1,
      "completeness < 1": lambda j: j.completeness is not None and j.completeness < 1.0}),
    ("honest not-in-docs",
     "VAC-203은 진공압 저하 알람이며 정상 범위는 -60 ~ -80 kPa입니다. 로봇 컨트롤러 IP 주소는 제공된 문서에서 확인할 수 없습니다.",
     {"grounded": lambda j: j.groundedness == 1.0}),
    ("fabricated cause",
     "VAC-203은 진공압 저하 알람입니다. 정상 범위는 -60 ~ -80 kPa이며, 주요 원인은 PLC 펌웨어 버그입니다.",
     {"groundedness < 1.0": lambda j: j.groundedness is not None and j.groundedness < 1.0,
      "contradiction or flagged": lambda j: j.contradicts_reference or any("PLC" in s for s in j.unsupported)}),
    ("machine-state range comparison (spec §18)",
     "현재 VAC-203 알람이 발생해 있으며 진공압은 -43 kPa입니다. 매뉴얼 기준 정상 범위인 -60 ~ -80 kPa보다 진공압이 부족합니다. 비전과 로봇은 정상이므로 진공 계통을 우선 점검하는 것이 적절합니다.",
     {"grounded (comparison counts)": lambda j: j.groundedness == 1.0}),
    ("elaborated mechanism",
     "VAC-203은 진공압 저하 알람입니다. 원인 중 suction cup 손상은 패드 고무가 경화되어 미세 누설이 생기는 현상으로, 보통 6개월마다 교체가 필요합니다.",
     {"groundedness < 1.0": lambda j: j.groundedness is not None and j.groundedness < 1.0,
      "flags the elaboration": lambda j: any("경화" in s or "6개월" in s for s in j.unsupported)}),
    ("over-citation",
     "VAC-203은 진공압 저하 알람입니다. 정상 범위는 -60 ~ -80 kPa입니다.\n출처: Vacuum System Manual §6.3; Maintenance Log",
     {"grounded": lambda j: j.groundedness == 1.0, "did not use source 2": lambda j: 2 not in j.used_sources}),
]


MACHINE_STATE = {"machine_state": "HOLD", "alarm": "VAC-203", "vacuum_pressure": -43, "vision_detection": True, "robot_state": "NORMAL"}


def main() -> int:
    from app.config import settings
    from app.llm.client import LlamaCppClient
    from evaluation.judge import AnswerJudge

    llm = LlamaCppClient(settings.llm_model_path, n_ctx=settings.llm_context_length, n_gpu_layers=settings.llm_n_gpu_layers, enable_thinking=False).load()
    judge = AnswerJudge(llm)
    failures = 0
    for name, answer, expects in CASES:
        ms = MACHINE_STATE if "machine-state" in name else None
        j = judge.judge(Q if ms is None else "지금 박스를 왜 못 집어?", answer, CTX, reference=None if ms else REF, machine_state=ms)
        print(f"\n=== {name} === ok={j.ok} groundedness={j.groundedness} completeness={j.completeness} "
              f"contradicts={j.contradicts_reference} used={j.used_sources}")
        for s in j.unsupported:
            print(f"   ✗ {s[:120]}")
        for m in j.missing_from_reference:
            print(f"   ⊘ {m[:120]}")
        for label, pred in expects.items():
            ok = j.ok and bool(pred(j))
            failures += 0 if ok else 1
            print(f"   {'✅' if ok else '❌'} {label}")
        if not j.ok:
            print("   raw:", j.raw[:300])
    llm.unload()
    print(f"\n{'✅ judge calibration passed' if failures == 0 else f'❌ {failures} expectation(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
