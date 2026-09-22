"""Retrieval smoke cases against corpus/processed/sample_chunks.json.

Shared by scripts/test_*_retrieval.py so dense / sparse / hybrid baselines are comparable.
(query, expected chunk_id, filters)
"""

CASES = [
    ("VAC-203이 뭐야?",                    "VAC_203_001", None),
    ("VAC203이 뭐야?",                     "VAC_203_001", None),          # unnormalized code
    ("진공압 정상 범위 알려줘",              "VAC_203_001", None),          # spec also in VAC_MANUAL_SPEC_001
    ("빨판이 박스를 못 잡아",                "TS_PICKUP_FAIL_001", None),   # colloquial 빨판
    ("박스 흡착 실패 원인이 뭐야?",           "TS_PICKUP_FAIL_001", None),
    ("캘리브레이션 방법 알려줘",              "SOP_CALIB_001", None),
    ("카메라가 응답이 없어",                 "VIS_301_001", None),
    ("서보 과부하 알람",                    "ROB_105_001", None),
    ("VAC-203 지난번에도 있었어?",           "MAINT_20260910_001", {"document_type": "maintenance"}),
    ("진공 문제 공식 매뉴얼만",              "VAC_203_001", {"trust_level": {"gte": 85}}),
]


def report(name: str, run):
    """run(query, filters) -> list[chunk_id]. Prints a rank table and R@k."""
    hits_at = {1: 0, 3: 0, 5: 0}
    print(f"\n[{name}]\n{'rank':>4}  {'top1':<22} {'query'}")
    for query, expected, filters in CASES:
        ids = run(query, filters)
        rank = ids.index(expected) + 1 if expected in ids else None
        for k in hits_at:
            if rank and rank <= k:
                hits_at[k] += 1
        mark = "✅" if rank == 1 else ("🟡" if rank and rank <= 3 else "❌")
        top1 = ids[0] if ids else "(no hits)"
        print(f"{mark}{(rank or '-'):>3}  {top1:<22} {query}" + (f"  [filter {filters}]" if filters else ""))
    n = len(CASES)
    print(f"{name}: R@1={hits_at[1]}/{n}  R@3={hits_at[3]}/{n}  R@5={hits_at[5]}/{n}")
    return hits_at
