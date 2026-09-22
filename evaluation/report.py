"""Persist results, print a Markdown summary, and gate on regressions vs. the previous run."""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

RESULTS_DIR = Path(__file__).parent / "results"

# metric name -> max allowed drop before the gate fails
GATES = {
    "retrieval": {"final_hit@5": 0.0, "final_recall@5": 0.0, "final_mrr": 0.02},
    "full": {"answer_correct": 0.0, "citation_accuracy": 0.0, "refusal_recall": 0.0},
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if v == v else "nan"
    return str(v)


def save(mode: str, result: dict) -> Tuple[Path, Optional[dict]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest = RESULTS_DIR / f"latest_{mode}.json"
    previous = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {"mode": mode, "timestamp": stamp, **result}
    (RESULTS_DIR / f"{stamp}_{mode}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return latest, previous


def markdown(mode: str, result: dict, previous: Optional[dict]) -> str:
    s = result["summary"]
    prev = (previous or {}).get("summary", {})
    lines = [f"## {mode} evaluation — {s.get('cases')} cases", "", "| metric | now | prev | Δ |", "|---|---|---|---|"]
    for k, v in s.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            p = prev.get(k)
            delta = f"{v - p:+.3f}" if isinstance(p, (int, float)) and isinstance(v, float) and p == p and v == v else ""
            lines.append(f"| {k} | {_fmt(v)} | {_fmt(p)} | {delta} |")
    for k in ("final_hit@5_by_category", "answer_correct_by_category"):
        if k in s:
            lines += ["", f"**{k}**", ""] + [f"- {c}: {_fmt(v)}" for c, v in s[k].items()]
    for k in ("misses_at_5", "failures"):
        if s.get(k):
            lines += ["", f"**{k}:** {', '.join(s[k])}"]
    return "\n".join(lines)


def gate(mode: str, result: dict, previous: Optional[dict]) -> Tuple[bool, list]:
    if not previous:
        return True, []
    s, p = result["summary"], previous.get("summary", {})
    failures = []
    for metric, tol in GATES.get(mode, {}).items():
        now, before = s.get(metric), p.get(metric)
        if isinstance(now, float) and isinstance(before, float) and now == now and before == before:
            if before - now > tol + 1e-9:
                failures.append(f"{metric}: {before:.3f} → {now:.3f} (drop {before-now:.3f} > {tol})")
    return not failures, failures
