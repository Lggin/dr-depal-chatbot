"""Persist results, print a Markdown summary, and gate on regressions vs. the previous run."""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

RESULTS_DIR = Path(__file__).parent / "results"

# metric name -> max allowed drop before the gate fails
GATES = {
    "retrieval": {"final_hit@5": 0.0, "final_recall@5": 0.0, "final_mrr": 0.02},
    # answer_correct tolerates one flipped case in 40 (0.025): at temperature 0.1 a borderline answer
    # (e.g. a 9-point series summarised to 3 points) flips between runs without any code change.
    # Judge metrics carry a little tolerance too: the judge runs at temp 0 but is not bit-stable.
    "full": {"answer_correct": 0.025, "citation_accuracy": 0.0, "refusal_recall": 0.0, "tool_selection_accuracy": 0.0,
             "groundedness_mean": 0.03},
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if v == v else "nan"
    return str(v)


def save(mode: str, result: dict, tag: str = "sample", collection: Optional[str] = None,
         baseline_tag: Optional[str] = None) -> Tuple[Path, Optional[dict]]:
    """Results are keyed by (mode, dataset tag) so different corpora never gate against each other.

    `baseline_tag` (A/B variants): compare against that tag's latest result instead of our own previous run,
    so e.g. `full_knowhow+lora` is always gated against the plain `full_knowhow` baseline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{mode}_{tag}"
    latest = RESULTS_DIR / f"latest_{key}.json"
    prev_path = RESULTS_DIR / f"latest_{mode}_{baseline_tag}.json" if baseline_tag else latest
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {"mode": mode, "dataset": tag, "collection": collection, "timestamp": stamp, **result}
    (RESULTS_DIR / f"{stamp}_{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return latest, previous


def markdown(mode: str, result: dict, previous: Optional[dict]) -> str:
    s = result["summary"]
    prev = (previous or {}).get("summary", {})
    lines = [f"## {mode} evaluation — {s.get('cases')} cases", "", "| metric | now | prev | Δ |", "|---|---|---|---|"]
    if previous and previous.get("dataset") and previous.get("collection"):
        lines.insert(1, f"prev: {previous['dataset']} @ {previous['collection']} ({previous.get('timestamp')})")
    for k, v in s.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            p = prev.get(k)
            delta = f"{v - p:+.3f}" if isinstance(p, (int, float)) and isinstance(v, float) and p == p and v == v else ""
            lines.append(f"| {k} | {_fmt(v)} | {_fmt(p)} | {delta} |")
    for k in ("final_hit@5_by_category", "answer_correct_by_category"):
        if k in s:
            lines += ["", f"**{k}**", ""] + [f"- {c}: {_fmt(v)}" for c, v in s[k].items()]
    for k in ("misses_at_5", "failures", "ungrounded_cases"):
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
