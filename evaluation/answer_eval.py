"""End-to-end answer evaluation through ChatPipeline.

Per case: must_include / must_not_include, refusal correctness, citation hit
(expected chunk in `sources`), validator flags, latency. Follow-up cases replay
their history questions in the same session first.
"""

import re
from collections import defaultdict
from typing import Dict, List

from evaluation.dataset import EvalCase
from evaluation.metrics import mean


def _norm(s: str) -> str:
    # Qwen writes "2,000 줄" / "12 회" for "2,000줄" / "12회"; compare without whitespace or markdown.
    return re.sub(r"[\s*`_]+", "", s).lower()


def _contains(text: str, needle: str) -> bool:
    return _norm(needle) in _norm(text)


class AnswerEvaluator:
    def __init__(self, chat_pipeline):
        self.pipe = chat_pipeline

    def run_case(self, case: EvalCase) -> dict:
        sid = f"eval-{case.id}"
        self.pipe.sessions.delete(sid)
        for prior in case.history:
            self.pipe.answer(sid, prior)
        r = self.pipe.answer(sid, case.question, filters=case.filters)

        missing = [s for s in case.must_include if not _contains(r.answer, s)]
        forbidden = [s for s in case.must_not_include if _contains(r.answer, s)]
        refusal_ok = r.validation.is_refusal == case.expect_refusal
        source_ids = [s.chunk_id for s in r.sources]
        if case.expected_source is None:
            citation_ok = None                       # unknown ground truth — excluded from the metric
        elif case.expected_source:
            citation_ok = any(c in source_ids for c in case.expected_source)
        else:
            citation_ok = not source_ids
        correct = not missing and not forbidden and refusal_ok

        return {
            "id": case.id, "category": case.category, "question": case.question,
            "answer": r.answer, "is_refusal": r.validation.is_refusal, "expect_refusal": case.expect_refusal,
            "missing": missing, "forbidden": forbidden, "refusal_ok": refusal_ok,
            "sources": source_ids, "expected": case.expected_source, "citation_ok": citation_ok,
            "validation_ok": r.validation.ok, "unsupported_numbers": r.validation.unsupported_numbers,
            "unsupported_codes": r.validation.unsupported_codes, "invalid_citations": r.validation.invalid_citations,
            "danger_flags": r.validation.danger_flags, "alarm_codes": r.alarm_codes,
            "intent": str(r.intent.value if hasattr(r.intent, "value") else r.intent), "expected_intent": case.expected_intent,
            "correct": correct, "latency_ms": r.latency_ms, "timing_ms": r.timing_ms,
        }

    def run(self, cases: List[EvalCase]) -> dict:
        rows = [self.run_case(c) for c in cases]
        answered = [r for r in rows if not r["expect_refusal"]]
        refusals = [r for r in rows if r["expect_refusal"]]
        summary = {
            "cases": len(rows),
            "answer_correct": mean([1.0 if r["correct"] else 0.0 for r in rows]),
            "must_include_pass": mean([1.0 if not r["missing"] else 0.0 for r in answered]),
            "must_not_include_pass": mean([1.0 if not r["forbidden"] else 0.0 for r in rows]),
            "citation_accuracy": mean([1.0 if r["citation_ok"] else 0.0 for r in rows if r["citation_ok"] is not None]),
            "refusal_recall": mean([1.0 if r["is_refusal"] else 0.0 for r in refusals]),        # should refuse → did
            "false_refusal_rate": mean([1.0 if r["is_refusal"] else 0.0 for r in answered]),     # shouldn't → did
            "unsupported_claim_rate": mean([0.0 if r["validation_ok"] else 1.0 for r in answered]),
            "danger_flag_rate": mean([1.0 if r["danger_flags"] else 0.0 for r in rows]),
            "intent_accuracy": mean([1.0 if r["intent"] == r["expected_intent"] else 0.0 for r in rows]),
            "latency_ms_median": sorted(r["latency_ms"] for r in rows)[len(rows) // 2] if rows else None,
            "latency_ms_p95": sorted(r["latency_ms"] for r in rows)[int(len(rows) * 0.95) - 1] if rows else None,
        }
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(1.0 if r["correct"] else 0.0)
        summary["answer_correct_by_category"] = {c: mean(v) for c, v in sorted(by_cat.items())}
        summary["failures"] = [r["id"] for r in rows if not r["correct"]]
        return {"summary": summary, "rows": rows}
