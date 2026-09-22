"""Retrieval-only evaluation: Recall@k / Hit@k / MRR / nDCG + alarm parsing accuracy.

Runs the retrieval pipeline exactly as the chat pipeline would (including
follow-up contextualization) but never calls the LLM, so it finishes in seconds.
"""

from collections import defaultdict
from typing import Dict, List

from app.llm.generator import contextualize
from app.router.alarm_parser import extract_alarm_codes
from evaluation.dataset import EvalCase
from evaluation.metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k

K_LIST = (1, 3, 5)


def _history_messages(case: EvalCase) -> List[Dict[str, str]]:
    # Prior questions only; we don't have prior answers without the LLM. Codes live in questions anyway.
    return [{"role": "user", "content": q} for q in case.history]


class RetrievalEvaluator:
    def __init__(self, retrieval_pipeline):
        self.rp = retrieval_pipeline

    def run_case(self, case: EvalCase) -> dict:
        query, _, carried = contextualize(case.question, _history_messages(case))
        fused = self.rp.hybrid.retrieve(query, filters=case.filters)
        fused_ids = fused.chunk_ids()
        if self.rp.reranker is not None:
            final_ids = self.rp.search(query, filters=case.filters).chunk_ids()
        else:
            final_ids = fused_ids

        parsed = extract_alarm_codes(case.question) or carried
        alarm_ok = (parsed[0] if parsed else None) == case.expected_alarm_code

        row = {
            "id": case.id, "category": case.category, "question": case.question,
            "expected": case.expected_source, "retrieval_query": query if query != case.question else None,
            "fused_top": fused_ids[:10], "final": final_ids,
            "alarm_parsed": parsed, "alarm_ok": alarm_ok,
            "out_of_scope": case.expected_source == [],
            "unknown_source": case.expected_source is None,
        }
        if case.expected_source:
            for k in K_LIST:
                row[f"fused_recall@{k}"] = recall_at_k(fused_ids, case.expected_source, k)
                row[f"fused_hit@{k}"] = hit_at_k(fused_ids, case.expected_source, k)
                row[f"final_recall@{k}"] = recall_at_k(final_ids, case.expected_source, k)
                row[f"final_hit@{k}"] = hit_at_k(final_ids, case.expected_source, k)
            row["fused_mrr"] = mrr(fused_ids, case.expected_source)
            row["final_mrr"] = mrr(final_ids, case.expected_source)
            row["final_ndcg@5"] = ndcg_at_k(final_ids, case.expected_source, 5)
            row["final_rank"] = next((i for i, c in enumerate(final_ids, 1) if c in case.expected_source), None)
        return row

    def run(self, cases: List[EvalCase]) -> dict:
        rows = [self.run_case(c) for c in cases]
        scored = [r for r in rows if not r["out_of_scope"] and not r["unknown_source"]]
        summary = {
            "cases": len(rows), "scored": len(scored),
            "out_of_scope": sum(1 for r in rows if r["out_of_scope"]),
            "unknown_source": sum(1 for r in rows if r["unknown_source"]),
            "alarm_parsing_accuracy": mean([1.0 if r["alarm_ok"] else 0.0 for r in rows]),
        }
        for stage in ("fused", "final"):
            for k in K_LIST:
                summary[f"{stage}_recall@{k}"] = mean([r[f"{stage}_recall@{k}"] for r in scored])
                summary[f"{stage}_hit@{k}"] = mean([r[f"{stage}_hit@{k}"] for r in scored])
            summary[f"{stage}_mrr"] = mean([r[f"{stage}_mrr"] for r in scored])
        summary["final_ndcg@5"] = mean([r["final_ndcg@5"] for r in scored])

        by_cat: Dict[str, List[float]] = defaultdict(list)
        for r in scored:
            by_cat[r["category"]].append(r["final_hit@5"])
        summary["final_hit@5_by_category"] = {c: mean(v) for c, v in sorted(by_cat.items())}
        summary["misses_at_5"] = [r["id"] for r in scored if r["final_hit@5"] == 0.0]
        return {"summary": summary, "rows": rows}
