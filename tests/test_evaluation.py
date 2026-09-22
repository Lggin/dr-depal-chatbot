import json
import math
from pathlib import Path

import pytest

from evaluation import report
from evaluation.dataset import EvalCase, load_cases
from evaluation.metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k
from evaluation.retrieval_eval import RetrievalEvaluator
from app.schemas.retrieval import RetrievedDoc, RetrievalResult

ROOT = Path(__file__).parent.parent


# ---------- metrics ----------

def test_recall_hit_mrr_ndcg_basic():
    ranked = ["b", "a", "c", "d"]
    assert recall_at_k(ranked, ["a", "z"], 3) == 0.5
    assert hit_at_k(ranked, ["a"], 1) == 0.0 and hit_at_k(ranked, ["a"], 2) == 1.0
    assert mrr(ranked, ["a"]) == 0.5 and mrr(ranked, ["zzz"]) == 0.0
    assert ndcg_at_k(["a"], ["a"], 5) == 1.0
    assert ndcg_at_k(["x", "a"], ["a"], 5) == pytest.approx(1 / math.log2(3))


def test_metrics_nan_for_no_relevant_and_mean_skips_nan():
    assert math.isnan(recall_at_k(["a"], [], 3)) and math.isnan(mrr(["a"], []))
    assert mean([1.0, float("nan"), 0.0]) == 0.5 and math.isnan(mean([]))


# ---------- dataset ----------

def test_dataset_loads_and_expected_sources_exist_in_sample_corpus():
    cases = load_cases()
    assert len(cases) >= 30
    chunk_ids = {c["chunk_id"] for c in json.loads((ROOT / "corpus/processed/sample_chunks.json").read_text(encoding="utf-8"))}
    for c in cases:
        for src in c.expected_source:
            assert src in chunk_ids, f"{c.id}: unknown chunk {src}"
        assert c.expect_refusal == (not c.expected_source) or c.id == "Q073", c.id
    assert {c.category for c in cases} >= {"alarm_definition", "normal_value", "out_of_scope", "follow_up", "history"}


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"id": "X", "question": "q", "expected_intent": "knowledge"}] * 2), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(p)


# ---------- retrieval evaluator with fakes ----------

def _doc(cid, rank):
    return RetrievedDoc(chunk_id=cid, text=cid, score=1.0, rank=rank, source="hybrid", metadata={})


class _FakeHybrid:
    def __init__(self): self.queries = []
    def retrieve(self, q, filters=None, top_k=None):
        self.queries.append(q)
        ids = ["VAC_203_001", "X", "Y"] if "VAC-203" in q.upper().replace("VAC203", "VAC-203") else ["X", "Y", "VAC_203_001"]
        return RetrievalResult(query=q, docs=[_doc(c, i) for i, c in enumerate(ids, 1)])


class _FakeRP:
    def __init__(self): self.hybrid = _FakeHybrid(); self.reranker = None


def test_retrieval_evaluator_scores_and_contextualizes_follow_ups():
    rp = _FakeRP()
    cases = [
        EvalCase(id="a", question="VAC203이 뭐야?", expected_intent="knowledge", expected_alarm_code="VAC-203", expected_source=["VAC_203_001"]),
        EvalCase(id="b", question="캘리브레이션?", expected_intent="knowledge", expected_source=["VAC_203_001"]),
        EvalCase(id="c", question="IP 주소?", expected_intent="general", expected_source=[]),
        EvalCase(id="d", history=["VAC-203이 뭐야?"], question="그 알람 원인?", expected_intent="knowledge", expected_alarm_code="VAC-203", expected_source=["VAC_203_001"]),
    ]
    cases.append(EvalCase(id="e", question="정답 위치 모름", expected_intent="knowledge", expected_source=None))
    res = RetrievalEvaluator(rp).run(cases)
    s = res["summary"]
    assert s["scored"] == 3 and s["out_of_scope"] == 1 and s["unknown_source"] == 1
    assert s["final_hit@1"] == pytest.approx(2 / 3) and s["final_hit@3"] == 1.0
    assert s["alarm_parsing_accuracy"] == 1.0            # a: parsed, b/c: None==None, d: carried
    assert rp.hybrid.queries[3] == "그 알람 원인? VAC-203"      # case "d" (follow-up)
    assert s["misses_at_5"] == []


# ---------- report gate ----------

def test_gate_fails_on_drop_beyond_tolerance_and_passes_without_previous():
    now = {"summary": {"final_hit@5": 0.9, "final_recall@5": 0.9, "final_mrr": 0.85}}
    prev = {"summary": {"final_hit@5": 1.0, "final_recall@5": 0.9, "final_mrr": 0.86}}
    ok, problems = report.gate("retrieval", now, prev)
    assert not ok and problems and "final_hit@5" in problems[0] and len(problems) == 1
    assert report.gate("retrieval", now, None) == (True, [])
    assert "| final_hit@5 | 0.900 | 1.000 | -0.100 |" in report.markdown("retrieval", now, prev)
