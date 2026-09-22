"""Pure-logic tests for RRF, gap cut, glossary expansion, exact pinning (no models)."""

import pytest

from app.rag.hybrid_retriever import HybridRetriever, rrf_fuse
from app.rag.query_expander import GlossaryExpander
from app.rag.reranker import gap_cut
from app.schemas.retrieval import RetrievedDoc, RetrievalResult


def d(cid, rank, source="dense", **meta):
    return RetrievedDoc(chunk_id=cid, text=cid, score=1.0, rank=rank, source=source, metadata=meta)


# ---------- RRF ----------

def test_rrf_doc_in_both_lists_beats_single_list_top():
    dense = [d("a", 1), d("b", 2)]
    sparse = [d("b", 1, "sparse"), d("c", 2, "sparse")]
    fused = rrf_fuse([dense, sparse], k=60)
    assert fused[0].chunk_id == "b"          # 1/61 + 1/62 > 1/61
    assert [x.rank for x in fused] == [1, 2, 3]
    assert all(x.source == "hybrid" for x in fused)


def test_rrf_recovers_doc_missing_from_one_list():
    fused = rrf_fuse([[d("only_dense", 2)], []])
    assert fused[0].chunk_id == "only_dense"


# ---------- gap cut ----------

def test_gap_cut_stops_at_first_big_drop_before_top_k():
    assert gap_cut([-2.5, -5.7, -6.0, -6.1], top_k=4, gap=3.0) == 1


def test_gap_cut_no_drop_uses_top_k_cap():
    assert gap_cut([-5.1, -5.4, -5.9, -6.3, -6.5, -7.0], top_k=4, gap=3.0) == 4


def test_gap_cut_sees_past_top_k():
    # drop happens at position 6 (> top_k): cap applies, gap rule didn't fire early
    scores = [1, 0.9, 0.8, 0.7, 0.6, 0.5, -9]
    assert gap_cut(scores, top_k=4, gap=3.0) == 4
    assert gap_cut(scores, top_k=10, gap=3.0) == 6


def test_gap_cut_empty():
    assert gap_cut([], top_k=4, gap=3.0) == 0


# ---------- glossary ----------

def test_glossary_appends_synonyms_and_keeps_original():
    g = GlossaryExpander({"빨판": ["suction cup", "vacuum pad"]})
    q, matched = g.expand_with_terms("빨판이 박스를 못 잡아")
    assert q.startswith("빨판이 박스를 못 잡아")
    assert "suction cup" in q and "vacuum pad" in q
    assert matched == ["빨판"]


def test_glossary_noop_when_no_match():
    g = GlossaryExpander({"빨판": ["suction cup"]})
    assert g.expand_with_terms("서보 과부하") == ("서보 과부하", [])


def test_glossary_does_not_duplicate_terms_already_in_query():
    g = GlossaryExpander({"비전": ["vision", "camera"]})
    q, _ = g.expand_with_terms("비전 camera 문제")
    assert q.count("camera") == 1


# ---------- exact pinning ----------

class _FakeRetriever:
    def __init__(self, docs): self._docs = docs
    def retrieve(self, q, top_k=None, filters=None):
        return RetrievalResult(query=q, docs=self._docs)


class _FakeStore:
    def __init__(self, rows): self.rows = rows
    def scroll_by_filter(self, filters, limit=20):
        return [r for r in self.rows if all(r["metadata"].get(k) == v for k, v in filters.items())]


def test_exact_alarm_hits_pinned_first_in_trust_order():
    store = _FakeStore([
        {"chunk_id": "note", "text": "n", "metadata": {"alarm_code": "VAC-203", "trust_level": 50}},
        {"chunk_id": "manual", "text": "m", "metadata": {"alarm_code": "VAC-203", "trust_level": 100}},
        {"chunk_id": "log", "text": "l", "metadata": {"alarm_code": "VAC-203", "trust_level": 70}},
    ])
    dense = _FakeRetriever([d("note", 1), d("other", 2)])
    sparse = _FakeRetriever([d("log", 1, "sparse")])
    h = HybridRetriever(dense, sparse, store, expander=None)
    res = h.retrieve("vac203 뭐야")
    assert res.alarm_codes == ["VAC-203"]
    assert res.chunk_ids()[:3] == ["manual", "log", "note"]
    assert res.chunk_ids()[3] == "other"
    assert res.exact_chunk_ids == ["manual", "log", "note"]
    assert [x.source for x in res.docs[:3]] == ["exact"] * 3


def test_no_code_means_no_exact_stage():
    h = HybridRetriever(_FakeRetriever([d("a", 1)]), _FakeRetriever([]), _FakeStore([]), expander=None)
    res = h.retrieve("빨판이 안 붙어")
    assert res.alarm_codes == [] and res.exact_chunk_ids == [] and res.chunk_ids() == ["a"]
