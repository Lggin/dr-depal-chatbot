import pytest

from app.rag.sparse_retriever import SparseRetriever, _matches
from app.rag.tokenizer import KoreanTokenizer


@pytest.fixture(scope="module")
def tok():
    return KoreanTokenizer()


def test_tokenizer_keeps_alarm_code_whole_and_normalized(tok):
    toks = tok.tokenize("VAC203 알람이 떴는데")
    assert "vac-203" in toks
    assert "203" not in toks and "vac" not in toks


def test_tokenizer_drops_particles_and_endings(tok):
    toks = tok.tokenize("빨판이 박스를 못 잡아요")
    assert "빨판" in toks and "박스" in toks and "잡" in toks
    assert "이" not in toks and "를" not in toks and "어요" not in toks


def test_tokenizer_lowercases_english(tok):
    assert "suction" in tok.tokenize("Suction cup 손상")


CHUNKS = [
    {"chunk_id": "vac", "text": "[Alarm] VAC-203 진공압이 낮음. suction cup 점검",
     "metadata": {"alarm_code": "VAC-203", "document_type": "alarm_manual", "trust_level": 100}},
    {"chunk_id": "rob", "text": "[Alarm] ROB-105 서보 과부하. payload 확인",
     "metadata": {"alarm_code": "ROB-105", "document_type": "alarm_manual", "trust_level": 100}},
    {"chunk_id": "log", "text": "2026-09-10 VAC-203 발생, 호스 교체 후 정상",
     "metadata": {"alarm_code": "VAC-203", "document_type": "maintenance", "trust_level": 70}},
]


@pytest.fixture(scope="module")
def sparse(tok):
    return SparseRetriever(tok, top_k=10).build(CHUNKS)


def test_unnormalized_query_code_matches_normalized_doc_code(sparse):
    ids = sparse.retrieve("vac203 뭐야").chunk_ids()
    assert set(ids) == {"vac", "log"}
    assert "rob" not in ids


def test_zero_score_docs_are_dropped(sparse):
    assert sparse.retrieve("서보").chunk_ids() == ["rob"]


def test_no_hits_returns_empty(sparse):
    assert sparse.retrieve("완전히 무관한 문장").docs == []


def test_filter_applied_after_scoring(sparse):
    assert sparse.retrieve("VAC-203", filters={"document_type": "maintenance"}).chunk_ids() == ["log"]
    assert sparse.retrieve("VAC-203", filters={"trust_level": {"gte": 85}}).chunk_ids() == ["vac"]


def test_matches_semantics():
    m = {"a": 5, "t": "x"}
    assert _matches(m, None) and _matches(m, {"a": 5}) and _matches(m, {"t": ["x", "y"]})
    assert _matches(m, {"a": {"gte": 5, "lt": 6}}) and not _matches(m, {"a": {"gt": 5}})
    assert not _matches(m, {"missing": {"gte": 1}}) and _matches(m, {"missing": None})


def test_retrieve_before_build_raises():
    with pytest.raises(RuntimeError):
        SparseRetriever().retrieve("x")
