"""DenseRetriever + metadata filter tests with a fake embedder and in-memory Qdrant."""

import numpy as np
import pytest

from app.rag.dense_retriever import DenseRetriever
from app.rag.metadata_filter import build_filter
from app.rag.qdrant_client import QdrantClient

DIM = 4


class FakeEmbedder:
    """Maps known strings to fixed unit vectors so ranking is deterministic."""
    table = {
        "vacuum": [1, 0, 0, 0],
        "robot":  [0, 1, 0, 0],
        "vision": [0, 0, 1, 0],
        "mostly vacuum": [0.9, 0.1, 0, 0],
    }
    embedding_dim = DIM

    def embed_single(self, text):
        v = np.asarray(self.table[text], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed(self, texts, batch_size=32):
        return np.stack([self.embed_single(t) for t in texts])


@pytest.fixture
def retriever():
    emb = FakeEmbedder()
    store = QdrantClient(url=":memory:", collection_name="t", embedding_dim=DIM)
    store.connect()
    store.create_collection(force=True)
    texts = ["vacuum", "robot", "vision"]
    store.upsert_documents(
        texts=texts,
        embeddings=emb.embed(texts),
        metadata=[
            {"alarm_code": "VAC-203", "document_type": "alarm_manual", "trust_level": 100},
            {"alarm_code": "ROB-105", "document_type": "alarm_manual", "trust_level": 100},
            {"alarm_code": None,      "document_type": "maintenance",  "trust_level": 70},
        ],
        chunk_ids=["vac", "rob", "vis"],
    )
    return DenseRetriever(emb, store, top_k=3)


def test_ranks_by_similarity_with_1_based_rank(retriever):
    res = retriever.retrieve("mostly vacuum")
    assert res.chunk_ids() == ["vac", "rob", "vis"]
    assert [d.rank for d in res.docs] == [1, 2, 3]
    assert all(d.source == "dense" for d in res.docs)
    assert res.docs[0].alarm_code == "VAC-203"


def test_exact_filter_restricts_candidates(retriever):
    res = retriever.retrieve("mostly vacuum", filters={"alarm_code": "ROB-105"})
    assert res.chunk_ids() == ["rob"]


def test_list_filter_is_match_any(retriever):
    res = retriever.retrieve("vacuum", filters={"document_type": ["maintenance"]})
    assert res.chunk_ids() == ["vis"]


def test_range_filter(retriever):
    res = retriever.retrieve("vacuum", filters={"trust_level": {"gte": 85}})
    assert set(res.chunk_ids()) == {"vac", "rob"}


def test_top_k_override(retriever):
    assert len(retriever.retrieve("vacuum", top_k=1).docs) == 1


def test_build_filter_ignores_none_and_empty():
    assert build_filter(None) is None
    assert build_filter({}) is None
    assert build_filter({"alarm_code": None}) is None
