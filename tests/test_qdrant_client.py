"""QdrantClient unit tests — run against in-memory Qdrant, no Docker needed."""

import numpy as np
import pytest

from app.rag.qdrant_client import QdrantClient

DIM = 8


@pytest.fixture
def client():
    c = QdrantClient(url=":memory:", collection_name="test_docs", embedding_dim=DIM)
    c.connect()
    c.create_collection(force=True)
    return c


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_upsert_and_search_returns_nearest_first(client):
    e0 = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    e1 = _unit([0, 1, 0, 0, 0, 0, 0, 0])
    client.upsert_documents(
        texts=["doc zero", "doc one"],
        embeddings=np.stack([e0, e1]),
        metadata=[{"alarm_code": "VAC-203"}, {}],
        chunk_ids=["c0", "c1"],
    )
    hits = client.search(_unit([0.9, 0.1, 0, 0, 0, 0, 0, 0]), top_k=2)
    assert [h["chunk_id"] for h in hits] == ["c0", "c1"]
    assert hits[0]["metadata"]["alarm_code"] == "VAC-203"
    assert "text" not in hits[0]["metadata"]


def test_reingest_same_chunk_id_overwrites_not_duplicates(client):
    e = _unit(np.ones(DIM))
    client.upsert_documents(["v1"], np.stack([e]), chunk_ids=["same"])
    client.upsert_documents(["v2"], np.stack([e]), chunk_ids=["same"])
    assert client.get_collection_info()["points_count"] == 1
    assert client.search(e, top_k=1)[0]["text"] == "v2"


def test_length_mismatch_rejected(client):
    with pytest.raises(ValueError):
        client.upsert_documents(["a", "b"], np.zeros((1, DIM)))


def test_create_collection_is_idempotent_without_force(client):
    client.upsert_documents(["x"], np.stack([_unit(np.ones(DIM))]), chunk_ids=["x"])
    client.create_collection(force=False)
    assert client.get_collection_info()["points_count"] == 1
