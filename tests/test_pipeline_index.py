import numpy as np
import pytest

from ingestion.pipeline import IngestionPipeline


def test_token_batches_respect_budget_and_count():
    lengths = [100] * 20 + [2500, 2500, 2500] + [50] * 5
    batches = IngestionPipeline.token_batches(lengths, max_items=16, token_budget=6000)
    assert batches[0] == slice(0, 16)                       # count cap
    assert batches[1] == slice(16, 22)                      # 4×100 + 2×2500 = 5400 ≤ 6000; a 3rd 2500 would exceed
    assert batches[2] == slice(22, 28)                      # 2500 + 5×50
    assert sum(b.stop - b.start for b in batches) == len(lengths)
    assert all(sum(lengths[b]) <= 6000 or b.stop - b.start == 1 for b in batches)


def test_token_batches_single_oversize_item_gets_its_own_batch():
    assert IngestionPipeline.token_batches([9000, 10], token_budget=6000) == [slice(0, 1), slice(1, 2)]
    assert IngestionPipeline.token_batches([]) == []


class _Emb:
    embedding_dim = 4
    def count_tokens(self, s): return len(s)
    def embed(self, texts, batch_size=32):
        return np.ones((len(texts), 4), dtype=np.float32)


class _Store:
    collection_name = "t"
    def __init__(self, lose_last=False): self.points = 0; self.lose_last = lose_last; self.calls = []
    def create_collection(self, force=False): self.points = 0
    def upsert_documents(self, texts, embeddings, metadata, chunk_ids):
        self.calls.append(len(chunk_ids)); self.points += len(chunk_ids)
    def count(self): return self.points - (1 if self.lose_last else 0)


def _data(n):
    texts = [f"t{i}" for i in range(n)]
    metas = [{"search_text": "x" * 100} for _ in range(n)]
    return texts, metas, [f"c{i}" for i in range(n)]


def test_index_verifies_point_count():
    texts, metas, ids = _data(40)
    store = _Store()
    assert IngestionPipeline.index(texts, metas, ids, _Emb(), store) == 40
    assert sum(store.calls) == 40 and max(store.calls) <= 16


def test_index_raises_on_silent_loss():
    texts, metas, ids = _data(5)
    with pytest.raises(RuntimeError, match="index verification failed"):
        IngestionPipeline.index(texts, metas, ids, _Emb(), _Store(lose_last=True))
