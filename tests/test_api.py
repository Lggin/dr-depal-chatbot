"""API tests with a fake pipeline injected — no models, in-memory SQLite."""

import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.config import get_settings  # noqa: E402
import app.config as config_module    # noqa: E402
config_module.settings = get_settings()  # re-read env for this test module

from app.main import app  # noqa: E402
from app.memory.session import SessionStore  # noqa: E402
from app.schemas.chat import ChatResponse, DocumentSource, IntentType, RetrievalInfo, ValidationInfo  # noqa: E402


class _FakeStore:
    def get_collection_info(self): return {"name": "t", "points_count": 2}
    def scroll_all(self):
        return [{"chunk_id": "a", "text": "x", "metadata": {"document_name": "M", "document_type": "manual"}},
                {"chunk_id": "b", "text": "y", "metadata": {"document_name": "M", "document_type": "manual"}}]


class _FakePipeline:
    tools = None
    mock = None

    def __init__(self):
        self.sessions = SessionStore(max_turns=1)
        self.llm = type("L", (), {"llm": object()})()
        self.retrieval = type("R", (), {"hybrid": type("H", (), {"store": _FakeStore()})(), "reranker": object()})()
        self.calls = []

    def reindex(self, config_path=None):
        if config_path == "missing.yaml":
            raise FileNotFoundError(config_path)
        return {"collection": "t", "chunks": 5, "per_source": [{"source": "a"}, {"source": "b"}], "bm25_chunks": 5}

    def answer(self, session_id, question, filters=None, machine_state=None):
        self.calls.append((session_id, question))
        self.sessions.record(session_id, question, "답변")
        src = DocumentSource(chunk_id="a", document="M", section="1", trust_level=100)
        return ChatResponse(
            session_id=session_id, message=question, answer="답변\n\n출처: M §1", intent=IntentType.KNOWLEDGE,
            sources=[src], retrieval=RetrievalInfo(documents_found=1, reranked=True, top_k=1, sources=[src], scores=[0.5]),
            validation=ValidationInfo(ok=True, is_refusal=False), alarm_codes=[], timing_ms={"retrieval": 1.0}, latency_ms=2.0,
        )

    def close(self): pass


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def loaded(client):
    client.app.state.pipeline = _FakePipeline()
    yield client
    client.app.state.pipeline = None


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_503_without_models_then_200_with(client):
    assert client.get("/api/ready").status_code == 503
    client.app.state.pipeline = _FakePipeline()
    r = client.get("/api/ready")
    assert r.status_code == 200 and all(r.json()["services"].values())


def test_chat_503_when_not_loaded(client):
    assert client.post("/api/chat", json={"session_id": "s", "message": "hi"}).status_code == 503


def test_chat_returns_response_and_logs(loaded):
    r = loaded.post("/api/chat", json={"session_id": "s1", "message": "VAC-203이 뭐야?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("답변") and body["intent"] == "knowledge"
    assert body["sources"][0]["chunk_id"] == "a" and body["validation"]["ok"] is True
    assert isinstance(body["log_id"], int)
    assert loaded.app.state.pipeline.calls == [("s1", "VAC-203이 뭐야?")]

    logs = loaded.get("/api/admin/logs?limit=5").json()
    assert logs[0]["id"] == body["log_id"] and logs[0]["question"] == "VAC-203이 뭐야?"
    assert logs[0]["retrieved_chunk_ids"] == ["a"]


def test_chat_validation_error_on_missing_fields(loaded):
    assert loaded.post("/api/chat", json={"message": "no session"}).status_code == 422


def test_feedback_roundtrip_and_404(loaded):
    log_id = loaded.post("/api/chat", json={"session_id": "s2", "message": "q"}).json()["log_id"]
    r = loaded.post("/api/feedback", json={"log_id": log_id, "feedback": "down", "note": "wrong range"})
    assert r.status_code == 200 and r.json()["feedback"] == "down"
    assert loaded.get("/api/admin/logs?session_id=s2").json()[0]["user_feedback"] == "down"
    assert loaded.post("/api/feedback", json={"log_id": 999999, "feedback": "up"}).status_code == 404
    assert loaded.post("/api/feedback", json={"log_id": log_id, "feedback": "meh"}).status_code == 422


def test_session_lifecycle(loaded):
    r = loaded.post("/api/session")
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert loaded.get(f"/api/session/{sid}").json()["turns"] == 0
    loaded.post("/api/chat", json={"session_id": sid, "message": "q"})
    assert loaded.get(f"/api/session/{sid}").json()["turns"] == 1
    assert loaded.delete(f"/api/session/{sid}").status_code == 204
    assert loaded.get(f"/api/session/{sid}").status_code == 404
    assert loaded.delete(f"/api/session/{sid}").status_code == 404


def test_admin_documents(loaded):
    r = loaded.get("/api/admin/documents")
    assert r.status_code == 200 and r.json()["chunks"] == 2 and r.json()["documents"] == [{"document": "M", "chunks": 2}]


def test_admin_reindex_and_ingest(loaded):
    r = loaded.post("/api/admin/reindex")
    assert r.status_code == 200
    assert r.json()["chunks"] == 5 and r.json()["sources"] == 2 and "per_source" not in r.json()
    assert loaded.post("/api/admin/ingest").status_code == 200
    assert loaded.post("/api/admin/reindex?config_path=missing.yaml").status_code == 404


def test_admin_tools_endpoints_with_mock_backend(loaded):
    from pathlib import Path
    from app.tools.mock import build_mock_registry
    reg, mock = build_mock_registry(Path(__file__).parent.parent / "configs/machine.yaml")
    loaded.app.state.pipeline.tools, loaded.app.state.pipeline.mock = reg, mock
    t = loaded.get("/api/admin/tools").json()
    assert t["backend"] == "mock" and t["read_only"] and "get_vacuum_pressure" in [x["name"] for x in t["tools"]]
    m = loaded.get("/api/admin/machine").json()
    assert m["machine_state"]["vacuum_pressure"] == -43 and "get_current_alarm" in m["tools_used"]
    s = loaded.post("/api/admin/mock/scenario?name=normal").json()
    assert s["scenario"] == "normal" and loaded.get("/api/admin/machine").json()["machine_state"]["alarm"] is None
    assert loaded.post("/api/admin/mock/scenario?name=nope").status_code == 404


def test_admin_tools_without_backend(loaded):
    loaded.app.state.pipeline.tools = None
    assert loaded.get("/api/admin/tools").json()["tools"] == []
    assert loaded.get("/api/admin/machine").status_code == 404


def test_admin_reindex_503_without_models(client):
    assert client.post("/api/admin/reindex").status_code == 503
