"""ChatPipeline intent routing + multi-turn troubleshooting with fakes (no models)."""

from pathlib import Path

import pytest

from app.llm.client import GenerationResult
from app.llm.generator import GENERAL_HELP, LIVE_UNAVAILABLE, ChatPipeline
from app.llm.prompt import ContextBuilder, PromptBuilder
from app.memory.session import SessionStore
from app.router.intent_router import IntentRouter
from app.schemas.retrieval import RetrievedDoc, RetrievalResult
from app.troubleshooting import TroubleshootingEngine, load_rules

RULES = Path(__file__).parent.parent / "configs/troubleshooting.yaml"


def doc(cid, text, **meta):
    return RetrievedDoc(chunk_id=cid, text=text, score=1.0, rank=1, source="rerank", metadata=meta)


MANUAL = doc("VAC_203_001", "[Alarm] VAC-203 [Normal Range] -60 ~ -80 kPa. hose 누설, suction cup 손상",
             document_name="Vacuum System Manual", section="6.3", alarm_code="VAC-203", document_type="alarm_manual", trust_level=100)
LOG = doc("MAINT_1", "2026-09-10 VAC-203 발생, 호스 교체", document_name="Maintenance Log", document_type="maintenance", trust_level=70)


class FakeRetrieval:
    reranker = object()
    def __init__(self, docs_by_filter=None, default=None):
        self.default = default or [MANUAL]; self.by_filter = docs_by_filter or {}; self.calls = []
    def search(self, q, filters=None, top_k=None):
        self.calls.append((q, filters))
        key = (filters or {}).get("document_type")
        docs = self.by_filter.get(key, self.default) if key else self.default
        return RetrievalResult(query=q, docs=docs, alarm_codes=["VAC-203"] if "VAC" in q.upper() else [])


class FakeLLM:
    def __init__(self, text="답변입니다. -60 ~ -80 kPa\n출처: Vacuum System Manual §6.3"): self.text = text; self.calls = []
    def generate(self, messages, **kw):
        self.calls.append(messages)
        return GenerationResult(text=self.text, prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    def count_tokens(self, s): return len(s)


def make(retrieval=None, llm=None, tools=None):
    llm = llm or FakeLLM()
    retrieval = retrieval or FakeRetrieval()
    pipe = ChatPipeline(retrieval, PromptBuilder(ContextBuilder(len), history_turns=1), llm, SessionStore(max_turns=1),
                        router=IntentRouter(), engine=TroubleshootingEngine(load_rules(RULES)), tools=tools)
    return pipe, retrieval, llm


def make_with_mock(llm=None, scenario="vac203_low", retrieval=None):
    from app.tools.mock import build_mock_registry
    reg, mock = build_mock_registry(Path(__file__).parent.parent / "configs/machine.yaml")
    mock.set_scenario(scenario)
    pipe, ret, llm = make(retrieval=retrieval, llm=llm, tools=reg)
    return pipe, ret, llm, mock


# ---------- Phase 4: tools ----------

def test_live_status_with_mock_tools_answers_from_machine_state():
    pipe, ret, llm, _ = make_with_mock(llm=FakeLLM("현재 진공압은 -43 kPa 입니다."))
    r = pipe.answer("s", "지금 진공압 얼마야?")
    assert r.route == "live_status.tools" and not r.validation.is_refusal
    assert "vacuum_pressure: -43" in llm.calls[0][-1]["content"] and "get_vacuum_pressure" in r.tools_used
    assert r.machine_state["alarm"] == "VAC-203" and r.validation.ok          # -43 is evidenced by the machine state


def test_live_diagnosis_with_tools_has_no_unavailable_notice():
    pipe, ret, llm, _ = make_with_mock(llm=FakeLLM("VAC-203, -43 kPa. 기준 -60 ~ -80 kPa. 진공 계통 우선.\n출처: Vacuum System Manual §6.3"))
    r = pipe.answer("s", "지금 박스를 왜 못 집어?")
    assert r.route == "live_diagnosis.tools" and not r.answer.startswith("⚠️")
    assert "[MACHINE STATE]" in llm.calls[0][-1]["content"] and "정상으로 확인된 계통은 제외" in llm.calls[0][-1]["content"]
    assert r.tools_used and r.sources


def test_tree_prefilled_from_tools_asks_only_unknown_fact_then_recommends():
    pipe, ret, llm, _ = make_with_mock(llm=FakeLLM("hose → cup → generator\n출처: Vacuum System Manual §6.3"))
    r1 = pipe.answer("t", "박스를 못 집어.")
    assert r1.route == "troubleshooting.ask" and "로봇" in r1.answer and "비전" not in r1.answer.split("\n\n")[-1]
    assert "자동 확인" in r1.answer and "vacuum_pressure=-43" in r1.answer
    assert r1.troubleshooting["facts"] == {"vision_detection": True, "vacuum_ok": False} and r1.tools_used
    r2 = pipe.answer("t", "응")
    assert r2.route == "troubleshooting.recommend" and r2.troubleshooting["current_step"] == "check_vacuum"
    assert "자동 확인" in llm.calls[0][-1]["content"]


def test_history_includes_maintenance_records_from_tool():
    ret = FakeRetrieval(docs_by_filter={"maintenance": [LOG]})
    pipe, _, llm, _ = make_with_mock(retrieval=ret, llm=FakeLLM("2026-09-10 호스 교체.\n출처: Maintenance Log"))
    r = pipe.answer("h", "VAC-203 지난번에도 있었어?")
    user_msg = llm.calls[0][-1]["content"]
    assert "[MAINTENANCE HISTORY]" in user_msg and "2026-09-10" in user_msg and "vacuum hose 교체" in user_msg
    assert "get_maintenance_history" in r.tools_used


def test_caller_supplied_machine_state_overrides_tools():
    pipe, ret, llm, _ = make_with_mock(llm=FakeLLM("현재 진공압은 -71 kPa 입니다."))
    r = pipe.answer("s", "지금 진공압 얼마야?", machine_state={"vacuum_pressure": -71})
    assert r.tools_used == [] and r.machine_state == {"vacuum_pressure": -71} and r.route == "live_status.state"


# ---------- deterministic routes ----------

def test_general_and_live_status_skip_retrieval_and_llm():
    pipe, ret, llm = make()
    g = pipe.answer("s", "안녕")
    l = pipe.answer("s", "지금 진공압 얼마야?")
    assert g.route == "general" and g.answer == GENERAL_HELP and g.intent == "general"
    assert l.route == "live.no_tools" and l.answer == LIVE_UNAVAILABLE and l.intent == "live_status" and l.sources == []
    assert ret.calls == [] and llm.calls == []


def test_live_status_with_machine_state_goes_through_llm():
    pipe, ret, llm = make()
    r = pipe.answer("s", "지금 진공압 얼마야?", machine_state={"vacuum_pressure": -43})
    assert r.route == "live_status.state" and llm.calls and "[MACHINE STATE]" in llm.calls[0][-1]["content"]


def test_live_diagnosis_without_tools_adds_notice_and_uses_documents():
    pipe, ret, llm = make()
    r = pipe.answer("s", "지금 박스를 왜 못 집어?")
    assert r.intent == "live_diagnosis" and r.route == "live_diagnosis.no_tools"
    assert r.answer.startswith("⚠️ 현재 설비 값은 조회 도구 미연결") and "-60" in r.answer
    assert "현재 설비 값은 조회할 수 없습니다" in llm.calls[0][-1]["content"]


# ---------- history ----------

def test_history_filters_maintenance_first():
    ret = FakeRetrieval(docs_by_filter={"maintenance": [LOG]})
    pipe, _, llm = make(retrieval=ret, llm=FakeLLM("2026-09-10 호스 교체 이력이 있습니다.\n출처: Maintenance Log"))
    r = pipe.answer("s", "VAC-203 지난번에도 있었어?")
    assert r.intent == "history" and r.route == "history.filtered"
    assert ret.calls[0][1] == {"document_type": "maintenance"}
    assert [s.chunk_id for s in r.sources] == ["MAINT_1"]
    assert "이력 질문" in llm.calls[0][-1]["content"]


def test_history_falls_back_to_unfiltered_when_no_records():
    ret = FakeRetrieval(docs_by_filter={"maintenance": []})
    pipe, _, _ = make(retrieval=ret)
    r = pipe.answer("s", "ROB-105 전에도 발생했어?")
    assert r.route == "history.unfiltered" and len(ret.calls) == 2 and ret.calls[1][1] is None


# ---------- troubleshooting multi-turn (spec §20) ----------

def test_troubleshooting_dialogue_asks_then_recommends_then_resolves():
    pipe, ret, llm = make(llm=FakeLLM("1. hose 누설 확인 2. suction cup 확인\n출처: Vacuum System Manual §6.3"))
    r1 = pipe.answer("t", "박스를 못 집어.")
    assert r1.intent == "troubleshooting" and r1.route == "troubleshooting.ask"
    assert "비전" in r1.answer and r1.troubleshooting["issue"] == "pickup_failure"
    assert ret.calls == [] and llm.calls == []                       # asking is deterministic

    r2 = pipe.answer("t", "응")
    assert r2.route == "troubleshooting.ask" and "로봇" in r2.answer and r2.troubleshooting["facts"] == {"vision_detection": True}

    r3 = pipe.answer("t", "응")
    assert r3.route == "troubleshooting.ask" and "진공압" in r3.answer

    r4 = pipe.answer("t", "아니 안 올라가")
    assert r4.route == "troubleshooting.recommend"
    assert ret.calls and "VAC-203" in ret.calls[0][0]                 # step's doc_query drove retrieval
    assert "[TROUBLESHOOTING STATE]" in llm.calls[0][-1]["content"] and "check_vacuum" in llm.calls[0][-1]["content"]
    assert "hose" in r4.answer and r4.answer.rstrip().endswith("답해 주시면 됩니다.")
    assert r4.troubleshooting["current_step"] == "check_vacuum" and r4.sources

    r5 = pipe.answer("t", "해결됐어 고마워")
    assert r5.route == "troubleshooting.resolved" and pipe.sessions.get("t").troubleshooting is None


def test_unclear_reply_reasks_and_new_topic_exits_diagnosis():
    pipe, ret, llm = make()
    pipe.answer("u", "빨판이 박스를 못 잡아")
    r = pipe.answer("u", "잘 모르겠어")
    assert r.route == "troubleshooting.ask" and "예/아니오" in r.answer and "비전" in r.answer
    k = pipe.answer("u", "VAC-203 정상 범위가 얼마야?")
    assert k.intent == "knowledge" and k.route == "knowledge" and pipe.sessions.get("u").troubleshooting is None


def test_negative_first_answer_recommends_immediately():
    pipe, ret, llm = make(llm=FakeLLM("카메라와 조명을 확인하세요.\n출처: Vacuum System Manual §6.3"))
    pipe.answer("n", "박스를 못 집어")
    r = pipe.answer("n", "아니, 검출이 안 돼")
    assert r.route == "troubleshooting.recommend" and r.troubleshooting["current_step"] == "check_vision"


def test_cause_question_about_a_symptom_answers_from_documents_not_the_tree():
    pipe, ret, llm = make()
    r = pipe.answer("c", "박스 흡착 실패 원인이 뭐야?")       # matches the pickup_failure trigger, but asks for an explanation
    assert r.intent == "troubleshooting" and r.route == "troubleshooting" and ret.calls and llm.calls
    assert pipe.sessions.get("c").troubleshooting is None


def test_live_no_tools_notice_counts_as_refusal():
    pipe, _, _ = make()
    assert pipe.answer("l", "지금 진공압 얼마야?").validation.is_refusal is True


def test_troubleshooting_intent_without_matching_issue_falls_back_to_rag():
    pipe, ret, llm = make()
    r = pipe.answer("f", "VIS-301 알람이 떴어. 뭘 확인해?")      # troubleshooting intent, no tree for it
    assert r.intent == "troubleshooting" and r.route == "troubleshooting" and ret.calls and llm.calls


def test_machine_state_prefills_facts():
    pipe, _, _ = make()
    r = pipe.answer("m", "박스를 못 집어", machine_state={"vision_detection": True})
    assert r.route == "troubleshooting.ask" and "로봇" in r.answer       # vision already known → asks robot next
