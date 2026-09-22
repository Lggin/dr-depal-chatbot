"""ChatPipeline flow with fake retrieval and fake LLM (no models)."""

from app.llm.client import GenerationResult
from app.llm.generator import ChatPipeline
from app.llm.prompt import ContextBuilder, PromptBuilder
from app.memory.session import SessionStore
from app.schemas.retrieval import RetrievedDoc, RetrievalResult


def doc(cid, text, **meta):
    return RetrievedDoc(chunk_id=cid, text=text, score=1.0, rank=1, source="exact", metadata=meta)


MANUAL = doc("VAC_203_001", "[Alarm] VAC-203 [Normal Range] -60 ~ -80 kPa",
             document_name="Vacuum System Manual", section="6.3", alarm_code="VAC-203", document_type="alarm_manual", trust_level=100)


class FakeRetrieval:
    reranker = object()
    def __init__(self, docs): self.docs = docs; self.queries = []
    def search(self, q, filters=None, top_k=None):
        self.queries.append(q)
        return RetrievalResult(query=q, docs=self.docs, alarm_codes=["VAC-203"] if "VAC" in q.upper() else [])


class FakeLLM:
    def __init__(self, text): self.text = text; self.calls = []
    def generate(self, messages, **kw):
        self.calls.append(messages)
        return GenerationResult(text=self.text, prompt_tokens=10, completion_tokens=5, latency_ms=1.0)
    def count_tokens(self, s): return len(s)


def make(text, docs):
    llm = FakeLLM(text)
    pipe = ChatPipeline(FakeRetrieval(docs), PromptBuilder(ContextBuilder(len), history_turns=1), llm, SessionStore(max_turns=1))
    return pipe, llm


def test_grounded_answer_gets_canonical_sources_and_ok_validation():
    pipe, _ = make("정상 범위는 -60 ~ -80 kPa 입니다.\n출처: Vacuum System Manual §6.3 [Alarm] VAC-203", [MANUAL])
    r = pipe.answer("s1", "VAC-203 정상 범위?")
    assert r.answer.endswith("출처: Vacuum System Manual §6.3 (VAC-203)")
    assert [s.chunk_id for s in r.sources] == ["VAC_203_001"]
    assert r.validation.ok and not r.validation.is_refusal
    assert r.alarm_codes == ["VAC-203"] and r.intent == "knowledge"
    assert set(r.timing_ms) == {"retrieval", "generation", "validation"}


def test_refusal_has_no_sources_and_is_flagged():
    pipe, _ = make("현재 제공된 문서에서는 해당 내용을 확인할 수 없습니다.", [MANUAL])
    r = pipe.answer("s1", "로봇 IP?")
    assert r.validation.is_refusal and "출처" not in r.answer
    assert r.sources == [] and r.retrieval.top_k == 1  # retrieval info still reports what was provided


def test_anaphoric_follow_up_carries_previous_alarm_code():
    pipe, llm = make("답변", [MANUAL])
    pipe.answer("f", "VAC-203이 뭐야?")
    r = pipe.answer("f", "그 알람 원인이 뭐야?")
    assert pipe.retrieval.queries[-1] == "그 알람 원인이 뭐야? VAC-203"
    assert "(이전 대화에서 언급된 알람: VAC-203)" in llm.calls[-1][-1]["content"]
    assert r.alarm_codes == ["VAC-203"] and r.message == "그 알람 원인이 뭐야?"


def test_follow_up_with_its_own_code_or_without_anaphora_is_untouched():
    pipe, _ = make("답변", [MANUAL])
    pipe.answer("g", "VAC-203이 뭐야?")
    pipe.answer("g", "ROB-105는 뭐야?")
    pipe.answer("g", "캘리브레이션 방법 알려줘")
    assert pipe.retrieval.queries[1:] == ["ROB-105는 뭐야?", "캘리브레이션 방법 알려줘"]


def test_unsupported_number_is_flagged_not_blocked():
    pipe, _ = make("정상 범위는 -55 kPa 입니다.", [MANUAL])
    r = pipe.answer("s1", "VAC-203?")
    assert r.validation.unsupported_numbers == ["-55"] and not r.validation.ok
    assert "-55" in r.answer


def test_dangerous_instruction_appends_notice():
    pipe, _ = make("PLC를 리셋하세요.", [MANUAL])
    r = pipe.answer("s1", "어떻게 해?")
    assert r.validation.danger_flags == ["plc reset"] and "담당 엔지니어" in r.answer


def test_session_history_is_passed_on_second_turn_and_capped():
    pipe, llm = make("답변", [MANUAL])
    pipe.answer("s2", "첫 질문")
    pipe.answer("s2", "둘째 질문")
    pipe.answer("s2", "셋째 질문")
    third_msgs = llm.calls[2]
    roles = [m["role"] for m in third_msgs]
    assert roles == ["system", "user", "assistant", "user"]      # exactly one prior turn
    assert third_msgs[1]["content"] == "둘째 질문"
    assert pipe.sessions.get("s2").turns == 3


def test_sessions_are_isolated():
    pipe, llm = make("답변", [MANUAL])
    pipe.answer("a", "q-a")
    pipe.answer("b", "q-b")
    assert [m["role"] for m in llm.calls[1]] == ["system", "user"]
