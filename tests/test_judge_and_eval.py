"""Judge parsing/aggregation and Phase 5 evaluator fields with fakes (no model)."""

import json

from app.llm.client import GenerationResult
from evaluation.answer_eval import AnswerEvaluator
from evaluation.dataset import EvalCase
from evaluation.judge import AnswerJudge, parse_json_block
from evaluation.generate_questions import seed_must_include, select_chunks


class _LLM:
    def __init__(self, text): self.text = text; self.calls = []
    def generate(self, messages, **kw):
        self.calls.append((messages, kw))
        return GenerationResult(text=self.text, prompt_tokens=1, completion_tokens=1, latency_ms=1)


CTX = [{"chunk_id": "a", "text": "정상 -60 ~ -80 kPa", "metadata": {"document_name": "M", "section": "6.3"}},
       {"chunk_id": "b", "text": "호스 교체", "metadata": {"document_name": "Log"}}]


def test_parse_json_block_tolerates_fences_prose_and_trailing_commas():
    assert parse_json_block('```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_json_block('Here you go: {"sentences": [], "x": [1,2,]} thanks') == {"sentences": [], "x": [1, 2]}
    assert parse_json_block("no json here") is None and parse_json_block("") is None


def test_judge_groundedness_completeness_and_used_sources():
    verdict = {"sentences": [{"text": "정상 범위는 -60 ~ -80 kPa", "supported": True, "evidence": "1"},
                             {"text": "차이는 20 kPa", "supported": False, "evidence": ""}],
               "used_sources": [1], "contradicts_reference": False, "missing_from_reference": ["호스 교체 사례"], "notes": "ok"}
    llm = _LLM(json.dumps(verdict))
    j = AnswerJudge(llm).judge("q", "a", CTX, reference="ref", machine_state={"vacuum_pressure": -43})
    assert j.ok and j.groundedness == 0.5 and j.unsupported == ["차이는 20 kPa"] and j.used_sources == [1]
    assert j.completeness == 0.5 and j.contradicts_reference is False
    user = llm.calls[0][0][-1]["content"]
    assert "[SOURCE 1] M / 6.3" in user and "MACHINE STATE" in user and "REFERENCE" in user
    assert llm.calls[0][1]["temperature"] == 0.0


def test_judge_runs_on_base_model_when_answering_llm_has_lora():
    """A/B runs: the adapter is switched off for the judgement and back on afterwards (order matters)."""
    class _LoraLLM(_LLM):
        def __init__(self, text): super().__init__(text); self.events = []
        def set_lora_enabled(self, enabled):
            self.events.append(("lora", enabled)); return True
        def generate(self, messages, **kw):
            self.events.append(("generate",)); return super().generate(messages, **kw)
    llm = _LoraLLM(json.dumps({"sentences": [{"text": "x", "supported": True}], "used_sources": [1]}))
    assert AnswerJudge(llm).judge("q", "a", CTX).ok
    assert llm.events == [("lora", False), ("generate",), ("lora", True)]
    # an LLM without an adapter is left alone
    plain = _LLM(json.dumps({"sentences": [], "used_sources": []}))
    assert AnswerJudge(plain).judge("q", "a", CTX).ok and len(plain.calls) == 1


def test_judge_strips_citation_line_before_and_after():
    verdict = {"sentences": [{"text": "정상 범위는 -60 ~ -80 kPa", "supported": True},
                             {"text": "출처: Vacuum System Manual §6.3", "supported": False}], "used_sources": [1]}
    llm = _LLM(json.dumps(verdict))
    j = AnswerJudge(llm).judge("q", "정상 범위는 -60 ~ -80 kPa\n\n출처: Vacuum System Manual §6.3", CTX)
    assert "출처:" not in llm.calls[0][0][-1]["content"].split("ANSWER TO JUDGE:")[1]
    assert j.groundedness == 1.0 and j.unsupported == []          # citation "sentence" from the model is dropped


def test_judge_unparseable_and_no_reference():
    assert not AnswerJudge(_LLM("nonsense")).judge("q", "a", CTX).ok
    j = AnswerJudge(_LLM(json.dumps({"sentences": [{"text": "x", "supported": True}], "used_sources": []}))).judge("q", "a", CTX)
    assert j.ok and j.groundedness == 1.0 and j.completeness is None


# ---------- evaluator: tool selection + judge aggregation ----------

class _Resp:
    def __init__(self, tools_used, route="knowledge", sources=("a",)):
        from app.schemas.chat import ChatResponse, DocumentSource, RetrievalInfo, ValidationInfo
        srcs = [DocumentSource(chunk_id=s, document="M") for s in sources]
        self.r = ChatResponse(session_id="s", message="q", answer="정상 범위는 -60 ~ -80 kPa\n출처: M", intent="knowledge",
                              sources=srcs, tools_used=list(tools_used), retrieval=RetrievalInfo(documents_found=len(srcs), reranked=True, top_k=len(srcs), sources=srcs),
                              validation=ValidationInfo(ok=True, is_refusal=False), route=route, machine_state=None, latency_ms=1.0)


class _Pipe:
    def __init__(self, tools_used): self.tools_used = tools_used; self.sessions = type("S", (), {"delete": lambda *a: None})()
    def answer(self, sid, q, filters=None): return _Resp(self.tools_used).r
    class retrieval:
        class hybrid:
            class store:
                @staticmethod
                def get_by_ids(ids): return [c for c in CTX if c["chunk_id"] in ids]


def test_tool_selection_accuracy_scoring():
    ev = AnswerEvaluator(_Pipe(["get_vacuum_pressure", "get_current_alarm"]))
    cases = [EvalCase(id="1", question="q", expected_intent="knowledge", expected_tools=["get_vacuum_pressure"]),
             EvalCase(id="2", question="q", expected_intent="knowledge", expected_tools=[]),
             EvalCase(id="3", question="q", expected_intent="knowledge")]
    res = ev.run(cases)
    rows = {r["id"]: r for r in res["rows"]}
    assert rows["1"]["tools_ok"] is True and rows["2"]["tools_ok"] is False and rows["3"]["tools_ok"] is None
    assert res["summary"]["tool_selection_accuracy"] == 0.5 and not rows["2"]["correct"]


def test_judge_metrics_aggregated_and_overcitation():
    verdict = {"sentences": [{"text": "s", "supported": True}], "used_sources": [], "contradicts_reference": False, "missing_from_reference": []}
    ev = AnswerEvaluator(_Pipe([]), judge=AnswerJudge(_LLM(json.dumps(verdict))))
    res = ev.run([EvalCase(id="1", question="q", expected_intent="knowledge", reference_answer="ref")])
    s = res["summary"]
    assert s["groundedness_mean"] == 1.0 and s["fully_grounded_rate"] == 1.0 and s["completeness_mean"] == 1.0
    assert s["overcitation_rate"] == 1.0                     # cited source 1, judge says none used
    assert res["rows"][0]["judge"]["overcited"] == [1]


# ---------- question generation helpers ----------

def test_seed_must_include_and_chunk_selection():
    assert seed_must_include("PR당 2,000줄 이하, v0.0.52부터 `plane_fill_enabled` 기본 false", ["2,000줄"])[:3] == ["v0.0.52", "plane_fill_enabled", "2,000줄"]
    chunks = [{"chunk_id": f"c{i}", "source": "s" if i < 5 else "t", "tokens": 100 + i * 50, "text": "x" * 500} for i in range(8)]
    chunks.append({"chunk_id": "tbl", "source": "u", "tokens": 900, "text": "| a | b |\n" * 60})
    sel = select_chunks(chunks, n=6, min_tokens=150, per_source_cap=3, seed=1)
    assert len(sel) == 6 and all(c["tokens"] >= 150 for c in sel) and "tbl" not in [c["chunk_id"] for c in sel]
    from collections import Counter
    assert max(Counter(c["source"] for c in sel).values()) <= 3
