from app.llm.prompt import (NO_DOCS_NOTICE, ContextBuilder, PromptBuilder, SYSTEM_PROMPT,
                            format_source, trust_label)
from app.schemas.retrieval import RetrievedDoc


def doc(cid, text, **meta):
    return RetrievedDoc(chunk_id=cid, text=text, score=1.0, rank=1, source="rerank", metadata=meta)


def chars(s):  # deterministic "tokenizer": 1 token per char
    return len(s)


def test_source_block_has_document_section_alarm_trust_content():
    d = doc("c", "본문", document_name="Vacuum Manual", section="6.3", alarm_code="VAC-203", document_type="alarm_manual")
    b = format_source(1, d)
    assert b.startswith("[SOURCE 1]\nDocument: Vacuum Manual\nSection: 6.3\nAlarm: VAC-203\nTrust: Official Alarm Manual")
    assert b.endswith("Content:\n본문")


def test_trust_label_marks_non_official_sources():
    assert "not an official" in trust_label(doc("c", "x", document_type="maintenance"))
    assert "not official" in trust_label(doc("c", "x", document_type="engineer_note"))
    assert trust_label(doc("c", "x", document_type="unknown_type", trust_level=42)) == "Trust 42"


def test_context_keeps_pipeline_order_and_respects_max_docs():
    docs = [doc(f"d{i}", "x" * 10, document_name=f"D{i}") for i in range(7)]
    built = ContextBuilder(chars, max_tokens=10_000, max_docs=5).build(docs)
    assert [d.chunk_id for d in built.included] == ["d0", "d1", "d2", "d3", "d4"]
    assert built.dropped == ["d5", "d6"]
    assert "[SOURCE 5]" in built.text and "[SOURCE 6]" not in built.text


def test_context_drops_docs_that_exceed_budget_but_keeps_earlier_ones():
    docs = [doc("a", "x" * 50, document_name="A"), doc("b", "y" * 400, document_name="B"), doc("c", "z" * 50, document_name="C")]
    built = ContextBuilder(chars, max_tokens=250, max_docs=5).build(docs)
    assert [d.chunk_id for d in built.included] == ["a", "c"]
    assert built.dropped == ["b"] and not built.truncated
    assert built.tokens <= 250


def test_context_truncates_first_doc_if_it_alone_exceeds_budget():
    docs = [doc("big", "가" * 1000, document_name="Big")]
    built = ContextBuilder(chars, max_tokens=200, max_docs=5).build(docs)
    assert built.truncated and built.included[0].chunk_id == "big"
    assert "[truncated]" in built.text and built.tokens <= 200 + 20


def test_prompt_messages_layout_and_no_docs_notice():
    pb = PromptBuilder(ContextBuilder(chars), history_turns=1)
    msgs, built = pb.build("질문?", docs=[])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[-1]["role"] == "user"
    assert NO_DOCS_NOTICE in msgs[-1]["content"] and "[QUESTION]\n질문?" in msgs[-1]["content"]
    assert built.included == []


def test_history_trimmed_to_last_turn_only():
    pb = PromptBuilder(ContextBuilder(chars), history_turns=1)
    history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    msgs, _ = pb.build("q3", docs=[], history=history)
    contents = [m["content"] for m in msgs[1:-1]]
    assert contents == ["q2", "a2"]


def test_machine_state_block_only_when_provided():
    pb = PromptBuilder(ContextBuilder(chars))
    without, _ = pb.build("q", docs=[])
    with_state, _ = pb.build("q", docs=[], machine_state={"alarm": "VAC-203", "vacuum_pressure": -43})
    assert "[MACHINE STATE]" not in without[-1]["content"]
    assert "[MACHINE STATE]" in with_state[-1]["content"] and "vacuum_pressure: -43" in with_state[-1]["content"]
