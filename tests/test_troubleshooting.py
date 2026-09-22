from pathlib import Path

import pytest

from app.troubleshooting import TroubleshootingEngine, load_rules, parse_yes_no

RULES = Path(__file__).parent.parent / "configs/troubleshooting.yaml"


@pytest.fixture(scope="module")
def engine():
    return TroubleshootingEngine(load_rules(RULES))


def test_rules_load_and_validate(engine):
    assert set(engine.rules.issues) >= {"pickup_failure", "detection_failure", "position_offset"}
    pf = engine.rules.issues["pickup_failure"]
    assert pf.order[0] == "vision_detection" and pf.steps[0].id == "check_vision"


def test_invalid_rules_rejected(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text("issues:\n  x:\n    label: X\n    triggers: [a]\n    facts: {f1: {question: q}}\n    steps:\n      - {id: s, when: {nope: true}, actions: [a], doc_query: q}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="undefined facts"):
        load_rules(bad)


@pytest.mark.parametrize("text,issue", [
    ("박스를 못 집어", "pickup_failure"),
    ("빨판이 박스를 못 잡아", "pickup_failure"),
    ("박스가 있는데 검출이 안 돼", "detection_failure"),
    ("로봇이 박스 위치에서 항상 조금씩 어긋나", "position_offset"),
    ("VAC-203이 뭐야?", None),
])
def test_issue_matching(engine, text, issue):
    assert engine.match_issue(text) == issue


@pytest.mark.parametrize("text,expected", [
    ("응", True), ("네 정상이에요", True), ("잘 나와요", True), ("맞아", True), ("ok", True),
    ("아니", False), ("안 나와", False), ("아니요 안 보여요", False), ("검은 화면이야", False), ("못 잡아", False),
    ("VAC-203이 뭐야?", None), ("잘 모르겠어", None), ("어디부터 봐?", None), ("예전에도 발생했어?", None),
])
def test_yes_no_parsing(text, expected):
    assert parse_yes_no(text) is expected


def test_spec_dialogue_asks_in_order_then_recommends_vacuum(engine):
    """spec §20: 못 집어 → 비전? 응 → 로봇 위치? 응 → 진공 확인 권고."""
    st = engine.start("pickup_failure")
    d1 = engine.next(st)
    assert d1.kind == "ask" and d1.fact == "vision_detection" and "비전" in d1.question
    assert engine.apply_reply(st, "응") == "answered" and st.facts == {"vision_detection": True}
    d2 = engine.next(st)
    assert d2.kind == "ask" and d2.fact == "robot_position_ok"
    engine.apply_reply(st, "응")
    d3 = engine.next(st)
    assert d3.kind == "ask" and d3.fact == "vacuum_ok"
    engine.apply_reply(st, "아니, -45 정도에서 안 올라가")
    d4 = engine.next(st)
    assert d4.kind == "recommend" and d4.step == "check_vacuum"
    assert any("hose" in a for a in d4.actions) and "VAC-203" in d4.doc_query
    assert d4.confirmed == {"vision_detection": True, "robot_position_ok": True, "vacuum_ok": False}
    assert d4.unknown == ["box_surface_ok"]


def test_negative_first_fact_short_circuits_to_vision_step(engine):
    st = engine.start("pickup_failure")
    engine.next(st); engine.apply_reply(st, "아니 안 보여")
    d = engine.next(st)
    assert d.kind == "recommend" and d.step == "check_vision"


def test_irrelevant_facts_are_not_asked(engine):
    """Once vision=false, robot/vacuum questions are pointless (no remaining step needs them)."""
    st = engine.start("pickup_failure", known_facts={"vision_detection": False})
    d = engine.next(st)
    assert d.kind == "recommend" and d.step == "check_vision"


def test_escalate_when_everything_normal(engine):
    st = engine.start("pickup_failure", known_facts={"vision_detection": True, "robot_position_ok": True, "vacuum_ok": True, "box_surface_ok": True})
    d = engine.next(st)
    assert d.kind == "escalate" and "개발팀" in d.message


def test_unclear_reply_reasks_same_fact(engine):
    st = engine.start("pickup_failure")
    engine.next(st)
    assert engine.apply_reply(st, "글쎄 잘 모르겠는데") == "unclear"
    d = engine.next(st)
    assert d.kind == "ask" and d.fact == "vision_detection"


def test_resolved_and_cancel(engine):
    st = engine.start("pickup_failure"); engine.next(st)
    assert engine.apply_reply(st, "아 해결됐어 고마워") == "resolved" and engine.next(st).kind == "resolved"
    st2 = engine.start("pickup_failure"); engine.next(st2)
    assert engine.apply_reply(st2, "그만할래 다른 질문") == "cancelled" and st2.resolved


def test_is_reply_distinguishes_answers_from_new_questions(engine):
    assert engine.is_reply("응") and engine.is_reply("아니요 안 나와요") and engine.is_reply("해결됐어")
    assert engine.is_reply("아니, -45 정도에서 안 올라가") and engine.is_reply("잘 모르겠어")
    assert not engine.is_reply("VAC-203 정상 범위가 얼마야?")
    assert not engine.is_reply("그 알람 원인이 뭐야?")
    assert not engine.is_reply("캘리브레이션 방법 알려줘")


def test_prefill_from_machine_state(engine):
    st = engine.start("pickup_failure")
    engine.prefill_from_tools(st, {"vision_detection": True, "vacuum_pressure": -43})   # non-bool ignored
    assert st.facts == {"vision_detection": True}
    d = engine.next(st)
    assert d.kind == "ask" and d.fact == "robot_position_ok"
