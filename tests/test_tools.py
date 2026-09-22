from pathlib import Path

import pytest

from app.tools.base import BaseTool, ToolRegistry
from app.tools.mock import MockMachine, build_mock_registry
from app.troubleshooting import TroubleshootingEngine, load_rules
from app.troubleshooting.engine import evaluate_ok_when

CFG = Path(__file__).parent.parent / "configs/machine.yaml"
RULES = Path(__file__).parent.parent / "configs/troubleshooting.yaml"


class _Writer(BaseTool):
    name, description, read_only = "robot_reset", "resets the robot", False
    def _run(self): return "boom"


class _Flaky(BaseTool):
    name = "get_vacuum_pressure"
    def _run(self): raise TimeoutError("PLC timeout")


def test_registry_refuses_writable_tools():
    with pytest.raises(ValueError, match="READ ONLY"):
        ToolRegistry().register(_Writer())


def test_unknown_tool_and_failing_tool_return_error_results_not_exceptions():
    reg = ToolRegistry().register(_Flaky())
    assert not reg.call("nope").success and "unknown tool" in reg.call("nope").error
    r = reg.call("get_vacuum_pressure")
    assert not r.success and "TimeoutError" in r.error
    state, used = reg.snapshot()
    assert used == ["get_vacuum_pressure"] and state["vacuum_pressure"].startswith("unavailable")


def test_mock_registry_matches_spec_example():
    reg, m = build_mock_registry(CFG)
    assert m.scenario == "vac203_low"
    state, used = reg.snapshot()
    assert state["machine_state"] == "HOLD" and state["alarm"] == "VAC-203"
    assert state["vacuum_pressure"] == -43 and state["vision_detection"] is True and state["robot_state"] == "NORMAL"
    assert "get_vacuum_pressure" in used and "get_current_alarm" in used
    assert reg.call("get_plc_io", tag_name="X001_vacuum_ok").data["value"] is False
    assert reg.call("get_sensor_value", sensor_name="supply_air_mpa").data["value"] == 0.55
    assert not reg.call("get_sensor_value", sensor_name="nope").success


def test_mock_scenario_switch_and_maintenance_filter():
    reg, m = build_mock_registry(CFG)
    m.set_scenario("camera_timeout")
    state, _ = reg.snapshot()
    assert state["alarm"] == "VIS-301" and state["camera_state"] == "disconnected" and state["vision_detection"] is False
    with pytest.raises(KeyError):
        m.set_scenario("nope")
    rows = reg.call("get_maintenance_history", alarm_code="VAC-203").data
    assert len(rows) == 1 and rows[0]["date"] == "2026-09-10"
    assert len(reg.call("get_maintenance_history").data) == 3
    assert reg.call("get_maintenance_history", component="camera").data[0]["alarm_code"] == "VIS-301"
    assert reg.call("get_alarm_history", alarm_code="VIS-301").data[0]["date"] == "2026-08-02"


@pytest.mark.parametrize("value,rule,expected", [
    (-43, "<= -60", False), (-72, "<= -60", True), (-60, "<= -60", True),
    ("connected", "== connected", True), ("disconnected", "== connected", False),
    (5, ">= 5", True), (None, "<= -60", None), (-43, None, None), ("unavailable (x)", "<= -60", None),
    ("abc", "<= 3", None),
])
def test_evaluate_ok_when(value, rule, expected):
    assert evaluate_ok_when(value, rule) is expected


def test_prefill_uses_numeric_rule_and_skips_answered_questions():
    engine = TroubleshootingEngine(load_rules(RULES))
    reg, _ = build_mock_registry(CFG)
    state, _ = reg.snapshot()                       # vision True, vacuum -43
    st = engine.start("pickup_failure")
    filled = engine.prefill_from_tools(st, state)
    assert filled == ["vision_detection", "vacuum_ok"] and st.facts == {"vision_detection": True, "vacuum_ok": False}
    d = engine.next(st)
    assert d.kind == "ask" and d.fact == "robot_position_ok"       # only the unknown fact is asked
    engine.apply_reply(st, "응")
    assert engine.next(st).step == "check_vacuum"


def test_prefill_camera_state_string_rule():
    engine = TroubleshootingEngine(load_rules(RULES))
    reg, m = build_mock_registry(CFG); m.set_scenario("camera_timeout")
    state, _ = reg.snapshot()
    st = engine.start("detection_failure")
    assert engine.prefill_from_tools(st, state) == ["camera_connected"] and st.facts == {"camera_connected": False}
    assert engine.next(st).step == "check_camera"
