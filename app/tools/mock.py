"""Mock machine tools backed by configs/machine.yaml scenarios (spec §16).

Every tool reads from `MockMachine`; switching `MockMachine.scenario` changes what the
whole system "sees", which is how demos and tests reproduce specific faults.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.tools.base import BaseTool, ToolRegistry


class MockMachine:
    def __init__(self, config_path: Path):
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.machine_id: str = cfg.get("machine_id", "DP-01")
        self.scenarios: Dict[str, Dict[str, Any]] = cfg["scenarios"]
        self.records: List[Dict[str, Any]] = cfg.get("maintenance_records", [])
        self.scenario: str = cfg.get("scenario") or next(iter(self.scenarios))
        if self.scenario not in self.scenarios:
            raise ValueError(f"unknown scenario {self.scenario!r}; have {list(self.scenarios)}")

    @property
    def s(self) -> Dict[str, Any]:
        return self.scenarios[self.scenario]

    def set_scenario(self, name: str) -> None:
        if name not in self.scenarios:
            raise KeyError(f"unknown scenario {name!r}; have {list(self.scenarios)}")
        self.scenario = name


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class GetMachineState(BaseTool):
    name, description = "get_machine_state", "전체 설비 상태 (RUNNING/HOLD/ERROR 등)"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return {"machine_id": self.m.machine_id, "machine_state": self.m.s["machine_state"], "timestamp": _stamp()}


class GetCurrentAlarm(BaseTool):
    name, description = "get_current_alarm", "현재 발생 중인 알람 코드"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        a = self.m.s.get("current_alarm")
        return {"alarm": a["code"] if a else None, "alarm_severity": a["severity"] if a else None, "alarm_since": a["since"] if a else None}


class GetAlarmHistory(BaseTool):
    name, description = "get_alarm_history", "알람 코드별 과거 발생 이력"
    parameters = {"alarm_code": "알람 코드 (예: VAC-203)"}
    def __init__(self, m: MockMachine): self.m = m
    def _run(self, alarm_code: str) -> List[Dict[str, Any]]:
        return [r for r in self.m.records if r.get("alarm_code") == alarm_code]


class GetRobotState(BaseTool):
    name, description = "get_robot_state", "로봇 상태 (NORMAL/ERROR/MOVING)"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return {"robot_state": self.m.s["robot_state"]}


class GetRobotPose(BaseTool):
    name, description = "get_robot_pose", "로봇 TCP 현재 위치/자세"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return dict(self.m.s["robot_pose"])


class GetVacuumPressure(BaseTool):
    name, description = "get_vacuum_pressure", "그리퍼 진공압 (kPa)"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return {"vacuum_pressure": self.m.s["vacuum_pressure"], "vacuum_pressure_unit": "kPa"}


class GetCameraState(BaseTool):
    name, description = "get_camera_state", "카메라 연결 상태"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return {"camera_state": self.m.s["camera_state"]}


class GetVisionDetectionState(BaseTool):
    name, description = "get_vision_detection_state", "비전 검출 결과 (검출 여부, 객체 수)"
    def __init__(self, m: MockMachine): self.m = m
    def _run(self) -> Dict[str, Any]:
        return {"vision_detection": self.m.s["vision_detection"], "detected_objects": self.m.s.get("detected_objects", 0)}


class GetSensorValue(BaseTool):
    name, description = "get_sensor_value", "센서 값 조회"
    parameters = {"sensor_name": "센서 이름 (예: vacuum_pressure_kpa)"}
    def __init__(self, m: MockMachine): self.m = m
    def _run(self, sensor_name: str) -> Dict[str, Any]:
        sensors = self.m.s.get("sensors", {})
        if sensor_name not in sensors:
            raise KeyError(f"unknown sensor {sensor_name!r}; have {sorted(sensors)}")
        return {"sensor": sensor_name, "value": sensors[sensor_name], "timestamp": _stamp()}


class GetPlcIo(BaseTool):
    name, description = "get_plc_io", "PLC I/O 태그 값 조회"
    parameters = {"tag_name": "PLC 태그 (예: X001_vacuum_ok)"}
    def __init__(self, m: MockMachine): self.m = m
    def _run(self, tag_name: str) -> Dict[str, Any]:
        io = self.m.s.get("plc_io", {})
        if tag_name not in io:
            raise KeyError(f"unknown tag {tag_name!r}; have {sorted(io)}")
        return {"tag": tag_name, "value": io[tag_name], "timestamp": _stamp()}


class GetMaintenanceHistory(BaseTool):
    name, description = "get_maintenance_history", "유지보수 이력 조회 (알람/부품별)"
    parameters = {"alarm_code": "알람 코드 (선택)", "component": "부품 (선택)", "limit": "최대 건수"}
    def __init__(self, m: MockMachine): self.m = m
    def _run(self, alarm_code: Optional[str] = None, component: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.m.records
        if alarm_code:
            rows = [r for r in rows if r.get("alarm_code") == alarm_code]
        if component:
            rows = [r for r in rows if r.get("component") == component]
        return sorted(rows, key=lambda r: r["date"], reverse=True)[:limit]


def build_mock_registry(config_path: Path) -> tuple[ToolRegistry, MockMachine]:
    m = MockMachine(config_path)
    reg = ToolRegistry()
    for cls in (GetMachineState, GetCurrentAlarm, GetAlarmHistory, GetRobotState, GetRobotPose, GetVacuumPressure,
                GetCameraState, GetVisionDetectionState, GetSensorValue, GetPlcIo, GetMaintenanceHistory):
        reg.register(cls(m))
    return reg, m
