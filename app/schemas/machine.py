"""Machine/equipment state schemas."""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum


class MachineState(str, Enum):
    """Machine operational state."""
    IDLE = "idle"
    RUNNING = "running"
    HOLD = "hold"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class RobotState(str, Enum):
    """Robot state."""
    IDLE = "idle"
    MOVING = "moving"
    TEACHING = "teaching"
    ERROR = "error"


class MachineStatus(BaseModel):
    """Current machine state."""
    machine_id: str
    state: MachineState
    timestamp: float

    # Current alarm
    current_alarm: Optional[str] = None
    alarm_timestamp: Optional[float] = None

    # System values
    vacuum_pressure: Optional[float] = None
    robot_state: Optional[RobotState] = None
    robot_position: Optional[Dict[str, float]] = None

    # Vision
    vision_detection: Optional[bool] = None
    detected_objects: Optional[int] = None

    # Sensors
    sensor_values: Dict[str, float] = {}

    # PLC I/O
    plc_inputs: Dict[str, Any] = {}
    plc_outputs: Dict[str, Any] = {}


class CurrentAlarm(BaseModel):
    """Current alarm information."""
    alarm_code: str
    timestamp: float
    severity: str  # "info", "warning", "error", "critical"
    description: Optional[str] = None


class RobotPose(BaseModel):
    """Robot pose/position."""
    x: float
    y: float
    z: float
    rx: Optional[float] = None
    ry: Optional[float] = None
    rz: Optional[float] = None
    frame: str = "base"


class SensorReading(BaseModel):
    """Single sensor reading."""
    sensor_name: str
    value: float
    unit: str
    timestamp: float
    status: str = "normal"  # "normal", "warning", "error"


class AlarmHistory(BaseModel):
    """Alarm occurrence history."""
    alarm_code: str
    occurrences: int
    first_occurrence: float
    last_occurrence: float
    resolved: bool
    typical_duration_minutes: Optional[float] = None
