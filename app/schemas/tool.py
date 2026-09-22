"""Tool/API request/response schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    """Tool invocation record."""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: float
    result: Optional[Any] = None
    error: Optional[str] = None


class ToolResponse(BaseModel):
    """Generic tool response."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float


class AlarmToolRequest(BaseModel):
    """Get alarm details."""
    alarm_code: str


class AlarmToolResponse(ToolResponse):
    """Alarm details response."""
    data: Optional[Dict[str, Any]] = None


class PLCIORequest(BaseModel):
    """Get PLC I/O value."""
    tag_name: str


class PLCIOResponse(ToolResponse):
    """PLC I/O value response."""
    data: Optional[float] = None


class RobotPoseRequest(BaseModel):
    """Get robot current pose."""
    robot_id: str = "default"


class RobotPoseResponse(ToolResponse):
    """Robot pose response."""
    data: Optional[Dict[str, float]] = None


class VacuumPressureRequest(BaseModel):
    """Get vacuum pressure."""
    gripper_id: str = "default"


class VacuumPressureResponse(ToolResponse):
    """Vacuum pressure response."""
    data: Optional[float] = None


class MaintenanceHistoryRequest(BaseModel):
    """Get maintenance history."""
    alarm_code: Optional[str] = None
    component: Optional[str] = None
    limit: int = 10


class MaintenanceHistoryResponse(ToolResponse):
    """Maintenance history response."""
    data: Optional[List[Dict[str, Any]]] = None
