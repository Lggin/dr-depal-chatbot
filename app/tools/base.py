"""Machine tool interface (spec §16, §22). V1 is strictly READ ONLY.

A tool is a named, side-effect-free read of machine state. The registry refuses
anything that declares itself writable — there is no permission layer yet, so
there must be nothing to permit.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    arguments: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict[str, str] = {}          # arg name -> description
    read_only: bool = True

    def call(self, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        try:
            data = self._run(**kwargs)
            return ToolResult(self.name, True, data, latency_ms=(time.perf_counter() - t0) * 1000, arguments=kwargs)
        except Exception as e:  # a failing sensor read must not take the chat down
            logger.warning(f"tool {self.name} failed: {type(e).__name__}: {e}")
            return ToolResult(self.name, False, None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000, kwargs)

    @abstractmethod
    def _run(self, **kwargs) -> Any: ...


class ToolRegistry:
    # Tools whose results together form the "current machine state" block for the LLM.
    STATE_TOOLS = ("get_machine_state", "get_current_alarm", "get_vacuum_pressure",
                   "get_vision_detection_state", "get_robot_state", "get_camera_state")

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        if not tool.read_only:
            raise ValueError(f"refusing to register writable tool {tool.name!r}: V1 is READ ONLY (spec §22)")
        if not tool.name:
            raise ValueError("tool needs a name")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters, "read_only": t.read_only}
                for t in self._tools.values()]

    def call(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, None, f"unknown tool {name!r}", arguments=kwargs)
        return tool.call(**kwargs)

    def snapshot(self) -> tuple[Dict[str, Any], List[str]]:
        """Flat current-state dict for [MACHINE STATE] + the tool names that produced it."""
        state: Dict[str, Any] = {}
        used: List[str] = []
        for name in self.STATE_TOOLS:
            if name not in self._tools:
                continue
            r = self.call(name)
            used.append(name)
            if r.success and isinstance(r.data, dict):
                state.update(r.data)
            elif r.success:
                state[name.replace("get_", "")] = r.data
            else:
                state[name.replace("get_", "")] = f"unavailable ({r.error})"
        return state, used
