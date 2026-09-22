"""Per-session diagnosis state (spec §20)."""

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TroubleshootingState:
    issue: str
    facts: Dict[str, bool] = field(default_factory=dict)      # confirmed observations
    pending_fact: Optional[str] = None                        # the fact we just asked about
    current_step: Optional[str] = None
    turns: int = 0
    resolved: bool = False
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def unknown(self, all_facts: List[str]) -> List[str]:
        return [f for f in all_facts if f not in self.facts]

    def snapshot(self) -> dict:
        d = asdict(self)
        d["confirmed"] = {k: v for k, v in self.facts.items()}
        return d
