"""Load and validate configs/troubleshooting.yaml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Fact:
    name: str
    question: str
    tool: Optional[str] = None          # machine_state key that can answer this fact (Phase 4)
    ok_when: Optional[str] = None       # for numeric/string values: "<= -60", ">= 5", "== connected"


@dataclass
class Step:
    id: str
    when: Dict[str, bool]
    actions: List[str]
    doc_query: str


@dataclass
class Issue:
    id: str
    label: str
    triggers: List[str]
    facts: Dict[str, Fact]
    order: List[str]
    steps: List[Step]
    escalate: str


@dataclass
class Rules:
    issues: Dict[str, Issue] = field(default_factory=dict)

    def match_issue(self, text: str) -> Optional[str]:
        low = text.lower().replace(" ", "")
        best, best_len = None, 0
        for issue in self.issues.values():
            for trig in issue.triggers:
                t = trig.lower().replace(" ", "")
                if t in low and len(t) > best_len:          # longest trigger wins
                    best, best_len = issue.id, len(t)
        return best


def load_rules(path: Path) -> Rules:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules = Rules()
    for iid, spec in (raw.get("issues") or {}).items():
        facts = {n: Fact(n, f["question"], f.get("tool"), f.get("ok_when")) for n, f in spec["facts"].items()}
        steps = [Step(s["id"], dict(s.get("when", {})), list(s["actions"]), s["doc_query"]) for s in spec["steps"]]
        for s in steps:
            unknown = set(s.when) - set(facts)
            if unknown:
                raise ValueError(f"{iid}/{s.id}: `when` references undefined facts {sorted(unknown)}")
        order = list(spec.get("order") or facts.keys())
        missing = set(order) - set(facts)
        if missing:
            raise ValueError(f"{iid}: `order` references undefined facts {sorted(missing)}")
        rules.issues[iid] = Issue(iid, spec["label"], list(spec["triggers"]), facts, order, steps, spec.get("escalate", ""))
    if not rules.issues:
        raise ValueError(f"no issues defined in {path}")
    return rules
