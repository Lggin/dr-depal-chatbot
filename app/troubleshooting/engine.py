"""Decision-tree troubleshooting engine (spec §19-§20).

The engine never guesses: it asks about one unknown fact at a time (in the
issue's `order`), and recommends the first step whose conditions are fully
satisfied by confirmed facts. All logic is deterministic Python; the LLM only
phrases the grounded inspection procedure for a recommended step.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.troubleshooting.rules import Issue, Rules, Step
from app.troubleshooting.state import TroubleshootingState

# No trailing \b: Hangul syllables are all \w, so "나와요" would never match "나와\b".
# Bare "어/엉" are excluded — "어디부터 봐?" must not read as "yes"; "예" must not match "예전에".
_YES = re.compile(r"^\s*(?:응|네|넵|예(?![전상외시])|그래|그렇|맞아|맞습|됩니다|돼|된다|되고\s*있|잘\s*(?:나와|보여|돼|되|나옵)|정상|이상\s*없|문제\s*없|켜져|연결돼|나와|보여|(?:ok|okay|yes|y)(?![a-z]))", re.IGNORECASE)
_NO = re.compile(r"(아니|아뇨|안\s*(돼|되|나와|보여|잡|집|켜|나옵|보입|나오)|못\s*(잡|집|찾|보)|없어|없습|불량|이상해|이상이|꺼져|끊겨|검은|멈춰|틀어|어긋|no\b|nope)", re.IGNORECASE)
_RESOLVED = re.compile(r"(해결|됐어|됐습|고쳤|정상으로\s*돌아|이제\s*(잘|돼|된다|나와)|잘\s*되네|괜찬|괜찮)")
_CANCEL = re.compile(r"(그만|취소|다른\s*질문|됐고|말고|아무것도)")
# A short message that asks something new is a topic change, not an answer to our question.
_NEW_TOPIC = re.compile(r"(뭐야|뭔지|무엇|알려|설명|방법|절차|얼마|어떻게|왜|정상\s*범위|기본\s*값|이력|지난)")
_ALARM = re.compile(r"[A-Za-z]{1,4}[\s\-_]?\d{2,4}")


def parse_yes_no(text: str) -> Optional[bool]:
    """True/False for a yes/no reply, None when ambiguous or not a reply."""
    t = text.strip()
    if _NO.search(t) and not re.match(r"^\s*(응|네|예|맞아|그래)", t):
        return False
    if _YES.match(t):
        return True
    if _NO.search(t):
        return False
    return None


@dataclass
class Decision:
    kind: str                         # "ask" | "recommend" | "escalate" | "resolved" | "cancelled"
    issue: str
    label: str
    fact: Optional[str] = None
    question: Optional[str] = None
    step: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    doc_query: Optional[str] = None
    message: Optional[str] = None
    confirmed: Dict[str, bool] = field(default_factory=dict)
    unknown: List[str] = field(default_factory=list)


class TroubleshootingEngine:
    def __init__(self, rules: Rules):
        self.rules = rules

    # ---------- lifecycle ----------

    def match_issue(self, text: str) -> Optional[str]:
        return self.rules.match_issue(text)

    def start(self, issue_id: str, known_facts: Optional[Dict[str, bool]] = None) -> TroubleshootingState:
        if issue_id not in self.rules.issues:
            raise KeyError(issue_id)
        return TroubleshootingState(issue=issue_id, facts=dict(known_facts or {}))

    def is_reply(self, text: str) -> bool:
        """Does this message look like an answer to the pending question rather than a new topic?

        Explicit yes/no/resolved/cancel replies count even with a short justification attached
        ("아니, -45 정도에서 안 올라가"). Anything else counts only if short and free of
        new-question markers and alarm codes — "잘 모르겠어" stays in the diagnosis, while
        "VAC-203 정상 범위가 얼마야?" leaves it."""
        t = text.strip()
        if parse_yes_no(t) is not None or _RESOLVED.search(t) or _CANCEL.search(t):
            return len(t) <= 40
        return len(t) <= 25 and not _NEW_TOPIC.search(t) and not _ALARM.search(t)

    def apply_reply(self, state: TroubleshootingState, text: str) -> str:
        """Update state from the user's reply. Returns 'answered' | 'resolved' | 'cancelled' | 'unclear'."""
        state.turns += 1
        if _RESOLVED.search(text):
            state.resolved = True
            return "resolved"
        if _CANCEL.search(text):
            state.resolved = True
            return "cancelled"
        if state.pending_fact is None:
            return "unclear"
        val = parse_yes_no(text)
        if val is None:
            return "unclear"
        state.facts[state.pending_fact] = val
        state.pending_fact = None
        return "answered"

    def prefill_from_tools(self, state: TroubleshootingState, machine_state: Optional[Dict[str, object]]) -> None:
        """Phase 4 hook: facts with a `tool` key are filled from live machine data when available."""
        if not machine_state:
            return
        issue = self.rules.issues[state.issue]
        for name, fact in issue.facts.items():
            if fact.tool and fact.tool in machine_state and name not in state.facts:
                v = machine_state[fact.tool]
                if isinstance(v, bool):
                    state.facts[name] = v

    # ---------- decision ----------

    def next(self, state: TroubleshootingState) -> Decision:
        issue: Issue = self.rules.issues[state.issue]
        base = dict(issue=issue.id, label=issue.label, confirmed=dict(state.facts), unknown=state.unknown(list(issue.facts)))
        if state.resolved:
            return Decision(kind="resolved", message="문제가 해결되었다고 하셨습니다. 추가로 확인할 것이 있으면 말씀해 주세요.", **base)

        step = self._matching_step(issue, state.facts)
        if step is not None:
            state.current_step = step.id
            return Decision(kind="recommend", step=step.id, actions=list(step.actions), doc_query=step.doc_query, **base)

        for fact_name in issue.order:
            if fact_name in state.facts or not self._fact_still_relevant(issue, state.facts, fact_name):
                continue
            state.pending_fact = fact_name
            return Decision(kind="ask", fact=fact_name, question=issue.facts[fact_name].question, **base)

        return Decision(kind="escalate", message=issue.escalate, **base)

    # ---------- helpers ----------

    @staticmethod
    def _matching_step(issue: Issue, facts: Dict[str, bool]) -> Optional[Step]:
        for step in issue.steps:
            if all(facts.get(k) is not None and facts[k] == v for k, v in step.when.items()):
                return step
        return None

    @staticmethod
    def _fact_still_relevant(issue: Issue, facts: Dict[str, bool], fact_name: str) -> bool:
        """A fact is worth asking only if some step that needs it is still consistent with known facts."""
        for step in issue.steps:
            if fact_name not in step.when:
                continue
            if all(facts[k] == v for k, v in step.when.items() if k in facts):
                return True
        return False
