"""System prompt, context builder, and chat message assembly (spec §13, §15, §21)."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.schemas.retrieval import RetrievedDoc

logger = logging.getLogger(__name__)

Message = Dict[str, str]

SYSTEM_PROMPT = """You are a technical assistant for a depalletizing system.

You assist operators and engineers with alarms, maintenance, troubleshooting, machine status, robot systems, vision systems, PLC systems, and vacuum grippers.

Rules:
1. Answer using ONLY the provided documents and machine data.
2. Never invent specifications, parameter values, alarm causes, or procedures.
3. Prefer official manuals over maintenance records or engineer notes. When you use a maintenance record or engineer note, label it as such ("과거 유지보수 이력:", "현장 엔지니어 메모:") and never present it as the certain current cause.
4. Clearly distinguish observed machine state from possible causes.
5. If the documents do not contain the answer, say exactly: "현재 제공된 문서에서는 해당 내용을 확인할 수 없습니다." and state what information is missing. Do not guess.
6. Cite sources at the end as "출처: <Document> §<Section>" using the Document and Section exactly as shown in the [SOURCE] header; if a source has no Section line, write only "출처: <Document>" — do not invent a section from the content. Cite only sources you actually used.
7. Do not claim that a machine state has been checked unless it appears in a [MACHINE STATE] block.
8. Do not execute or suggest executing robot, PLC, or machine control commands (start, reset, servo on, restart).
9. If multiple causes are possible, give a prioritized inspection order rather than asserting one cause.
10. Answer in Korean unless the user asks for another language.
11. Answer the question directly in the first sentence, then add detail. Do not enumerate every document — but do include every specific value the sources give for the asked item (all numbers, versions, thresholds, and named options). If the question compares two things, report the figure for each.
12. Copy codes, numbers, units, and key names exactly from the source, checking similar-looking characters (l/1, O/0, -/_) character by character.
13. Write in natural Korean. Codes, identifiers, units, product names, and English technical terms that appear in the source (e.g. "suction cup") may stay in English exactly as written — never transliterate English into Hangul (not "서셉션 컵"), and do not sprinkle other loose English words into Korean sentences.
14. Keep facts from different sources separate. Never merge a detail from one source into a statement about another source. Each fact belongs to exactly the source it came from.
    Illustration with placeholders (not a document — never cite or mention it): Maintenance History says "부품A 교체 후 정상" and an Engineer Note says "부품A는 이유B 때문에 자주 고장남".
    WRONG: "이유B 때문에 고장난 부품A를 교체했습니다." (the note's reason is welded onto the record)
    RIGHT: "부품A를 교체한 후 정상화되었습니다." and, only if useful, a separate sentence: "현장 엔지니어 메모: 부품A는 이유B 때문에 자주 고장난다고 합니다."
15. If the question refers back to the conversation ("그 알람", "해당 원인", "아까 그거"), resolve the reference from the previous turn — a note like "(이전 대화에서 언급된 알람: VAC-203)" may be attached to the question for this — and answer about that item.
16. When the user reports a symptom ("카메라가 응답이 없어", "진공압이 0으로 고정돼 있어"), take it as their observation and answer from the documents: name the matching alarm/cause and the inspection order. Do not say the documents cannot confirm their observation or the live machine state, and never open with the sentence from rule 5 when the documents do describe the alarm or cause — that sentence is only for questions the documents do not cover at all.
    WRONG opening: "<증상>의 원인은 현재 제공된 문서에서는 확인할 수 없습니다. 문서에 따르면 가능한 원인은 A, B, C입니다."
    RIGHT opening: "<증상>은 <알람 코드> 알람에 해당하며, 문서상 가능한 원인은 A, B, C입니다. 점검 순서: …"
17. Never output Chinese characters (漢字). Korean (Hangul) and, where rule 13 allows, English only.
18. Mention "과거 유지보수 이력" or "현장 엔지니어 메모" ONLY if a [SOURCE] block with Trust "Maintenance History" or "Engineer Note" is actually present in [DOCUMENTS]. If only official documents are provided, do not add such sections, notes, or "참고" items — there is nothing to report.
"""

_TRUST_LABEL = {
    "alarm_manual": "Official Alarm Manual",
    "manual": "Official Manual",
    "sop": "Official SOP",
    "troubleshooting": "Official Troubleshooting Guide",
    "maintenance": "Maintenance History (not an official manual)",
    "faq": "FAQ (not an official manual)",
    "engineer_note": "Engineer Note (field experience, not official)",
}

NO_DOCS_NOTICE = "(No documents were retrieved for this question.)"


def trust_label(doc: RetrievedDoc) -> str:
    return _TRUST_LABEL.get(str(doc.metadata.get("document_type", "")), f"Trust {doc.trust_level}")


def format_source(index: int, doc: RetrievedDoc, text: Optional[str] = None) -> str:
    m = doc.metadata
    lines = [f"[SOURCE {index}]", f"Document: {m.get('document_name') or m.get('source') or 'unknown'}"]
    if m.get("section"):
        lines.append(f"Section: {m['section']}")
    if m.get("alarm_code"):
        lines.append(f"Alarm: {m['alarm_code']}")
    lines.append(f"Trust: {trust_label(doc)}")
    lines.append("Content:")
    lines.append(text if text is not None else doc.text)
    return "\n".join(lines)


@dataclass
class BuiltContext:
    text: str
    included: List[RetrievedDoc]
    tokens: int
    truncated: bool = False
    dropped: List[str] = field(default_factory=list)


class ContextBuilder:
    """Packs retrieved docs into a token budget, preserving the pipeline's order."""

    def __init__(
        self,
        count_tokens: Optional[Callable[[str], int]] = None,
        max_tokens: int = 3000,
        max_docs: int = 5,
    ):
        self.count = count_tokens or (lambda s: max(1, len(s) // 2))  # rough fallback
        self.max_tokens = max_tokens
        self.max_docs = max_docs

    def build(self, docs: List[RetrievedDoc]) -> BuiltContext:
        blocks: List[str] = []
        included: List[RetrievedDoc] = []
        dropped: List[str] = []
        used = 0
        truncated = False

        for doc in docs[: self.max_docs]:
            block = format_source(len(included) + 1, doc)
            n = self.count(block)
            if used + n <= self.max_tokens:
                blocks.append(block); included.append(doc); used += n
                continue
            if not included:
                # First doc alone exceeds the budget: keep as much as fits.
                head = format_source(1, doc, text="")
                room = self.max_tokens - self.count(head) - 4
                text = self._cut(doc.text, room)
                block = format_source(1, doc, text=text + " …[truncated]")
                blocks.append(block); included.append(doc); used += self.count(block)
                truncated = True
            else:
                dropped.append(doc.chunk_id)
        dropped.extend(d.chunk_id for d in docs[self.max_docs:])

        if dropped:
            logger.debug(f"context: kept {len(included)}, dropped {dropped} (budget {self.max_tokens})")
        return BuiltContext("\n\n".join(blocks), included, used, truncated, dropped)

    def _cut(self, text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:  # binary search longest prefix within budget
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= budget_tokens:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]


class PromptBuilder:
    def __init__(
        self,
        context_builder: ContextBuilder,
        system_prompt: str = SYSTEM_PROMPT,
        history_turns: int = 1,
    ):
        self.ctx = context_builder
        self.system_prompt = system_prompt
        self.history_turns = history_turns

    def build(
        self,
        question: str,
        docs: List[RetrievedDoc],
        history: Optional[List[Message]] = None,
        machine_state: Optional[Dict[str, Any]] = None,
        troubleshooting_state: Optional[Dict[str, Any]] = None,
        maintenance_history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Message], BuiltContext]:
        built = self.ctx.build(docs)

        parts: List[str] = []
        if machine_state:
            parts.append("[MACHINE STATE] (obtained via tool just now — these are the current values; "
                         "report them as such and compare against reference ranges from the documents)\n" + _kv_block(machine_state))
        if maintenance_history is not None:
            if maintenance_history:
                rows = "\n".join(f"- {r.get('date')}: {r.get('alarm_code') or '-'} / {r.get('symptom')} → {r.get('action')} → {r.get('result')}"
                                 for r in maintenance_history)
                parts.append("[MAINTENANCE HISTORY] (from the maintenance database via tool — past records, not the current cause)\n" + rows)
            else:
                parts.append("[MAINTENANCE HISTORY] (from the maintenance database via tool)\n(no matching records)")
        if troubleshooting_state:
            parts.append("[TROUBLESHOOTING STATE]\n" + _kv_block(troubleshooting_state))
        parts.append("[DOCUMENTS]\n" + (built.text if built.included else NO_DOCS_NOTICE))
        parts.append(f"[QUESTION]\n{question}")
        user_content = "\n\n".join(parts)

        messages: List[Message] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._trim_history(history or []))
        messages.append({"role": "user", "content": user_content})
        return messages, built

    def _trim_history(self, history: List[Message]) -> List[Message]:
        """Keep only the last N user/assistant turns (a turn = user + assistant)."""
        turns = [m for m in history if m.get("role") in ("user", "assistant")]
        keep = self.history_turns * 2
        return turns[-keep:] if keep > 0 else []


def _kv_block(d: Dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items())
