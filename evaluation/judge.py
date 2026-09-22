"""LLM-as-judge for answer quality (spec §27: Groundedness, Correctness, Citation Accuracy).

Runs on the same local LLM as the chatbot, temperature 0, strict JSON. The judge
sees the exact context the answer was allowed to use, so "unsupported" means
"not in the documents/machine state that were provided" — not "false in the world".
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CITATION_LINE = re.compile(r"^\W*출처\W*[:：].*$(?:\n(?:[ \t]*[-*•].*|[ \t]+.*))*", re.MULTILINE)
_CITATION_SENTENCE = re.compile(r"^\W*(출처|Sources?|References?)\W*[:：]", re.IGNORECASE)

JUDGE_SYSTEM ="""You are a strict evaluator for a technical assistant's answer. Judge ONLY against the provided SOURCES (and MACHINE STATE if given). Output valid JSON only — no prose, no markdown fences.

Return exactly this object:
{
  "sentences": [ {"text": "<answer sentence>", "supported": true|false, "evidence": "<source index or 'machine_state' or ''>"} ],
  "used_sources": [<indices of sources whose content the answer actually relies on>],
  "contradicts_reference": true|false,
  "missing_from_reference": ["<key fact in REFERENCE that the answer omits>", ...],
  "notes": "<one short sentence>"
}

Rules:
- Split the answer into factual sentences. Ignore greetings, the "출처:" line, and generic advice with no factual claim.
- A sentence is supported only if its specific facts (numbers, names, causes, steps) appear in a SOURCE or MACHINE STATE. Paraphrase is fine; new numbers, new causes, new mechanisms/explanations, or arithmetic the sources do not state are NOT supported.
- Comparing a MACHINE STATE value against a range or threshold stated in a SOURCE IS supported (e.g. "-43 kPa is outside the normal -60 ~ -80 kPa range", "the alarm matches VAC-203"). Computing a NEW number from them (e.g. "15 kPa short") is NOT.
- Compare signed values numerically: -43 is higher than -60, so "-43 kPa is above the -60 ~ -80 kPa range / the vacuum is too weak" is a correct statement, not a contradiction.
- Restating what the user asked is supported (evidence "question").
- Sentences that honestly say something is not in the documents are supported (evidence "").
- If no REFERENCE is given, set contradicts_reference=false and missing_from_reference=[].
- missing_from_reference lists only facts the REFERENCE treats as part of the answer, not background.
"""


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction: strips fences/prose, takes the first {...} block."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        # common model slip: trailing commas
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", t[start:end + 1]))
        except json.JSONDecodeError:
            return None


@dataclass
class JudgeResult:
    ok: bool                                   # judge produced a parseable verdict
    groundedness: Optional[float] = None       # supported / factual sentences
    unsupported: List[str] = field(default_factory=list)
    used_sources: List[int] = field(default_factory=list)
    contradicts_reference: Optional[bool] = None
    missing_from_reference: List[str] = field(default_factory=list)
    completeness: Optional[float] = None       # 1 - missing/(reference key facts) — approximated as 1 if none missing
    notes: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


class AnswerJudge:
    def __init__(self, llm, max_tokens: int = 700):
        self.llm = llm
        self.max_tokens = max_tokens

    @staticmethod
    def _sources_block(contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for i, c in enumerate(contexts, start=1):
            m = c.get("metadata", {})
            head = " / ".join(str(m[k]) for k in ("document_name", "section") if m.get(k))
            parts.append(f"[SOURCE {i}] {head}\n{c.get('text', '')}")
        return "\n\n".join(parts) if parts else "(no sources were provided to the assistant)"

    def judge(
        self,
        question: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        reference: Optional[str] = None,
        machine_state: Optional[Dict[str, Any]] = None,
        cited_source_count: Optional[int] = None,
    ) -> JudgeResult:
        # The citation line is produced by our own source_validator from metadata — never a claim to judge.
        # Strip it before judging (the model ignores "ignore it" instructions often enough to matter).
        answer = _CITATION_LINE.sub("", answer).strip()
        user = f"QUESTION:\n{question}\n\nSOURCES:\n{self._sources_block(contexts)}\n"
        if machine_state:
            user += "\nMACHINE STATE:\n" + "\n".join(f"{k}: {v}" for k, v in machine_state.items()) + "\n"
        if reference:
            user += f"\nREFERENCE (ground-truth answer):\n{reference}\n"
        user += f"\nANSWER TO JUDGE:\n{answer}\n\nJSON:"
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
        # The judge must be the BASE model: if the answering LLM carries a LoRA adapter (A/B runs),
        # switch it off for the judgement and back on afterwards.
        toggle = getattr(self.llm, "set_lora_enabled", None)
        had_lora = bool(toggle and toggle(False))
        try:
            gen = self.llm.generate(messages, temperature=0.0, max_tokens=self.max_tokens, top_p=1.0)
        except Exception as e:
            logger.warning(f"judge generation failed: {e}")
            return JudgeResult(ok=False, notes=f"generation failed: {e}")
        finally:
            if had_lora:
                toggle(True)
        data = parse_json_block(gen.text)
        if not data or not isinstance(data.get("sentences"), list):
            return JudgeResult(ok=False, notes="unparseable", raw=gen.text)

        sents = [s for s in data["sentences"] if isinstance(s, dict) and s.get("text") and not _CITATION_SENTENCE.match(str(s["text"]))]
        supported = [s for s in sents if s.get("supported") is True]
        grounded = len(supported) / len(sents) if sents else 1.0
        missing = [str(x) for x in data.get("missing_from_reference") or []]
        completeness = None if reference is None else (1.0 if not missing else max(0.0, 1.0 - len(missing) / max(len(missing) + len(supported), 1)))
        used = [int(i) for i in data.get("used_sources") or [] if str(i).isdigit()]
        return JudgeResult(
            ok=True, groundedness=round(grounded, 3),
            unsupported=[str(s["text"]) for s in sents if s.get("supported") is not True],
            used_sources=used, contradicts_reference=bool(data.get("contradicts_reference", False)),
            missing_from_reference=missing, completeness=completeness,
            notes=str(data.get("notes", ""))[:200], raw=gen.text,
        )
