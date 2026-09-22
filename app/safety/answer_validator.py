"""Grounding checks on the generated answer (spec §23).

Deterministic string checks — no LLM judge in V1:
  1. numbers in the answer must appear in context / question / machine state
  2. alarm codes in the answer must appear in context / question
  3. cited document names must be among the documents actually provided
  4. detect the fixed refusal phrase
Results are flags for logging and the API response; they do not block.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from app.router.alarm_parser import extract_alarm_codes
from app.schemas.retrieval import RetrievedDoc

REFUSAL_PHRASE = "현재 제공된 문서에서는"

# Numbers incl. sign and decimals; skip bare 1-digit list markers like "1." / "2)".
_NUM = re.compile(r"(?<![A-Za-z0-9_])[-−–]?\d+(?:[.,]\d+)*(?![A-Za-z0-9_.)])")
_CITATION_LINE = re.compile(r"^\W*출처\W*[:：](.*)$", re.MULTILINE)


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("−", "-").replace("–", "-")
    neg = s.startswith("-")
    s = re.sub(r"^0+(?=\d)", "", s.lstrip("-"))  # "09" -> "9", but "0.5" stays "0.5"
    return ("-" if neg else "") + s


def extract_numbers(text: str) -> Set[str]:
    return {_norm_num(m.group(0)) for m in _NUM.finditer(text)}


def _parse_citations(text: str) -> List[str]:
    names: List[str] = []
    for m in _CITATION_LINE.finditer(text):
        for part in re.split(r"[;,]", m.group(1)):
            name = re.split(r"§|\(", part)[0].strip(" *[]")
            if name:
                names.append(name)
    return names


@dataclass
class ValidationResult:
    ok: bool
    is_refusal: bool
    unsupported_numbers: List[str] = field(default_factory=list)
    unsupported_codes: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    cited_documents: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "is_refusal": self.is_refusal,
            "unsupported_numbers": self.unsupported_numbers,
            "unsupported_codes": self.unsupported_codes,
            "invalid_citations": self.invalid_citations,
        }


class AnswerValidator:
    # Numbers that legitimately appear without being in the source (list markers, years in dates
    # are handled by normalization; small integers used as enumerators are ignored).
    IGNORE_SMALL_INTS_UPTO = 9

    def validate(
        self,
        answer: str,
        context_docs: Iterable[RetrievedDoc],
        question: str,
        extra_evidence: Optional[str] = None,
    ) -> ValidationResult:
        docs = list(context_docs)
        # Section/alarm metadata is shown to the model in the [SOURCE] header, so it counts as evidence.
        header_bits = [str(d.metadata.get(k) or "") for d in docs for k in ("section", "alarm_code", "document_name")]
        evidence = "\n".join([question] + [d.text for d in docs] + header_bits + ([extra_evidence] if extra_evidence else []))
        is_refusal = REFUSAL_PHRASE in answer
        # The model's own citation line is replaced later; judge only the body.
        body = _CITATION_LINE.sub("", answer)

        ev_nums = extract_numbers(evidence)
        ans_nums = {
            n for n in extract_numbers(body)
            if not (n.lstrip("-").isdigit() and int(n.lstrip("-")) <= self.IGNORE_SMALL_INTS_UPTO)
        }
        unsupported_numbers = sorted(ans_nums - ev_nums) if not is_refusal else []

        ev_codes = set(extract_alarm_codes(evidence))
        unsupported_codes = sorted(set(extract_alarm_codes(body)) - ev_codes)

        provided = {str(d.metadata.get("document_name") or "") for d in docs} | {str(d.metadata.get("source") or "") for d in docs}
        provided.discard("")
        cited = _parse_citations(answer)
        invalid = [c for c in cited if not any(c in p or p in c for p in provided)]

        ok = not unsupported_numbers and not unsupported_codes and not invalid
        return ValidationResult(ok, is_refusal, unsupported_numbers, unsupported_codes, invalid, cited)
