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
# A real refusal *opens* with the phrase (first sentence). The same words later in
# an answer ("…문서에서는 현재 상태를 확인할 수 없으므로 다음을 점검…") are a caveat.
_FIRST_SENTENCE_END = re.compile(r"[.!?\n]")


def is_refusal_text(answer: str) -> bool:
    first = _FIRST_SENTENCE_END.split(answer.strip(), maxsplit=1)[0]
    return REFUSAL_PHRASE in first

# Numbers incl. sign and decimals. List markers ("1.", "2)") are small ints and get
# filtered later — don't exclude ')' or '.' here or "(기본 0.5)" / "v2.3." are lost.
# Lookbehind allows a letter prefix so "v2.3" / "F3" still yield their numbers.
_NUM = re.compile(r"(?<![0-9_.])[-−–]?\d+(?:[.,]\d+)*(?![A-Za-z0-9_])")
_HAN = re.compile(r"[一-鿿]")  # Chinese characters — never legitimate in a Korean answer
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
    foreign_script: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "is_refusal": self.is_refusal,
            "unsupported_numbers": self.unsupported_numbers,
            "unsupported_codes": self.unsupported_codes,
            "invalid_citations": self.invalid_citations,
            "foreign_script": self.foreign_script,
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
        is_refusal = is_refusal_text(answer)
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

        foreign = bool(_HAN.search(answer))
        ok = not unsupported_numbers and not unsupported_codes and not invalid and not foreign
        return ValidationResult(ok, is_refusal, unsupported_numbers, unsupported_codes, invalid, cited, foreign)
