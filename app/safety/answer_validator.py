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
# A real refusal *opens* with the phrase (first sentence) and then has nothing grounded to
# offer. The same words later in an answer are a caveat; and an answer that opens with the
# phrase but goes on to cite a section and list causes is a hedged answer, not a refusal.
_FIRST_SENTENCE_END = re.compile(r"[.!?\n]")
_SECTION_REF = re.compile(r"§|\[SOURCE|Manual|매뉴얼|Log|Guide|SOP|Notes?\b|문서에\s*따르면|에\s*따르면", re.IGNORECASE)
REFUSAL_MAX_BODY_CHARS = 220   # real refusals run ~100-150 chars; hedged-but-grounded answers run 300+


def is_refusal_text(answer: str) -> bool:
    text = answer.strip()
    first = _FIRST_SENTENCE_END.split(text, maxsplit=1)[0]
    if REFUSAL_PHRASE not in first:
        return False
    body = _CITATION_LINE.sub("", text)
    return len(body) <= REFUSAL_MAX_BODY_CHARS or not _SECTION_REF.search(body[len(first):])

# Numbers incl. sign and decimals. List markers ("1.", "2)") are small ints and get
# filtered later — don't exclude ')' or '.' here or "(기본 0.5)" / "v2.3." are lost.
# Lookbehind allows a letter prefix so "v2.3" / "F3" still yield their numbers.
_NUM = re.compile(r"(?<![0-9_.])[-−–]?\d+(?:[.,]\d+)*(?![A-Za-z0-9_])")
_HAN = re.compile(r"[一-鿿]")  # Chinese characters — never legitimate in a Korean answer
# Answer phrases that assert a non-official source exists -> the context must actually contain one.
_SOURCE_TYPE_CLAIMS = {
    "maintenance": re.compile(r"(유지\s*보수\s*(이력|기록|로그)|정비\s*(이력|기록)|maintenance\s*(log|history|record))", re.IGNORECASE),
    "engineer_note": re.compile(r"(엔지니어\s*(메모|노트|경험)|현장\s*(메모|노트)|engineer\s*note)", re.IGNORECASE),
}
_CITATION_LINE = re.compile(r"^\W*출처\W*[:：](.*)$", re.MULTILINE)


_NEGATED = re.compile(r"(없|않|못|불가|미포함|제공되지|확인할\s*수\s*없)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _asserts_source(body: str, rx: "re.Pattern[str]") -> bool:
    """True if some sentence mentions the source type WITHOUT negating it
    ("유지보수 기록은 없습니다" is honest; "유지보수 이력에 따르면 …" fabricates)."""
    for sentence in _SENTENCE_SPLIT.split(body):
        if rx.search(sentence) and not _NEGATED.search(sentence):
            return True
    return False


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("−", "-").replace("–", "-")
    neg = s.startswith("-")
    s = re.sub(r"^0+(?=\d)", "", s.lstrip("-"))  # "09" -> "9", but "0.5" stays "0.5"
    return ("-" if neg else "") + s


def extract_numbers(text: str) -> Set[str]:
    return {_norm_num(m.group(0)) for m in _NUM.finditer(text)}


def _parse_citations(text: str) -> List[str]:
    """Document names from '출처:' lines. Entries are ';'-separated; a document name ends at '§'.
    Commas and parentheses are legal inside names/sections ("02. 타임라인 — 2026년 3월 ~ 8월",
    "(2026-08-21, v0.0.59 #470)"), so we never split on them."""
    names: List[str] = []
    for m in _CITATION_LINE.finditer(text):
        for part in m.group(1).split(";"):
            part = re.sub(r"\[?SOURCE\s*\d+\]?", " ", part, flags=re.IGNORECASE)
            name = part.split("§")[0]
            name = re.sub(r"\s*\([^()]*\)\s*$", "", name)      # trailing parenthetical (alarm code, date)
            name = name.strip(" *[]-–—:")
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
    fabricated_source_types: List[str] = field(default_factory=list)   # e.g. ["maintenance"] claimed but not provided

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "is_refusal": self.is_refusal,
            "unsupported_numbers": self.unsupported_numbers,
            "unsupported_codes": self.unsupported_codes,
            "invalid_citations": self.invalid_citations,
            "foreign_script": self.foreign_script,
            "fabricated_source_types": self.fabricated_source_types,
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
        # the model sometimes spaces out names ("2026 년 3 월" for "2026년 3월") — compare without whitespace
        provided_n = {_squash(p) for p in provided}
        invalid = [c for c in cited if not any(_squash(c) in p or p in _squash(c) for p in provided_n)]

        foreign = bool(_HAN.search(answer))
        provided_types = {str(d.metadata.get("document_type") or "") for d in docs}
        fabricated = [t for t, rx in _SOURCE_TYPE_CLAIMS.items() if t not in provided_types and _asserts_source(body, rx)]
        ok = not unsupported_numbers and not unsupported_codes and not invalid and not foreign and not fabricated
        return ValidationResult(ok, is_refusal, unsupported_numbers, unsupported_codes, invalid, cited, foreign, fabricated)
