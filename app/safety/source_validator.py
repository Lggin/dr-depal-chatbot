"""Structured sources from retrieval metadata; canonical citation line.

The API's `sources` field must come from what was actually put in the context,
not from what the model wrote. The model's own "출처:" line is used only to
select which of the provided documents it relied on; the line itself is then
replaced with a canonical one so sections are never invented.
"""

import re
from typing import List, Optional

from app.schemas.chat import DocumentSource
from app.schemas.retrieval import RetrievedDoc

_CITATION_LINE = re.compile(r"^\W*출처\W*[:：].*$(?:\n(?:[ \t]*[-*•].*|[ \t]+.*))*", re.MULTILINE)


def to_source(doc: RetrievedDoc) -> DocumentSource:
    m = doc.metadata
    return DocumentSource(
        chunk_id=doc.chunk_id,
        document=str(m.get("document_name") or m.get("source") or "unknown"),
        section=m.get("section"),
        alarm_code=m.get("alarm_code"),
        trust_level=doc.trust_level,
        system=m.get("system"),
        document_type=m.get("document_type"),
    )


_NUM_TOKEN = re.compile(r"-?\d+(?:[.,]\d+)*")
_CODE_TOKEN = re.compile(r"`([^`]{2,40})`|\b([A-Za-z_][A-Za-z0-9_.]{3,}(?:_[A-Za-z0-9]+)+)\b|\b(v0\.0\.\d+|#\d{2,5}|[A-Z]{2,4}-\d{2,4})\b")
_STOP = {"경우", "이후", "이전", "확인", "문서", "출처", "가능", "상태", "설정", "값", "때", "것", "수", "등", "및", "또는", "그리고", "해당", "관련", "기준", "정상", "범위"}

_tokenizer = None


def _content_words(text: str) -> set:
    """Kiwi content words (nouns/verbs/foreign) ≥2 chars minus generic stopwords; falls back to a regex split."""
    global _tokenizer
    try:
        if _tokenizer is None:
            from app.rag.tokenizer import KoreanTokenizer
            _tokenizer = KoreanTokenizer()
        words = _tokenizer.tokenize(text)
    except Exception:
        words = re.findall(r"[가-힣A-Za-z]{2,}", text.lower())
    return {w for w in words if len(w) >= 2 and w not in _STOP}


_DATE_LIKE = re.compile(r"^(20\d\d|\d{1,2}|0\.0)$")   # years, month/day fragments, version prefixes — shared by everything


def _anchors(text: str) -> set:
    nums = {n.replace(",", "") for n in _NUM_TOKEN.findall(text)}
    nums = {n for n in nums if len(n.lstrip("-")) >= 2 and not _DATE_LIKE.match(n.lstrip("-"))}
    codes = {next(g for g in m if g).lower() for m in _CODE_TOKEN.findall(text)}
    return nums | codes


def select_used_sources(answer: str, included: List[RetrievedDoc], min_overlap: float = 0.30) -> List[RetrievedDoc]:
    """Keep the provided docs the answer body actually draws on, decided lexically (no LLM).

    Attribution by *distinctive* anchors: a number/code in the answer is credited to the docs that
    contain it; a doc is used if it is the ONLY provided doc holding some answer anchor. Anchors
    shared by several docs (the same version string in every release note) credit none of them —
    for those, and for prose-only answers, a doc needs ≥ min_overlap of the answer's content words.
    The top-ranked doc is always kept so an answer never ends up uncited."""
    if not included:
        return []
    body = _CITATION_LINE.sub("", answer)
    a_words, a_anchors = _content_words(body), _anchors(body)
    texts = [f"{d.metadata.get('document_name', '')} {d.metadata.get('section', '')} {d.text}" for d in included]
    doc_anchors = [_anchors(t) & a_anchors for t in texts]
    kept: List[RetrievedDoc] = []
    for i, d in enumerate(included):
        others = set().union(*(doc_anchors[j] for j in range(len(included)) if j != i)) if len(included) > 1 else set()
        distinctive = doc_anchors[i] - others
        d_words = _content_words(texts[i])
        overlap = len(a_words & d_words) / len(a_words) if a_words else 0.0
        if i == 0 or distinctive or overlap >= min_overlap:
            kept.append(d)
    return kept


def build_sources(included: List[RetrievedDoc], cited_names: Optional[List[str]] = None,
                  answer: Optional[str] = None) -> List[DocumentSource]:
    """Sources actually provided, narrowed to the ones the answer relies on.

    Two filters, both deterministic: the model's own '출처:' names (when they resolve), then
    lexical usage against the answer body (when `answer` is given)."""
    docs = list(included)
    if cited_names:
        sq = lambda s: re.sub(r"\s+", "", s).lower()
        picked = [d for d in docs if any(sq(c) in sq(str(d.metadata.get("document_name") or "")) or
                                          sq(str(d.metadata.get("document_name") or "")) in sq(c) for c in cited_names)]
        if picked:
            docs = picked
    if answer is not None:
        docs = select_used_sources(answer, docs)
    return [to_source(d) for d in docs]


def format_citation(sources: List[DocumentSource]) -> str:
    parts = []
    for s in sources:
        p = s.document
        if s.section:
            p += f" §{s.section}"
        if s.alarm_code:
            p += f" ({s.alarm_code})"
        parts.append(p)
    return "출처: " + "; ".join(dict.fromkeys(parts))  # de-dup, keep order


_ORPHAN_HEADING = re.compile(r"(?:\n\s*[*#]*\s*(?:참고|참고\s*사항|출처|Sources?|References?)\s*[:：]?\s*[*]*\s*)+$", re.IGNORECASE)


def canonicalize_citations(answer: str, sources: List[DocumentSource], is_refusal: bool) -> str:
    body = _CITATION_LINE.sub("", answer).rstrip()
    body = _ORPHAN_HEADING.sub("", body).rstrip()      # e.g. a "**참고:**" heading whose only item was the citation
    if is_refusal or not sources:
        return body
    return f"{body}\n\n{format_citation(sources)}"
