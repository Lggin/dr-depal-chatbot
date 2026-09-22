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


def build_sources(included: List[RetrievedDoc], cited_names: Optional[List[str]] = None) -> List[DocumentSource]:
    """Sources actually provided; narrowed to those the model cited when that resolves to ≥1 doc."""
    all_sources = [to_source(d) for d in included]
    if cited_names:
        picked = [s for s in all_sources if any(c in s.document or s.document in c for c in cited_names)]
        if picked:
            return picked
    return all_sources


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


def canonicalize_citations(answer: str, sources: List[DocumentSource], is_refusal: bool) -> str:
    body = _CITATION_LINE.sub("", answer).rstrip()
    if is_refusal or not sources:
        return body
    return f"{body}\n\n{format_citation(sources)}"
