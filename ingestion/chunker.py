"""Token-bounded semantic chunking of parsed sections (spec §5).

- Unit = one section (heading-scoped). Kept whole when it fits.
- Small sections (< min_tokens) are merged with following siblings under the same
  parent heading until they reach min_tokens.
- Oversize sections (> max_tokens) are split on block boundaries — paragraphs,
  with fenced code and tables kept atomic — into "(i/n)" parts with an
  `overlap_tokens` tail from the previous part.
- chunk_id is deterministic: kebab(source + heading path) [+ -pN].
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ingestion.markdown_parser import ParsedDocument, Section

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    chunk_id: str
    source: str
    document_title: Optional[str]
    section: str                # heading path joined with " > "
    heading_path: List[str]
    text: str                   # body only
    search_text: str            # "doc > h2 > h3\n\nbody" — what gets embedded / BM25-indexed
    part: int = 1
    n_parts: int = 1
    tokens: int = 0
    merged_from: List[str] = field(default_factory=list)


def kebab(*parts: str) -> str:
    s = " ".join(re.sub(r"\.(md|txt)$", "", p) for p in parts if p)
    s = re.sub(r"[/\\]", " ", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s.strip()).lower()
    return re.sub(r"-{2,}", "-", s)


class Chunker:
    def __init__(self, count_tokens: TokenCounter, max_tokens: int = 800, min_tokens: int = 60, overlap_tokens: int = 80):
        self.count = count_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    # ---------- public ----------

    def chunk_document(self, doc: ParsedDocument, source: str) -> List[Chunk]:
        sections = self._merge_small(doc.sections)
        chunks: List[Chunk] = []
        seen: dict = {}
        for sec in sections:
            for c in self._split_section(sec, doc, source):
                base = c.chunk_id
                if base in seen:
                    seen[base] += 1
                    c.chunk_id = f"{base}-{seen[base]}"
                else:
                    seen[base] = 1
                chunks.append(c)
        return chunks

    # ---------- merging ----------

    def _merge_small(self, sections: List[Section]) -> List[Section]:
        """A section under min_tokens absorbs what follows — its own first child (a short
        "##" intro folds into the first "###") and then siblings — until it is big enough."""
        out: List[Section] = []
        i = 0
        while i < len(sections):
            sec = sections[i]
            if self.count(sec.text) >= self.min_tokens:
                out.append(sec); i += 1; continue
            merged_titles, texts, j = [sec.title], [sec.text], i + 1
            total = self.count(sec.text)
            # A following section may join if it is a sibling or a child of anything
            # already in the group (so "도입부 → ## intro → ### child" chains collapse).
            group_paths = {tuple(sec.path), tuple(sec.path[:-1])}
            while j < len(sections) and total < self.min_tokens:
                nxt = sections[j]
                if tuple(nxt.path[:-1]) not in group_paths or self.count(nxt.text) + total > self.max_tokens:
                    break
                merged_titles.append(nxt.title); texts.append(f"### {nxt.title}\n{nxt.text}")
                total += self.count(nxt.text); j += 1
                group_paths.add(tuple(nxt.path))
            if j == i + 1:                       # nothing absorbed
                out.append(sec); i += 1; continue
            path = sec.path[:-1] + [" · ".join(merged_titles)]
            out.append(Section(" · ".join(merged_titles), sec.level, path, "\n\n".join(texts), sec.start_line))
            i = j
        return out

    # ---------- splitting ----------

    def _split_section(self, sec: Section, doc: ParsedDocument, source: str) -> List[Chunk]:
        header = self._header(doc, sec)
        full = f"{header}\n\n{sec.text}"
        if self.count(full) <= self.max_tokens:
            return [self._make(sec, doc, source, sec.text, full, 1, 1)]

        blocks = self._blocks(sec.text)
        parts: List[str] = []
        cur: List[str] = []
        budget = self.max_tokens - self.count(header) - 8
        for b in blocks:
            if self.count(b) > budget:              # a single giant block: hard-split by lines
                if cur:
                    parts.append("\n\n".join(cur)); cur = []
                parts.extend(self._hard_split(b, budget))
                continue
            candidate = "\n\n".join(cur + [b])
            if cur and self.count(candidate) > budget:
                parts.append("\n\n".join(cur))
                cur = self._overlap_tail(cur) + [b]
            else:
                cur.append(b)
        if cur:
            parts.append("\n\n".join(cur))

        n = len(parts)
        return [
            self._make(sec, doc, source, p, f"{header} ({i}/{n})\n\n{p}", i, n)
            for i, p in enumerate(parts, start=1)
        ]

    def _overlap_tail(self, blocks: List[str]) -> List[str]:
        tail: List[str] = []
        total = 0
        for b in reversed(blocks):
            t = self.count(b)
            if total + t > self.overlap_tokens:
                break
            tail.insert(0, b); total += t
        return tail

    def _hard_split(self, block: str, budget: int) -> List[str]:
        out, cur = [], []
        for line in block.splitlines():
            if cur and self.count("\n".join(cur + [line])) > budget:
                out.append("\n".join(cur)); cur = []
            cur.append(line)
        if cur:
            out.append("\n".join(cur))
        return out

    @staticmethod
    def _blocks(text: str) -> List[str]:
        """Paragraph blocks; fenced code and contiguous table rows stay atomic."""
        blocks, cur, in_fence = [], [], False
        for line in text.splitlines():
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                cur.append(line)
                if not in_fence:
                    blocks.append("\n".join(cur)); cur = []
                continue
            if in_fence:
                cur.append(line); continue
            if not line.strip():
                if cur:
                    blocks.append("\n".join(cur)); cur = []
                continue
            is_table = line.lstrip().startswith("|")
            if cur and (cur[-1].lstrip().startswith("|") != is_table):
                blocks.append("\n".join(cur)); cur = []
            cur.append(line)
        if cur:
            blocks.append("\n".join(cur))
        return blocks

    # ---------- helpers ----------

    @staticmethod
    def _header(doc: ParsedDocument, sec: Section) -> str:
        parts = ([doc.title] if doc.title else []) + (sec.path or [sec.title])
        return " > ".join(parts)

    def _make(self, sec: Section, doc: ParsedDocument, source: str, text: str, search_text: str, part: int, n: int) -> Chunk:
        base = kebab(source, *(sec.path or [sec.title]))
        cid = base if n == 1 else f"{base}-p{part}"
        return Chunk(
            chunk_id=cid, source=source, document_title=doc.title,
            section=" > ".join(sec.path or [sec.title]), heading_path=list(sec.path or [sec.title]),
            text=text, search_text=search_text, part=part, n_parts=n, tokens=self.count(search_text),
            merged_from=[t.strip() for t in sec.title.split(" · ")] if " · " in sec.title else [],
        )
