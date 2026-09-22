"""Markdown → document title + heading-scoped sections.

Headings inside fenced code blocks are ignored. Text before the first level-2
heading becomes a "도입부" section. Each heading of level >= 2 opens a section;
a section's text is what follows its heading up to the next heading of any
level >= 2 — so a "##" with "###" children yields the ## intro as its own
section plus one section per child, each carrying the full heading path.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

INTRO_TITLE = "도입부"


@dataclass
class Section:
    title: str
    level: int
    path: List[str]          # heading path excluding the document title, e.g. ["1. 윗면", "두 경로의 역사"]
    text: str
    start_line: int

    @property
    def heading(self) -> str:
        return " > ".join(self.path) if self.path else self.title


@dataclass
class ParsedDocument:
    title: Optional[str]
    sections: List[Section] = field(default_factory=list)


def parse_markdown(text: str, default_title: Optional[str] = None) -> ParsedDocument:
    lines = text.splitlines()
    title = default_title
    sections: List[Section] = []
    stack: List[str] = []           # current heading path for levels >= 2 (index 0 == level 2)
    cur_title, cur_level, cur_path, cur_start = INTRO_TITLE, 2, [], 0
    buf: List[str] = []
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(cur_title, cur_level, list(cur_path), body, cur_start))

    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else _HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 1:
            if title is None or title == default_title:
                title = heading
            else:
                buf.append(line)  # a second H1 mid-document: keep as text
            continue
        flush()
        buf = []
        depth = level - 2
        stack = stack[:depth] + [heading]
        cur_title, cur_level, cur_path, cur_start = heading, level, list(stack), i + 1
    flush()
    return ParsedDocument(title=title, sections=sections)
