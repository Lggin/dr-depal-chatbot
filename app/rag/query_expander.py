"""Glossary-based query expansion.

Appends domain synonyms for field jargon found in the query; the original query
text is always preserved. Entries in glossary.json must be backed by a reproduced
retrieval failure — no speculative synonyms (they add noise to BM25).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class GlossaryExpander:
    def __init__(self, glossary: Dict[str, List[str]]):
        # Longest keys first so "진공 발생기" wins over "진공" when both exist.
        self._entries = sorted(glossary.items(), key=lambda kv: -len(kv[0]))

    @classmethod
    def from_file(cls, path: str | Path) -> "GlossaryExpander":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info(f"glossary: {len(data)} entries from {path}")
        return cls(data)

    def expand_with_terms(self, query: str) -> Tuple[str, List[str]]:
        """Returns (expanded_query, matched_glossary_keys)."""
        matched, extra = [], []
        for key, synonyms in self._entries:
            if key in query:
                matched.append(key)
                extra.extend(s for s in synonyms if s not in query and s not in extra)
        if not extra:
            return query, []
        return f"{query} {' '.join(extra)}", matched

    def expand(self, query: str) -> str:
        return self.expand_with_terms(query)[0]
