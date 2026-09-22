"""Regex-based alarm code extraction and normalization.

Deterministic — never delegated to the LLM or to vector search.

    VAC-203 / VAC203 / VAC_203 / vac 203 / E203 / SV032  ->  VAC-203 / E-203 / SV-032
"""

import re
from typing import List

# prefix: 1-4 letters, optional separator, 2-4 digits. Word-bounded so that
# "DP-01" still matches (machine ids share the shape) but "1.0mm" / "9개" don't.
ALARM_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,4})[\s\-_]?(\d{2,4})(?![A-Za-z0-9])")


def normalize_alarm_code(raw: str) -> str:
    m = ALARM_CODE_RE.fullmatch(raw.strip())
    if not m:
        raise ValueError(f"not an alarm code: {raw!r}")
    return f"{m.group(1).upper()}-{m.group(2)}"


def extract_alarm_codes(text: str) -> List[str]:
    """Normalized codes in order of first appearance, de-duplicated."""
    seen, out = set(), []
    for prefix, num in ALARM_CODE_RE.findall(text):
        code = f"{prefix.upper()}-{num}"
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def strip_alarm_codes(text: str) -> str:
    """Remove code spans so a morphological tokenizer doesn't split them."""
    return ALARM_CODE_RE.sub(" ", text)
