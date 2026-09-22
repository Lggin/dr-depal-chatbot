"""Korean/English tokenizer for BM25.

Kiwi morphological analysis keeps content words only (nouns, verbs, adjectives,
foreign words, numbers) and drops particles/endings. Alarm codes are lifted out
before analysis and emitted as single normalized tokens, because Kiwi would
otherwise split "VAC203" into "VAC" + "203".
"""

from typing import List

from app.router.alarm_parser import extract_alarm_codes, strip_alarm_codes

# Kiwi tag prefixes to keep (tag may carry a suffix like "VV-R").
_KEEP_PREFIXES = ("NNG", "NNP", "NNB", "NR", "SL", "SN", "SH", "VV", "VA", "XR")

_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


class KoreanTokenizer:
    def tokenize(self, text: str) -> List[str]:
        tokens = [c.lower() for c in extract_alarm_codes(text)]
        for t in _get_kiwi().tokenize(strip_alarm_codes(text)):
            if t.tag.split("-")[0] in _KEEP_PREFIXES:
                tokens.append(t.form.lower())
        return tokens
