"""Build Qdrant payload filters from plain dicts.

{"alarm_code": "VAC-203"}                     -> must match exactly
{"document_type": ["alarm_manual", "manual"]} -> must match any of
{"trust_level": {"gte": 85}}                  -> range
Keys with value None are ignored.
"""

from typing import Any, Dict, Optional
from qdrant_client.http import models

# Payload fields that get a keyword index so exact-match filtering is fast.
INDEXED_KEYWORD_FIELDS = ("alarm_code", "document_type", "machine", "system", "component", "language")
INDEXED_INTEGER_FIELDS = ("trust_level",)


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
    if not filters:
        return None
    must = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, dict):
            must.append(models.FieldCondition(key=key, range=models.Range(**value)))
        elif isinstance(value, (list, tuple, set)):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None
