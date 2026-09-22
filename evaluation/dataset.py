"""Evaluation dataset loading/validation (spec §26)."""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

DEFAULT_PATH = Path(__file__).parent / "datasets" / "test_questions.json"


class EvalCase(BaseModel):
    id: str
    category: str = "uncategorized"
    question: str
    history: List[str] = []                 # prior user questions in the same session (follow-ups)
    expected_intent: str
    expected_alarm_code: Optional[str] = None
    expected_source: List[str] = []         # chunk_ids; empty for out-of-scope
    must_include: List[str] = []
    must_not_include: List[str] = []
    expect_refusal: bool = False
    filters: Optional[dict] = None
    note: Optional[str] = None


def load_cases(path: Path = DEFAULT_PATH) -> List[EvalCase]:
    cases = [EvalCase(**c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    return cases
