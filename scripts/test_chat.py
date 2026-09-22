#!/usr/bin/env python3
"""Step 7 check: full ChatPipeline (retrieval → prompt → LLM → validation) incl. a 2-turn session."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dependencies import build_chat_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "sentence_transformers", "app.rag", "app.llm.embedding_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SINGLE = [
    "VAC203이 뭐야?",
    "진공압 정상 범위 알려줘",
    "빨판이 박스를 못 잡아. 뭘 확인해야 해?",
    "VAC-203 지난번에도 있었어?",
    "로봇 컨트롤러 IP 주소가 뭐야?",
]
FOLLOW_UP = [("VAC-203이 뭐야?", "그 알람 원인이 뭐야?")]


def show(r):
    print("-" * 78)
    print(r.answer)
    print("-" * 78)
    print(f"sources: {[f'{s.document}' + (f' §{s.section}' if s.section else '') for s in r.sources]}")
    print(f"validation: {json.dumps(r.validation.model_dump(), ensure_ascii=False)}")
    print(f"timing: {r.timing_ms}  total {r.latency_ms:.0f}ms")


def main() -> int:
    pipe = build_chat_pipeline()
    for i, q in enumerate(SINGLE):
        print(f"\n{'='*78}\nQ: {q}")
        show(pipe.answer(f"single-{i}", q))

    for first, second in FOLLOW_UP:
        print(f"\n{'='*78}\n[session follow-up]\nQ1: {first}")
        show(pipe.answer("follow", first))
        print(f"\nQ2: {second}")
        show(pipe.answer("follow", second))
    pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
