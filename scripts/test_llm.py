#!/usr/bin/env python3
"""Step 5 check: load Qwen3.5-9B Q4 via llama.cpp, render the chat template with
thinking disabled, generate a grounded Korean answer, report speed and VRAM."""

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.llm.client import LlamaCppClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SYSTEM = (
    "You are a technical assistant for a depalletizing system. Answer in Korean using only the "
    "provided documents. Never invent values. Cite the source document and section."
)
CONTEXT = (
    "[SOURCE 1] Document: Vacuum System Manual | Section: 6.3 | Alarm: VAC-203 | Trust: Official\n"
    "[Alarm] VAC-203 [Title] Vacuum Pressure Low [Symptom] 진공압이 정상 범위에 도달하지 못함. "
    "[Cause] 1. Vacuum hose leakage 2. Suction cup damage 3. Vacuum generator malfunction "
    "[Normal Range] -60 ~ -80 kPa"
)


def vram_mib() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return "n/a"


def main() -> int:
    print(f"VRAM before load: {vram_mib()}")
    llm = LlamaCppClient(
        settings.llm_model_path, n_ctx=settings.llm_context_length, n_gpu_layers=settings.llm_n_gpu_layers,
        max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
        enable_thinking=settings.llm_enable_thinking,
    ).load()
    print(f"VRAM after load:  {vram_mib()}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{CONTEXT}\n\n질문: VAC-203이 뭐야? 정상 진공압 범위도 알려줘."},
    ]
    prompt = llm.render(messages)
    print(f"\n--- rendered prompt tail (thinking={'on' if settings.llm_enable_thinking else 'off'}) ---")
    print(prompt[-200:])
    print(f"prompt tokens: {llm.count_tokens(prompt)}")

    r = llm.generate(messages)
    print("\n--- answer ---")
    print(r.text)
    tps = r.completion_tokens / (r.latency_ms / 1000) if r.latency_ms else 0
    print(f"\n{r.prompt_tokens} prompt + {r.completion_tokens} completion tokens in {r.latency_ms:.0f}ms "
          f"→ {tps:.1f} tok/s  finish={r.finish_reason}")
    print(f"VRAM during:      {vram_mib()}")

    # basic grounding checks
    ok = ("-60" in r.text and "-80" in r.text) and ("6.3" in r.text or "Vacuum System Manual" in r.text)
    print("\ngrounding check (range + source cited):", "✅" if ok else "❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
