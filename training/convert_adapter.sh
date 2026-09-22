#!/usr/bin/env bash
# Convert a PEFT LoRA adapter (training/train_lora.py output) into a GGUF adapter that llama.cpp
# applies on top of the SAME quantized base the service already runs (no merge, no re-quantization,
# so the A/B is exactly "base" vs "base + adapter").
#
#   training/convert_adapter.sh training/output/qwen35-9b-depal-lora [models/gguf/depal-lora.gguf]
#
# Then: ./run.sh python3 scripts/run_eval.py --mode full --dataset ... --collection ... --judge --lora models/gguf/depal-lora.gguf
# Requires third_party/llama.cpp (git clone --depth 1 https://github.com/ggml-org/llama.cpp) and its
# gguf-py installed into venv-train (pip install -e third_party/llama.cpp/gguf-py).
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER=${1:?adapter dir}
OUT=${2:-models/gguf/$(basename "$ADAPTER").gguf}
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 venv-train/bin/python \
  third_party/llama.cpp/convert_lora_to_gguf.py "$ADAPTER" --base models/hf/Qwen3.5-9B --outtype f16 --outfile "$OUT"
