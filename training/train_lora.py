#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen3.5-9B on the SFT dataset (spec §28 Phase 6, §29).

Run inside venv-train (see training/requirements-train.txt):
  source venv-train/bin/activate
  env -u PYTHONPATH PYTHONNOUSERSITE=1 python training/train_lora.py --smoke      # 10 steps, verifies the stack
  env -u PYTHONPATH PYTHONNOUSERSITE=1 python training/train_lora.py             # full run

Design notes
- 4-bit NF4 base (bitsandbytes), LoRA on all linear projections of the language model
  (previous project: all-linear ≫ attention-only, r=8 sufficient, ~400 examples enough).
- Qwen3.5-9B is Qwen3_5ForConditionalGeneration (text + vision). We train text only; vision
  modules are excluded from LoRA targets by name.
- Loss on assistant tokens only (prompt/context masked) — the model must learn to ANSWER in our
  format, not to reproduce the retrieved documents.
- Facts remain in RAG: the training prompt already contains the retrieved context.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_examples(rows: List[Dict], tokenizer, max_len: int) -> List[Dict]:
    """Tokenize with the model's chat template; mask everything before the assistant turn."""
    out = []
    for r in rows:
        msgs = r["messages"]
        prompt_ids = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        full_ids = tokenizer.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
        if not isinstance(prompt_ids, list):
            prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
        if len(full_ids) > max_len:
            continue  # keep the eval-time budget; long contexts are cut upstream anyway
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        out.append({"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/hf/Qwen3.5-9B"))
    ap.add_argument("--train", default=str(ROOT / "training/datasets/sft.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "training/datasets/sft.val.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "training/output/qwen35-9b-depal-lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096,
                    help="measured on the SFT set with the Qwen tokenizer: p50 2362 / p90 3396 / max 3617 → 4096 drops nothing")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="10 steps on 16 examples to validate the environment")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token

    train_rows, val_rows = load_jsonl(Path(args.train)), load_jsonl(Path(args.val))
    if args.smoke:
        train_rows, val_rows = train_rows[:16], val_rows[:4]
    train_ds = Dataset.from_list(build_examples(train_rows, tok, args.max_len))
    val_ds = Dataset.from_list(build_examples(val_rows, tok, args.max_len))
    print(f"examples: train={len(train_ds)} val={len(val_ds)} (dropped {len(train_rows)-len(train_ds)} over {args.max_len} tokens)")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    # Not prepare_model_for_kbit_training(): it upcasts every non-quantized parameter to fp32, and with a
    # 248k-token vocabulary that is ~4 GB each for embed_tokens and lm_head (measured: 11.2 GB after wrap
    # vs ~6 GB for the 4-bit body). Keep them bf16; only enable checkpointing + input grads for LoRA.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # LoRA on the language model's linear projections only (skip vision tower / merger).
    # Qwen3.5-9B is a hybrid: 24 of 32 layers are Gated DeltaNet ("linear_attn": in_proj_qkv /
    # in_proj_z / out_proj, plus tiny in_proj_a / in_proj_b gates) and 8 are full attention
    # (q/k/v/o_proj). "all-linear" therefore has to name both families or 3/4 of the layers get no adapter.
    LM_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                 "in_proj_qkv", "in_proj_z", "out_proj")
    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and "visual" not in n and "vision" not in n
                      and n.split(".")[-1] in LM_LINEAR})
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=targets, exclude_modules=r".*(visual|vision).*")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    print(f"LoRA targets: {targets}; load+wrap {time.perf_counter()-t0:.0f}s; VRAM {torch.cuda.memory_allocated()/2**30:.1f} GB")

    steps_per_epoch = math.ceil(len(train_ds) / args.grad_accum)
    total_steps = 10 if args.smoke else math.ceil(steps_per_epoch * args.epochs)
    targs = TrainingArguments(   # transformers 5: warmup_ratio is gone, use warmup_steps
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=10 if args.smoke else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, round(total_steps * 0.05)), weight_decay=0.0,
        logging_steps=1 if args.smoke else 5, eval_strategy="steps", eval_steps=10 if args.smoke else max(10, steps_per_epoch // 2),
        save_strategy="no" if args.smoke else "steps", save_steps=max(10, steps_per_epoch), save_total_limit=2,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", report_to=[], dataloader_pin_memory=False, remove_unused_columns=False,
    )
    import torch.nn.functional as F

    class AnswerOnlyTrainer(Trainer):
        """Compute logits only at positions that predict an answer token.

        With a 248k vocabulary, full-sequence logits for a 4k prompt are 4096×248320 fp32 ≈ 4 GB (+ grad):
        the smoke run OOMed at 13.4 GB in backward. Only ~100–200 assistant positions carry loss, so we pass
        their indices as `logits_to_keep` and do the shifted cross-entropy ourselves. Batch size is 1."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            assert labels.shape[0] == 1, "AnswerOnlyTrainer assumes per_device batch size 1"
            idx = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]  # position t predicts token t+1
            out = model(**inputs, logits_to_keep=idx)
            logits = out.logits[0].float()
            loss = F.cross_entropy(logits, labels[0, idx + 1])
            return (loss, out) if return_outputs else loss

    trainer = AnswerOnlyTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                                data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    metrics = trainer.evaluate()
    print("eval:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    if not args.smoke:
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        print(f"saved adapter to {args.out}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GB, total {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
