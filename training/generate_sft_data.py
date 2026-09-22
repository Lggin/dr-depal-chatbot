#!/usr/bin/env python3
"""Build the SFT dataset (spec §28 Phase 6 "Synthetic QA Dataset", §29 goals).

What the model should learn here is BEHAVIOUR, not facts: answer directly first, keep every
figure the sources give, cite with the canonical "출처:" line, refuse with the fixed sentence
when the documents don't cover the question, keep maintenance history separate from official
manuals, and never claim live values without [MACHINE STATE]. Facts stay in RAG — every example
carries the retrieved context in the prompt exactly as the serving PromptBuilder renders it.

Sources of examples:
  * the generated dataset (277 chunk-derived Q/A with reference_answer)
    → positive examples: prompt = real retrieval context, target = reference answer restyled
    (knowhow_questions 40 and the sample 39 are the A/B evaluation sets → held out, never trained on)
  * refusal examples: out-of-scope questions paired with the (irrelevant) top retrieval
    → target = "현재 제공된 문서에서는 해당 내용을 확인할 수 없습니다." + one line on what is missing
  * (optional) reviewed candidates from scripts/harvest_logs.py

Output: JSONL of {"messages": [...], "meta": {...}} in the model's chat format, split train/val.

  ./run.sh python3 training/generate_sft_data.py --collection knowhow_docs --out training/datasets/sft.jsonl
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.prompt import REFUSAL_PHRASE_FULL, SYSTEM_PROMPT, ContextBuilder, PromptBuilder
from app.safety.source_validator import build_sources, format_citation
from evaluation.dataset import load_cases

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.WARNING)

OUT_OF_SCOPE = [
    "로봇 컨트롤러 IP 주소가 뭐야?", "감속기 윤활유 교체 주기가 얼마야?", "설비 담당자 연락처 알려줘",
    "이 라인의 시간당 생산량 목표가 얼마야?", "PLC 프로그램 백업은 어디에 저장돼?", "안전 펜스 인터록 리셋 절차 알려줘",
    "카메라 렌즈 청소 주기는?", "로봇 컨트롤러 펌웨어 버전 확인은 어떻게 해?", "비상정지 후 재가동 승인자는 누구야?",
    "컨베이어 모터 정격 전류가 얼마야?", "그리퍼 교체 부품 발주 코드가 뭐야?", "현장 와이파이 비밀번호 알려줘",
]


def restyle_reference(ref: str, sources) -> str:
    """Reference answers are terse facts; make the target read like our answer format."""
    text = ref.strip()
    text = re.sub(r"\s*\n\s*", "\n", text)
    if sources:
        text += "\n\n" + format_citation(sources)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="knowhow_docs")
    # HOLD-OUT: knowhow_questions.json (40) and test_questions.json (39) are the A/B evaluation sets and
    # must never be trained on — the adapter is judged on them. Training uses the generated set only.
    ap.add_argument("--datasets", nargs="*", default=["evaluation/datasets/knowhow_generated.json"])
    ap.add_argument("--sample-dataset", default=None,
                    help="sample-corpus cases (contexts from the sample collection); off by default = held out for A/B")
    ap.add_argument("--sample-collection", default="depalletizer_docs")
    ap.add_argument("--out", default="training/datasets/sft.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-context-tokens", type=int, default=2200)
    args = ap.parse_args()

    from app.config import settings
    from app.dependencies import resolve_aux_device
    from app.llm.embedding_client import EmbeddingClient
    from app.rag import DenseRetriever, GlossaryExpander, HybridRetriever, QdrantClient, Reranker, RetrievalPipeline, SparseRetriever

    device = resolve_aux_device()
    emb = EmbeddingClient(settings.embedding_model, device=device); emb.load()
    reranker = Reranker(settings.reranker_model, device=device, max_length=settings.rerank_max_length,
                        top_k=settings.rerank_top_k, gap_cutoff=settings.rerank_gap_cutoff).load()

    def pipeline_for(collection: str) -> RetrievalPipeline:
        store = QdrantClient(settings.qdrant_url, collection, emb.embedding_dim); store.connect()
        hybrid = HybridRetriever(DenseRetriever(emb, store, settings.dense_top_k), SparseRetriever.from_qdrant(store, top_k=settings.sparse_top_k),
                                 store, GlossaryExpander.from_file(settings.glossary_path), settings.dense_top_k, settings.sparse_top_k,
                                 settings.fusion_top_k, settings.rrf_k, pin_min_trust=settings.exact_pin_min_trust)
        return RetrievalPipeline(hybrid, reranker)

    # token budget via the embedding tokenizer (close enough to the LLM's for a training-time cap)
    pb = PromptBuilder(ContextBuilder(emb.count_tokens, args.max_context_tokens, settings.context_max_docs), system_prompt=SYSTEM_PROMPT, history_turns=0)

    examples: List[Dict] = []
    stats = {"positive": 0, "refusal": 0, "skipped_no_ref": 0, "skipped_no_ctx": 0}

    def add_positive(case, rp: RetrievalPipeline, origin: str):
        if not case.reference_answer or case.expect_refusal:
            stats["skipped_no_ref"] += 1; return
        res = rp.search(case.question)
        if not res.docs:
            stats["skipped_no_ctx"] += 1; return
        messages, built = pb.build(case.question, res.docs)
        # cite the expected chunks if they made it into the context, else what was provided
        expected = set(case.expected_source or [])
        cited_docs = [d for d in built.included if d.chunk_id in expected] or built.included[:1]
        target = restyle_reference(case.reference_answer, build_sources(cited_docs))
        examples.append({"messages": messages + [{"role": "assistant", "content": target}],
                         "meta": {"id": case.id, "origin": origin, "kind": "positive", "chunks": [d.chunk_id for d in built.included]}})
        stats["positive"] += 1

    def add_refusal(question: str, rp: RetrievalPipeline, origin: str):
        res = rp.search(question)
        messages, built = pb.build(question, res.docs)
        target = f"{REFUSAL_PHRASE_FULL}\n\n제공된 문서는 알람·설정·절차·이력을 다루며, 질문하신 항목은 포함되어 있지 않습니다."
        examples.append({"messages": messages + [{"role": "assistant", "content": target}],
                         "meta": {"id": None, "origin": origin, "kind": "refusal", "chunks": [d.chunk_id for d in built.included]}})
        stats["refusal"] += 1

    rp_main = pipeline_for(args.collection)
    for ds in args.datasets:
        for case in load_cases(ROOT / ds):
            add_positive(case, rp_main, Path(ds).stem)
    for q in OUT_OF_SCOPE:
        add_refusal(q, rp_main, "out_of_scope")

    if args.sample_dataset:
        rp_sample = pipeline_for(args.sample_collection)
        for case in load_cases(ROOT / args.sample_dataset):
            if case.expect_refusal and case.expected_source == []:
                add_refusal(case.question, rp_sample, "sample_out_of_scope")
            elif case.reference_answer:
                add_positive(case, rp_sample, "sample")

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    val_path = out.with_name(out.stem + ".val.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for ex in examples[n_val:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for ex in examples[:n_val]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    lengths = sorted(sum(len(m["content"]) for m in ex["messages"]) for ex in examples)
    print(json.dumps({**stats, "train": len(examples) - n_val, "val": n_val,
                      "chars_p50": lengths[len(lengths)//2] if lengths else 0, "chars_max": lengths[-1] if lengths else 0}, ensure_ascii=False))
    print(f"wrote {out} and {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
