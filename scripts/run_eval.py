#!/usr/bin/env python3
"""Run the evaluation suite.

  ./run.sh python3 scripts/run_eval.py --mode retrieval   # no LLM, seconds
  ./run.sh python3 scripts/run_eval.py --mode full        # through the LLM, minutes
  ./run.sh python3 scripts/run_eval.py --mode full --show-failures
Exit code 1 if a gated metric regressed vs. evaluation/results/latest_<mode>.json.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import report
from evaluation.dataset import load_cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--dataset", default=None, help="eval JSON (default: evaluation/datasets/test_questions.json)")
    ap.add_argument("--collection", default=None, help="Qdrant collection to evaluate against (default: settings)")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--judge", action="store_true", help="full mode: also run the LLM judge (groundedness/completeness/over-citation)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    ap.add_argument("--show-ungrounded", action="store_true", help="print judge-flagged unsupported sentences")
    ap.add_argument("--lora", default=None, help="full mode: GGUF LoRA adapter applied on the base model (A/B of Phase 6 training)")
    ap.add_argument("--variant", default=None,
                    help="label for an A/B variant (e.g. 'lora'): results are saved under <tag>+<variant> and gated against the plain <tag> baseline")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from app.config import settings
    if args.collection:
        settings.qdrant_collection = args.collection
    if args.lora:
        settings.llm_lora_path = args.lora
        args.variant = args.variant or "lora"
    cases = load_cases(Path(args.dataset)) if args.dataset else load_cases()
    if args.limit:
        cases = cases[: args.limit]
    tag = Path(args.dataset).stem if args.dataset else "sample"
    if args.collection:
        tag = f"{tag}__{args.collection}"   # results/gates are per (dataset, collection)
    baseline_tag = tag
    if args.variant:
        tag = f"{tag}+{args.variant}"
    print(f"dataset={tag} ({len(cases)} cases)  collection={settings.qdrant_collection}  mode={args.mode}"
          + (f"  lora={settings.llm_lora_path}" if settings.llm_lora_path else ""))

    if args.mode == "retrieval":
        from app.dependencies import resolve_aux_device
        from app.llm.embedding_client import EmbeddingClient
        from app.rag import (DenseRetriever, GlossaryExpander, HybridRetriever, QdrantClient,
                             Reranker, RetrievalPipeline, SparseRetriever)
        from evaluation.retrieval_eval import RetrievalEvaluator

        device = resolve_aux_device()
        emb = EmbeddingClient(settings.embedding_model, device=device); emb.load()
        store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, emb.embedding_dim); store.connect()
        hybrid = HybridRetriever(DenseRetriever(emb, store, settings.dense_top_k), SparseRetriever.from_qdrant(store, top_k=settings.sparse_top_k),
                                 store, GlossaryExpander.from_file(settings.glossary_path), settings.dense_top_k, settings.sparse_top_k,
                                 settings.fusion_top_k, settings.rrf_k, pin_min_trust=settings.exact_pin_min_trust)
        reranker = Reranker(settings.reranker_model, device=device, max_length=settings.rerank_max_length,
                            top_k=settings.rerank_top_k, gap_cutoff=settings.rerank_gap_cutoff).load()
        result = RetrievalEvaluator(RetrievalPipeline(hybrid, reranker)).run(cases)
    else:
        from app.dependencies import build_chat_pipeline
        from evaluation.answer_eval import AnswerEvaluator
        pipe = build_chat_pipeline()
        judge = None
        if args.judge:
            from evaluation.judge import AnswerJudge
            judge = AnswerJudge(pipe.llm)
        try:
            result = AnswerEvaluator(pipe, judge=judge).run(cases)
        finally:
            pipe.close()

    latest, previous = report.save(args.mode, result, tag=tag, collection=settings.qdrant_collection,
                                   baseline_tag=baseline_tag if args.variant else None)
    print(report.markdown(args.mode, result, previous))
    print(f"\nsaved: {latest}")

    if args.show_failures:
        key = "failures" if args.mode == "full" else "misses_at_5"
        bad = set(result["summary"].get(key, []))
        for r in result["rows"]:
            if r["id"] in bad:
                print(f"\n--- {r['id']} ({r['category']}) {r['question']}")
                if args.mode == "full":
                    print(f"missing={r['missing']} forbidden={r['forbidden']} refusal_ok={r['refusal_ok']} sources={r['sources']}")
                    print(r["answer"][:600])
                else:
                    print(f"expected={r['expected']} final={r['final']} fused_top={r['fused_top']}")

    if args.show_ungrounded and args.mode == "full":
        for r in result["rows"]:
            j = r.get("judge")
            if j and j["ok"] and (j["unsupported"] or j["missing_from_reference"] or j["overcited"]):
                print(f"\n--- {r['id']} groundedness={j['groundedness']} completeness={j['completeness']} overcited={j['overcited']}")
                for s in j["unsupported"]:
                    print(f"   ✗ {s[:160]}")
                for m in j["missing_from_reference"]:
                    print(f"   ⊘ missing: {m[:160]}")

    ok, problems = (True, []) if args.no_gate else report.gate(args.mode, result, previous)
    if not ok:
        print("\n❌ REGRESSION GATE FAILED"); [print("  -", p) for p in problems]
        return 1
    print("\n✅ gate passed" + ("" if previous else " (no previous run to compare)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
