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
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cases = load_cases(Path(args.dataset)) if args.dataset else load_cases()

    if args.mode == "retrieval":
        from app.config import settings
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
        try:
            result = AnswerEvaluator(pipe).run(cases)
        finally:
            pipe.close()

    latest, previous = report.save(args.mode, result)
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

    ok, problems = (True, []) if args.no_gate else report.gate(args.mode, result, previous)
    if not ok:
        print("\n❌ REGRESSION GATE FAILED"); [print("  -", p) for p in problems]
        return 1
    print("\n✅ gate passed" + ("" if previous else " (no previous run to compare)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
