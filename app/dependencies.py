"""Runtime resource resolution shared by app and scripts."""

import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)


def build_chat_pipeline():
    """Wire every component from settings. Loads three models; call once at startup."""
    from app.llm.client import LlamaCppClient
    from app.llm.embedding_client import EmbeddingClient
    from app.llm.generator import ChatPipeline
    from app.llm.prompt import ContextBuilder, PromptBuilder
    from app.memory.session import SessionStore
    from app.rag import (DenseRetriever, GlossaryExpander, HybridRetriever, QdrantClient,
                         Reranker, RetrievalPipeline, SparseRetriever)

    t0 = time.perf_counter()
    device = resolve_aux_device()

    embedder = EmbeddingClient(settings.embedding_model, device=device)
    embedder.load()
    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, embedder.embedding_dim)
    store.connect()
    dense = DenseRetriever(embedder, store, top_k=settings.dense_top_k)
    sparse = SparseRetriever.from_qdrant(store, top_k=settings.sparse_top_k)
    hybrid = HybridRetriever(
        dense, sparse, store, GlossaryExpander.from_file(settings.glossary_path),
        settings.dense_top_k, settings.sparse_top_k, settings.fusion_top_k, settings.rrf_k,
        pin_min_trust=settings.exact_pin_min_trust,
    )
    reranker = Reranker(
        settings.reranker_model, device=device, max_length=settings.rerank_max_length,
        top_k=settings.rerank_top_k, gap_cutoff=settings.rerank_gap_cutoff,
    ).load()
    retrieval = RetrievalPipeline(hybrid, reranker)

    llm = LlamaCppClient(
        settings.llm_model_path, n_ctx=settings.llm_context_length, n_gpu_layers=settings.llm_n_gpu_layers,
        max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature, top_p=settings.llm_top_p,
        enable_thinking=settings.llm_enable_thinking,
    ).load()
    prompt_builder = PromptBuilder(
        ContextBuilder(llm.count_tokens, settings.context_max_tokens, settings.context_max_docs),
        history_turns=settings.history_turns,
    )
    sessions = SessionStore(settings.session_timeout_minutes, settings.history_turns)

    logger.info(f"chat pipeline ready in {time.perf_counter()-t0:.1f}s (aux={device}, chunks={sparse.size})")
    return ChatPipeline(retrieval, prompt_builder, llm, sessions)


def resolve_aux_device() -> str:
    """Device for auxiliary models (embedding, reranker).

    The LLM owns most of the VRAM. On a 16GB laptop we previously hit CUDA OOM
    with LLM + embedding + reranker co-resident, so below `aux_gpu_min_vram_gb`
    total VRAM the auxiliary models default to CPU. Override with AUX_DEVICE.
    """
    if settings.aux_device != "auto":
        return settings.aux_device
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device = "cuda" if total_gb >= settings.aux_gpu_min_vram_gb else "cpu"
        logger.info(f"aux device={device} (VRAM {total_gb:.1f}GB, threshold {settings.aux_gpu_min_vram_gb}GB)")
        return device
    except Exception as e:
        logger.warning(f"aux device detection failed ({e}), using cpu")
        return "cpu"
