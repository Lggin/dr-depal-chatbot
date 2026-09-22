"""Application configuration."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App
    app_name: str = "Depalletizer Chatbot"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM (llama.cpp GGUF)
    llm_model: str = "Qwen/Qwen3.5-9B"
    llm_gguf_repo: str = "unsloth/Qwen3.5-9B-GGUF"
    llm_gguf_file: str = "Qwen3.5-9B-Q4_K_M.gguf"
    llm_model_path: str = "models/gguf/Qwen3.5-9B-Q4_K_M.gguf"
    llm_context_length: int = 8192
    llm_n_gpu_layers: int = -1  # -1 = all layers on GPU
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.1  # spec range 0.1-0.3; lower end reduces run-to-run variance
    llm_top_p: float = 0.9
    llm_enable_thinking: bool = False
    # Optional QLoRA adapter (Phase 6) as GGUF, applied on the same quantized base — "" = base model only.
    llm_lora_path: str = ""
    llm_lora_scale: float = 1.0

    # Prompt / context budget (spec §13: 3-5 chunks, ~2-4k tokens)
    context_max_tokens: int = 3000
    context_max_docs: int = 5
    history_turns: int = 1  # longer histories made the model parrot old answers

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Auxiliary model placement: "auto" | "cuda" | "cpu".
    # The 20GB rule dated from Qwen3-14B (~10GB). Qwen3.5-9B Q4 takes ~6GB, and the CPU
    # reranker needs ~15s/query on 600-token chunks, so on 16GB the aux models go to GPU.
    aux_device: str = "auto"
    aux_gpu_min_vram_gb: float = 12.0

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "depalletizer_docs"

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 4
    rrf_k: int = 60
    # Exact alarm-code hits at/above this trust are pinned to the top of the context;
    # lower-trust hits (maintenance logs, engineer notes) compete through the reranker
    # instead, so a definitional question isn't force-fed field anecdotes.
    exact_pin_min_trust: int = 85
    # Cut candidates where consecutive reranker scores drop by more than this.
    # Absolute thresholds failed before (score scale varies per query); a gap
    # between neighbours is stable. Applied BEFORE the top_k cap, never after.
    rerank_gap_cutoff: float = 3.0
    rerank_max_length: int = 1024  # model supports 8192; 1024 keeps CPU latency sane
    glossary_path: str = "corpus/glossary/glossary.json"
    corpus_config: str = "configs/corpus.yaml"
    troubleshooting_config: str = "configs/troubleshooting.yaml"

    # Machine tools (Phase 4): "mock" reads configs/machine.yaml scenarios; "none" disables tools
    tools_backend: str = "mock"
    machine_config: str = "configs/machine.yaml"

    # Database
    database_url: str = "sqlite:///./data/chatbot.db"

    # Skip loading models at API startup (tests / schema checks)
    skip_model_load: bool = False

    # Paths
    corpus_path: str = "corpus"
    data_path: str = "data"
    log_path: str = "logs"

    # Session
    session_timeout_minutes: int = 30
    max_history: int = 10

    # Logging
    log_level: str = "INFO"

    # GPU
    cuda_visible_devices: str = "0"

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def ensure_directories():
    """Create necessary directories."""
    settings = get_settings()

    for path_str in [settings.data_path, settings.log_path, settings.corpus_path]:
        Path(path_str).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = get_settings()
