"""Runtime resource resolution shared by app and scripts."""

import logging

from app.config import settings

logger = logging.getLogger(__name__)


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
