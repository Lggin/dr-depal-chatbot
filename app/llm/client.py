"""Local LLM client (llama.cpp backend).

The chat template embedded in the GGUF is rendered here with Jinja2 rather than
via create_chat_completion(), so we control template kwargs — in particular
`enable_thinking=False`: Qwen3.5's thinking mode adds latency and, for a
grounded industrial assistant, no accuracy that the retrieval stack doesn't
already provide.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Protocol

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    first_token_ms: Optional[float] = None
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict, repr=False)


class LLMClient(Protocol):
    def load(self) -> "LLMClient": ...
    def generate(self, messages: List[Message], **kw) -> GenerationResult: ...
    def stream(self, messages: List[Message], **kw) -> Iterator[str]: ...
    def count_tokens(self, text: str) -> int: ...


class LlamaCppClient:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        n_threads: Optional[int] = None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.n_threads = n_threads
        self.llm = None
        self._template = None
        self._eos: List[str] = []

    def load(self) -> "LlamaCppClient":
        from jinja2 import Environment
        from llama_cpp import Llama

        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False,
        )
        meta = self.llm.metadata
        tmpl = meta.get("tokenizer.chat_template")
        if not tmpl:
            raise RuntimeError("GGUF has no tokenizer.chat_template; cannot render prompts")
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
        self._template = env.from_string(tmpl)
        # Qwen ends turns with <|im_end|>; keep <|endoftext|> as a safety net.
        self._eos = ["<|im_end|>", "<|endoftext|>"]
        logger.info(
            f"✅ LLM loaded in {time.perf_counter()-t0:.1f}s: {meta.get('general.name')} "
            f"arch={meta.get('general.architecture')} n_ctx={self.n_ctx} gpu_layers={self.n_gpu_layers}"
        )
        return self

    # ---- prompt ----

    def render(self, messages: List[Message]) -> str:
        if self._template is None:
            raise RuntimeError("Not loaded. Call load() first.")
        return self._template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def count_tokens(self, text: str) -> int:
        if self.llm is None:
            raise RuntimeError("Not loaded. Call load() first.")
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    # ---- generation ----

    def _params(self, kw: dict) -> dict:
        return {
            "max_tokens": kw.get("max_tokens", self.max_tokens),
            "temperature": kw.get("temperature", self.temperature),
            "top_p": kw.get("top_p", self.top_p),
            "stop": kw.get("stop", self._eos),
            "repeat_penalty": kw.get("repeat_penalty", 1.05),
        }

    def generate(self, messages: List[Message], **kw) -> GenerationResult:
        if self.llm is None:
            raise RuntimeError("Not loaded. Call load() first.")
        prompt = self.render(messages)
        t0 = time.perf_counter()
        out = self.llm.create_completion(prompt=prompt, stream=False, **self._params(kw))
        latency = (time.perf_counter() - t0) * 1000
        choice = out["choices"][0]
        usage = out.get("usage", {})
        text = self._strip_think(choice["text"])
        logger.debug(f"generate: {usage.get('prompt_tokens')}+{usage.get('completion_tokens')} tok in {latency:.0f}ms")
        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=out,
        )

    def stream(self, messages: List[Message], **kw) -> Iterator[str]:
        if self.llm is None:
            raise RuntimeError("Not loaded. Call load() first.")
        prompt = self.render(messages)
        for chunk in self.llm.create_completion(prompt=prompt, stream=True, **self._params(kw)):
            piece = chunk["choices"][0].get("text", "")
            if piece:
                yield piece

    @staticmethod
    def _strip_think(text: str) -> str:
        """Drop a leading <think>...</think> block if the model emitted one anyway."""
        s = text.lstrip()
        if s.startswith("<think>"):
            end = s.find("</think>")
            if end != -1:
                return s[end + len("</think>"):].lstrip()
        return text.strip()

    def unload(self):
        if self.llm is not None:
            try:
                self.llm.close()  # free llama.cpp resources explicitly (avoids __del__ noise at exit)
            except Exception:
                pass
            self.llm = None
            logger.info("LLM unloaded")
