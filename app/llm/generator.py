"""ChatPipeline: question -> retrieval -> prompt -> LLM -> validation -> ChatResponse."""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.llm.client import LLMClient
from app.llm.prompt import PromptBuilder
from app.memory.session import SessionStore
from app.rag.retriever import RetrievalPipeline
from app.router.alarm_parser import extract_alarm_codes
from app.safety.answer_validator import AnswerValidator
from app.safety.command_guard import DANGER_NOTICE, scan_dangerous_commands
from app.safety.source_validator import build_sources, canonicalize_citations
from app.schemas.chat import ChatResponse, IntentType, RetrievalInfo, ValidationInfo

logger = logging.getLogger(__name__)

Message = Dict[str, str]

# Korean/English anaphora that point back at the previous turn.
_ANAPHORA = re.compile(r"(그|이|저|해당|아까|방금|위|앞서)\s*(알람|경고|코드|원인|문제|거|것|건|증상|장비)|\bit\b|\bthat\b", re.IGNORECASE)


def contextualize(question: str, history: List[Message]) -> Tuple[str, str, List[str]]:
    """Carry alarm codes from the previous turn into an anaphoric follow-up.

    Returns (retrieval_query, prompt_question, carried_codes). Deterministic — no LLM.
    """
    if not history or extract_alarm_codes(question) or not _ANAPHORA.search(question):
        return question, question, []
    prev_codes = extract_alarm_codes(" ".join(m.get("content", "") for m in history))
    if not prev_codes:
        return question, question, []
    codes = " ".join(prev_codes)
    return f"{question} {codes}", f"{question}\n(이전 대화에서 언급된 알람: {', '.join(prev_codes)})", prev_codes


class ChatPipeline:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        prompt_builder: PromptBuilder,
        llm: LLMClient,
        sessions: SessionStore,
        validator: Optional[AnswerValidator] = None,
    ):
        self.retrieval = retrieval
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.sessions = sessions
        self.validator = validator or AnswerValidator()

    def answer(
        self,
        session_id: str,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        machine_state: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        t_start = time.perf_counter()
        timing: Dict[str, float] = {}

        session = self.sessions.get_or_create(session_id)
        retrieval_query, prompt_question, carried = contextualize(question, session.history)

        t = time.perf_counter()
        res = self.retrieval.search(retrieval_query, filters=filters)
        timing["retrieval"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        messages, built = self.prompt_builder.build(
            prompt_question, res.docs, history=session.history, machine_state=machine_state
        )
        gen = self.llm.generate(messages)
        timing["generation"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        validation = self.validator.validate(gen.text, built.included, prompt_question, extra_evidence=_ms_text(machine_state))
        sources = [] if validation.is_refusal else build_sources(built.included, validation.cited_documents)
        text = canonicalize_citations(gen.text, sources, validation.is_refusal)
        danger = scan_dangerous_commands(text)
        if danger:
            text = f"{text}\n\n{DANGER_NOTICE}"
        timing["validation"] = (time.perf_counter() - t) * 1000

        self.sessions.record(session_id, question, text)

        total = (time.perf_counter() - t_start) * 1000
        alarm_codes = list(dict.fromkeys(res.alarm_codes + carried))
        logger.info(
            f"chat[{session_id}] {total:.0f}ms retrieval={timing['retrieval']:.0f} gen={timing['generation']:.0f} "
            f"docs={len(built.included)} codes={alarm_codes}{' (carried)' if carried else ''} "
            f"ok={validation.ok} refusal={validation.is_refusal} danger={danger}"
        )
        return ChatResponse(
            session_id=session_id,
            message=question,
            answer=text,
            intent=IntentType.KNOWLEDGE,  # intent router arrives in Phase 3
            sources=sources,
            tools_used=[],
            retrieval=RetrievalInfo(
                documents_found=len(res.docs),
                reranked=self.retrieval.reranker is not None,
                top_k=len(built.included),
                sources=[build_sources([d])[0] for d in built.included],
            ),
            validation=ValidationInfo(**validation.to_dict(), danger_flags=danger),
            alarm_codes=alarm_codes,
            timing_ms={k: round(v, 1) for k, v in timing.items()},
            latency_ms=round(total, 1),
        )

    def close(self) -> None:
        unload = getattr(self.llm, "unload", None)
        if unload:
            unload()


def _ms_text(machine_state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not machine_state:
        return None
    return "\n".join(f"{k}: {v}" for k, v in machine_state.items())
