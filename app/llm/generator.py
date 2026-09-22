"""ChatPipeline: question -> intent route -> (troubleshooting | retrieval -> prompt -> LLM) -> validation -> ChatResponse.

Routes (spec §9, §17, §19-§21):
  KNOWLEDGE        retrieval → LLM (Phase 1 path)
  HISTORY          retrieval filtered to maintenance records first, unfiltered fallback
  LIVE_STATUS      needs a machine tool; without one, an honest deterministic notice (no LLM)
  LIVE_DIAGNOSIS   reference values + inspection order from documents, with a live-value notice
  TROUBLESHOOTING  decision-tree engine: ask one fact at a time (deterministic), recommend a step (RAG + LLM)
  GENERAL          deterministic help text
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.llm.client import LLMClient
from app.llm.prompt import PromptBuilder
from app.memory.session import Session, SessionStore
from app.rag.retriever import RetrievalPipeline
from app.router.alarm_parser import extract_alarm_codes
from app.router.intent_router import IntentRouter
from app.safety.answer_validator import AnswerValidator, ValidationResult
from app.safety.command_guard import DANGER_NOTICE, scan_dangerous_commands
from app.safety.source_validator import build_sources, canonicalize_citations
from app.schemas.chat import ChatResponse, IntentType, RetrievalInfo, ValidationInfo

logger = logging.getLogger(__name__)

Message = Dict[str, str]

_ANAPHORA = re.compile(r"(그|이|저|해당|아까|방금|위|앞서)\s*(알람|경고|코드|원인|문제|거|것|건|증상|장비)|\bit\b|\bthat\b", re.IGNORECASE)
# "…원인이 뭐야?" asks for an explanation → answer from documents; a bare symptom report starts the tree.
_ASKS_EXPLANATION = re.compile(r"(원인|이유|왜|뭐야|뭔지|무엇|설명|알려\s*줘|의미|뜻)")

LIVE_UNAVAILABLE = ("현재 설비 값(알람·진공압·로봇 상태 등)은 설비 조회 도구가 아직 연결되어 있지 않아 확인할 수 없습니다. "
                    "정상 범위나 점검 절차가 필요하시면 그렇게 질문해 주세요.")
GENERAL_HELP = ("디팔레타이징 시스템 기술 지원 도우미입니다. 알람 코드 설명, 정상 값·설정 범위, 캘리브레이션·라벨링 등 절차, "
                "증상별 점검 순서, 과거 유지보수 이력을 문서에 근거해 답합니다. 예: \"VAC-203이 뭐야?\", \"박스를 못 집는데 뭘 확인해?\"")


def contextualize(question: str, history: List[Message]) -> Tuple[str, str, List[str]]:
    """Carry alarm codes from the previous turn into an anaphoric follow-up. Deterministic."""
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
        router: Optional[IntentRouter] = None,
        engine=None,   # TroubleshootingEngine; optional so retrieval-only tests can skip it
        tools=None,    # ToolRegistry (READ ONLY machine tools); None = no live data
    ):
        self.retrieval = retrieval
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.sessions = sessions
        self.validator = validator or AnswerValidator()
        self.router = router or IntentRouter()
        self.engine = engine
        self.tools = tools

    # ------------------------------------------------------------------ tools

    def _live_state(self, machine_state: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Caller-supplied state wins; otherwise snapshot the machine through the registry."""
        if machine_state is not None or self.tools is None:
            return machine_state, []
        return self.tools.snapshot()

    def _maintenance(self, codes: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
        if self.tools is None or self.tools.get("get_maintenance_history") is None:
            return None, []
        rows: List[Dict[str, Any]] = []
        for code in codes or [None]:
            r = self.tools.call("get_maintenance_history", alarm_code=code) if code else self.tools.call("get_maintenance_history")
            if r.success and isinstance(r.data, list):
                rows.extend(x for x in r.data if x not in rows)
        return rows, ["get_maintenance_history"]

    # ------------------------------------------------------------------ public

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
        intent = self.router.classify(question).intent
        tools_used: List[str] = []

        # --- troubleshooting: sticky follow-up reply, or a new issue ---
        ts = session.troubleshooting
        if self.engine and ts is not None and not ts.resolved and self.engine.is_reply(question):
            outcome = self.engine.apply_reply(ts, question)
            machine_state, tools_used = self._live_state(machine_state)
            return self._troubleshoot(session, question, ts, outcome, machine_state, timing, t_start, tools_used)
        if self.engine and intent == IntentType.TROUBLESHOOTING and not _ASKS_EXPLANATION.search(question):
            issue = self.engine.match_issue(question)
            if issue:
                ts = self.engine.start(issue)
                machine_state, tools_used = self._live_state(machine_state)
                filled = self.engine.prefill_from_tools(ts, machine_state)
                session.troubleshooting = ts
                return self._troubleshoot(session, question, ts, "started", machine_state, timing, t_start, tools_used, filled)
        if ts is not None and not self.engine.is_reply(question) if self.engine and ts is not None else False:
            session.troubleshooting = None   # user moved on to a new topic

        # --- deterministic routes ---
        if intent == IntentType.GENERAL:
            return self._plain(session, question, GENERAL_HELP, intent, "general", t_start)
        if intent in (IntentType.LIVE_STATUS, IntentType.LIVE_DIAGNOSIS):
            machine_state, tools_used = self._live_state(machine_state)
        if intent == IntentType.LIVE_STATUS and machine_state is None:
            return self._plain(session, question, LIVE_UNAVAILABLE, intent, "live.no_tools", t_start)

        # --- retrieval routes ---
        retrieval_query, prompt_question, carried = contextualize(question, session.history)
        route = intent.value
        effective_filters = dict(filters or {})
        maintenance_history = None
        if intent == IntentType.HISTORY:
            effective_filters.setdefault("document_type", "maintenance")
            prompt_question += "\n(이력 질문: 과거 유지보수 기록을 기준으로 답하고, 공식 매뉴얼 내용과 구분해 주세요.)"
            maintenance_history, used = self._maintenance(list(dict.fromkeys(extract_alarm_codes(question) + carried)))
            tools_used += used
        if intent == IntentType.LIVE_DIAGNOSIS and machine_state is None:
            prompt_question += "\n(현재 설비 값은 조회할 수 없습니다. 문서의 기준값과 점검 순서만 제시하고, 현재 값은 확인되지 않았다고 명시해 주세요.)"
        elif intent == IntentType.LIVE_DIAGNOSIS:
            prompt_question += ("\n([MACHINE STATE]의 현재 값을 문서의 기준값과 비교해 관측 사실과 가능한 원인을 구분하고, "
                                "정상으로 확인된 계통은 제외한 뒤 우선 점검할 계통과 순서를 제시해 주세요.)")
            route = "live_diagnosis.tools" if tools_used else "live_diagnosis.state"

        t = time.perf_counter()
        res = self.retrieval.search(retrieval_query, filters=effective_filters or None)
        if intent == IntentType.HISTORY and not res.docs and effective_filters:
            res = self.retrieval.search(retrieval_query, filters=filters or None)   # no maintenance records → fall back
            route = "history.unfiltered"
            prompt_question = prompt_question.replace(
                "(이력 질문: 과거 유지보수 기록을 기준으로 답하고, 공식 매뉴얼 내용과 구분해 주세요.)",
                "(이력 질문이지만 유지보수 기록은 검색되지 않았습니다. 문서에 과거 사례가 있으면 그것을 답하고, "
                "없으면 '기록이 없습니다'라고 한 문장으로만 밝혀 주세요. 빈 '과거 유지보수 이력' 절을 만들지 마세요.)")
        elif intent == IntentType.HISTORY:
            route = "history.filtered"
        timing["retrieval"] = (time.perf_counter() - t) * 1000

        text, built, validation, danger, timing = self._generate(prompt_question, res.docs, session, machine_state, None, timing,
                                                                 maintenance_history=maintenance_history)
        if intent == IntentType.LIVE_DIAGNOSIS and machine_state is None:
            text = f"⚠️ 현재 설비 값은 조회 도구 미연결로 확인할 수 없습니다. 아래는 문서 기준값과 점검 순서입니다.\n\n{text}"
            route = "live_diagnosis.no_tools"
        if intent == IntentType.LIVE_STATUS:
            route = "live_status.tools" if tools_used else "live_status.state"   # .state = caller-supplied machine_state

        self.sessions.record(session_id, question, text)
        return self._response(session, question, text, intent, route, res, built, validation, danger,
                              alarm_codes=list(dict.fromkeys(res.alarm_codes + carried)),
                              retrieval_query=retrieval_query if retrieval_query != question else None,
                              timing=timing, t_start=t_start, tools_used=tools_used, machine_state=machine_state)

    def reindex(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild the corpus from configs/corpus.yaml into the live collection and refresh BM25.
        Caller must hold the generate lock: the collection is dropped and recreated."""
        from pathlib import Path
        from app.config import settings
        from app.rag.sparse_retriever import SparseRetriever
        from ingestion.pipeline import PROJECT_ROOT, IngestionPipeline

        hybrid = self.retrieval.hybrid
        embedder, store = hybrid.dense.embedder, hybrid.store
        pipe = IngestionPipeline(Path(config_path or PROJECT_ROOT / settings.corpus_config), embedder.count_tokens)
        pipe.cfg["collection"] = store.collection_name
        texts, metas, ids, manifest = pipe.build()
        pipe.write(texts, metas, ids, manifest)
        IngestionPipeline.index(texts, metas, ids, embedder, store)
        hybrid.sparse = SparseRetriever.from_qdrant(store, top_k=hybrid.sparse_top_k)
        manifest["bm25_chunks"] = hybrid.sparse.size
        logger.info(f"reindex complete: {manifest['chunks']} chunks into {store.collection_name}")
        return manifest

    def close(self) -> None:
        unload = getattr(self.llm, "unload", None)
        if unload:
            unload()

    # ------------------------------------------------------------- troubleshooting

    def _troubleshoot(self, session: Session, question: str, ts, outcome: str, machine_state, timing, t_start,
                      tools_used: Optional[List[str]] = None, prefilled: Optional[List[str]] = None) -> ChatResponse:
        decision = self.engine.next(ts)
        label = decision.label
        snapshot = ts.snapshot()
        tools_used = tools_used or []

        if decision.kind == "ask":
            if outcome == "started":
                text = f"'{label}' 증상을 단계적으로 확인하겠습니다."
                if prefilled:
                    text += "\n" + self._prefill_note(ts, prefilled, machine_state)
                text += f"\n\n{decision.question}"
            elif outcome == "unclear":
                text = f"예/아니오로 답해 주시면 다음 단계로 넘어갈 수 있습니다.\n\n{decision.question}"
            else:
                text = f"확인했습니다.\n\n{decision.question}"
            return self._plain(session, question, text, IntentType.TROUBLESHOOTING, "troubleshooting.ask", t_start, snapshot,
                               tools_used=tools_used, machine_state=machine_state)

        if decision.kind in ("escalate", "resolved"):
            session.troubleshooting = None
            return self._plain(session, question, decision.message, IntentType.TROUBLESHOOTING,
                               f"troubleshooting.{decision.kind}", t_start, snapshot, tools_used=tools_used)

        # recommend: ground the inspection procedure in the documents
        facts_txt = ", ".join(f"{k}={'예' if v else '아니오'}" for k, v in decision.confirmed.items()) or "없음"
        if ts.tool_filled:
            facts_txt += f" (설비 조회로 자동 확인: {', '.join(ts.tool_filled)})"
        prompt_question = (
            f"[{label}] 진단 중입니다. 확인된 사항: {facts_txt}.\n"
            f"다음 점검 단계 '{decision.step}'의 권장 조치는 {', '.join(decision.actions)} 입니다.\n"
            f"문서에 근거해 이 조치들을 점검 순서대로, 각 항목에서 무엇을 어떻게 확인하는지 설명해 주세요. "
            f"이미 확인된 정상 항목은 다시 점검하라고 하지 마세요."
        )
        ts_state = {"issue": label, "confirmed": decision.confirmed, "unknown": decision.unknown, "current_step": decision.step}

        t = time.perf_counter()
        res = self.retrieval.search(decision.doc_query)
        timing["retrieval"] = (time.perf_counter() - t) * 1000
        text, built, validation, danger, timing = self._generate(prompt_question, res.docs, session, machine_state, ts_state, timing)
        text += "\n\n점검 후 결과를 알려 주세요. 해결되면 \"해결됐어\"라고 답해 주시면 됩니다."
        self.sessions.record(session.session_id, question, text)
        return self._response(session, question, text, IntentType.TROUBLESHOOTING, "troubleshooting.recommend",
                              res, built, validation, danger, alarm_codes=res.alarm_codes,
                              retrieval_query=decision.doc_query, timing=timing, t_start=t_start, troubleshooting=snapshot,
                              tools_used=tools_used, machine_state=machine_state)

    def _prefill_note(self, ts, prefilled: List[str], machine_state: Optional[Dict[str, Any]]) -> str:
        issue = self.engine.rules.issues[ts.issue]
        bits = []
        for name in prefilled:
            fact = issue.facts[name]
            raw = machine_state.get(fact.tool) if machine_state and fact.tool else None
            verdict = "정상" if ts.facts.get(name) else "이상"
            bits.append(f"{name}={verdict}" + (f" ({fact.tool}={raw})" if raw is not None else ""))
        return "설비 조회로 자동 확인된 항목: " + ", ".join(bits) + " — 해당 질문은 건너뜁니다."

    # ------------------------------------------------------------------ helpers

    def _generate(self, prompt_question, docs, session, machine_state, ts_state, timing, maintenance_history=None):
        t = time.perf_counter()
        messages, built = self.prompt_builder.build(prompt_question, docs, history=session.history,
                                                    machine_state=machine_state, troubleshooting_state=ts_state,
                                                    maintenance_history=maintenance_history)
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
        built.sources = sources  # type: ignore[attr-defined]
        return text, built, validation, danger, timing

    def _plain(self, session: Session, question: str, text: str, intent: IntentType, route: str, t_start: float,
               troubleshooting: Optional[dict] = None, tools_used: Optional[List[str]] = None,
               machine_state: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Deterministic answer — no retrieval, no LLM."""
        self.sessions.record(session.session_id, question, text)
        total = (time.perf_counter() - t_start) * 1000
        logger.info(f"chat[{session.session_id}] {total:.0f}ms route={route} tools={tools_used or []}")
        return ChatResponse(
            session_id=session.session_id, message=question, answer=text, intent=intent, sources=[], tools_used=tools_used or [],
            retrieval=RetrievalInfo(documents_found=0, reranked=False, top_k=0, sources=[]),
            # a "no live tools" notice is a refusal to provide the value, and is scored as one
            validation=ValidationInfo(ok=True, is_refusal=(route == "live.no_tools")), alarm_codes=extract_alarm_codes(question),
            route=route, troubleshooting=troubleshooting, machine_state=machine_state, timing_ms={}, latency_ms=round(total, 1),
        )

    def _response(self, session, question, text, intent, route, res, built, validation: ValidationResult, danger,
                  alarm_codes, retrieval_query, timing, t_start, troubleshooting=None, tools_used=None,
                  machine_state=None) -> ChatResponse:
        total = (time.perf_counter() - t_start) * 1000
        sources = getattr(built, "sources", [])
        logger.info(
            f"chat[{session.session_id}] {total:.0f}ms route={route} retrieval={timing.get('retrieval', 0):.0f} "
            f"gen={timing.get('generation', 0):.0f} docs={len(built.included)} codes={alarm_codes} tools={tools_used or []} "
            f"ok={validation.ok} refusal={validation.is_refusal} danger={danger}"
        )
        return ChatResponse(
            session_id=session.session_id, message=question, answer=text, intent=intent, sources=sources, tools_used=tools_used or [],
            machine_state=machine_state,
            retrieval=RetrievalInfo(
                documents_found=len(res.docs), reranked=self.retrieval.reranker is not None, top_k=len(built.included),
                sources=[build_sources([d])[0] for d in built.included], scores=[round(d.score, 4) for d in built.included],
            ),
            validation=ValidationInfo(**validation.to_dict(), danger_flags=danger),
            alarm_codes=alarm_codes, retrieval_query=retrieval_query, route=route, troubleshooting=troubleshooting,
            timing_ms={k: round(v, 1) for k, v in timing.items()}, latency_ms=round(total, 1),
        )


def _ms_text(machine_state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not machine_state:
        return None
    return "\n".join(f"{k}: {v}" for k, v in machine_state.items())
