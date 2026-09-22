"""Rule-based intent classification (spec §9). Deterministic; no LLM call.

Precedence when several rule groups fire:
  LIVE_DIAGNOSIS > LIVE_STATUS > HISTORY > TROUBLESHOOTING > KNOWLEDGE.
GENERAL is only for explicit small talk. Anything unmatched falls back to
KNOWLEDGE — the retrieval stack, not the classifier, decides whether the
corpus can answer.
"""

import re
from dataclasses import dataclass, field
from typing import List

from app.schemas.chat import IntentType

# "now" markers: the user is asking about the machine's current state
_NOW = re.compile(r"(지금|현재|방금|오늘|실시간|요즘|이 순간)")
# reference/spec vocabulary: asking what the value *should* be, not what it is
_REFERENCE = re.compile(r"(정상\s*(범위|값|기준|인지|이야|이냐|인가|이니|이에요|입니까)|기준\s*값|허용\s*(범위|치)|스펙|사양|규격|기본\s*값|한계|최대|최소|이하|이상이어야)")
_VALUE_QUERY = re.compile(r"(얼마(야|예요|입니까|나|지)?|몇\s*(도|개|kpa|mm|%|퍼센트)?|값(은|이)?\s*(뭐|어떻|얼마)|상태(는|가)?\s*(어때|뭐|어떻)|켜져|꺼져|돌아가|가동\s*중|멈춰\s*있)")
_WHY_NOW = re.compile(r"(왜|원인이?\s*뭐|이유가?\s*뭐).*(멈|안\s*(돼|되|잡|집|나와|보여|움직)|못\s*(집|잡)|실패|떨어|안\s*떠|튀|느려)|(멈췄|안\s*돼|안\s*되는데|못\s*(집|잡)|실패).*(왜|원인)")
# Past tense alone ("…밀려나 있었어?") is not a history question — a documented measurement can be
# phrased in the past. History needs an occurrence/recurrence or maintenance-action marker.
_HISTORY = re.compile(r"(지난\s*번|이전에|전에도|과거에?|예전|이력|기록|히스토리|언제\s*(발생|났|있었)|몇\s*번\s*(발생|났|있었)|(또|다시|재)\s*(발생|났|떴)|(알람|경고|문제|고장|증상)(이|가|은|는)?\s*(있었|발생했|났었|떴었)|조치했|고쳤|수리했|교체했|어떻게\s*(조치|처리|대응)했)")
_TROUBLE = re.compile(r"(어떻게\s*(확인|점검|해결|고치|조치)|뭘\s*(확인|봐|점검|해야)|무엇을\s*(확인|점검)|어디(부터|를)\s*(봐|확인|점검)|점검\s*(순서|방법|항목)|해결\s*방법|조치\s*방법|트러블|troubleshoot|(못|안)\s*(집|잡|돼|되|나와|보여|인식|검출|움직|올라|내려)|실패|고장|이상해|안\s*떠|튀어|떨어져|응답이?\s*없|멈춰|멈춤|멈추|에러가?\s*(나|떠|계속)|과부하|과열|알람이?\s*(뜨|떠|났|계속)|어긋나|틀어져|벗어나|오차가?\s*(크|나|생)|편차|흔들|삐뚤|고정돼|고정되|안\s*바뀌|바뀌지\s*않|0\s*으로\s*(고정|표시|나와)|표시가?\s*(안|이상|틀|고정)|먹통|깜빡|끊겨|느려)")
_KNOWLEDGE = re.compile(r"(뭐야|무엇|뭔지|뜻|의미|설명|알려\s*줘|방법|절차|어떻게\s*(해|하는|사용|쓰)|가이드|매뉴얼|정의|차이|비교|왜\s*(필요|중요|그래|그런지))")
# no \b: Hangul syllables are all \w, so a word boundary never falls between "누구" and "야"
_SMALLTALK = re.compile(r"^\s*(안녕|하이|헬로|hello|hi|고마워|감사|잘\s*있어|수고|넌\s*누구|너는\s*누구|뭐\s*할\s*수\s*있어|도움말|help)")

# Live-state nouns — a "now" marker plus one of these is a status query even without a value verb.
_MACHINE_NOUN = re.compile(r"(진공압|압력|알람|상태|온도|속도|위치|포즈|좌표|검출|카메라|로봇|plc|센서|모터|서보|장비|설비|라인)")


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    matched: List[str] = field(default_factory=list)


class IntentRouter:
    def classify(self, text: str) -> IntentResult:
        t = text.strip()
        low = t.lower()
        matched: List[str] = []

        if _SMALLTALK.search(low) and len(t) < 30 and not _MACHINE_NOUN.search(low):
            return IntentResult(IntentType.GENERAL, 0.9, ["smalltalk"])

        now = bool(_NOW.search(low))
        reference = bool(_REFERENCE.search(low))
        why_now = bool(_WHY_NOW.search(low))
        history = bool(_HISTORY.search(low))
        trouble = bool(_TROUBLE.search(low))
        value_q = bool(_VALUE_QUERY.search(low))
        machine = bool(_MACHINE_NOUN.search(low))
        knowledge = bool(_KNOWLEDGE.search(low))

        # "지금 왜 멈췄어" / "지금 박스를 왜 못 집어" → diagnose current state
        if now and (why_now or (trouble and machine)):
            return IntentResult(IntentType.LIVE_DIAGNOSIS, 0.85, ["now", "why/trouble"])
        # "지금 진공압이 정상인지 확인해줘" → current value vs reference
        if now and reference and machine:
            return IntentResult(IntentType.LIVE_DIAGNOSIS, 0.8, ["now", "reference", "machine"])
        # "지금 진공압 얼마야" / "현재 알람 뭐야" → live value
        if now and machine and (value_q or knowledge or not reference):
            return IntentResult(IntentType.LIVE_STATUS, 0.85, ["now", "machine"])
        if history and not reference:
            return IntentResult(IntentType.HISTORY, 0.85, ["history"])
        if why_now and not reference:
            return IntentResult(IntentType.TROUBLESHOOTING, 0.8, ["why"])
        if trouble and not reference:
            return IntentResult(IntentType.TROUBLESHOOTING, 0.75, ["trouble"])
        if knowledge or reference:
            matched = [k for k, v in (("knowledge", knowledge), ("reference", reference)) if v]
            return IntentResult(IntentType.KNOWLEDGE, 0.8, matched)
        return IntentResult(IntentType.KNOWLEDGE, 0.5, ["default"])
