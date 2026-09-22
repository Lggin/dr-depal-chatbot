"""Detect machine-control instructions in generated text (spec §22).

V1 is READ ONLY: the assistant must never instruct the user to start/reset/
restart equipment. We flag such phrases and append a notice; blocking is left
to a future permission layer.
"""

import re
from typing import List

# (label, pattern) — Korean and English imperative forms.
_PATTERNS = [
    ("robot start",        r"(로봇|robot)\s*(을|를)?\s*(기동|시작|start)\s*(하|해|시키|하세요|하십시오|하면)"),
    ("robot reset",        r"(로봇|robot|컨트롤러|controller)\s*(을|를)?\s*(리셋|reset|재설정)\s*(하|해|시키|하세요|하십시오|하면)"),
    ("servo on",           r"(서보|servo)\s*(를|을)?\s*(on|온|켜|활성화)\s*(하|해|시키|하세요|하십시오|하면)?"),
    ("plc reset",          r"(plc)\s*(를|을)?\s*(리셋|reset|재설정|초기화)"),
    ("motor move",         r"(모터|motor|축|axis)\s*(를|을)?\s*(수동으로\s*)?(이동|move|구동)\s*(시키|하세요|하십시오|해)"),
    ("e-stop release",     r"(비상\s*정지|e-?stop|emergency\s*stop)\s*(를|을)?\s*(해제|release|풀)"),
    ("restart",            r"(장비|설비|시스템|machine|system|라인)\s*(을|를)?\s*(재시작|재기동|restart|reboot)\s*(하|해|시키|하세요|하십시오|하면)"),
    ("auto restart",       r"(자동\s*재시작|auto(matic)?\s*restart)"),
]
_COMPILED = [(label, re.compile(p, re.IGNORECASE)) for label, p in _PATTERNS]

# Negated mentions ("리셋하지 마세요") are fine — strip them before scanning.
_NEGATION = re.compile(r"[^.。\n]*(하지\s*마|하지\s*않|금지|do not|don't|never)[^.。\n]*", re.IGNORECASE)

DANGER_NOTICE = (
    "⚠️ 이 시스템은 장비 제어 명령을 실행하거나 지시하지 않습니다. "
    "위 내용 중 장비 조작이 포함되어 있다면 반드시 담당 엔지니어 확인 후 절차에 따라 수행하십시오."
)


def scan_dangerous_commands(text: str) -> List[str]:
    cleaned = _NEGATION.sub(" ", text)
    return [label for label, rx in _COMPILED if rx.search(cleaned)]
