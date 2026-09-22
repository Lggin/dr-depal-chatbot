import pytest

from app.router.intent_router import IntentRouter
from app.schemas.chat import IntentType as I

CASES = [
    # spec §9 examples
    ("VAC-203이 뭐야?", I.KNOWLEDGE),
    ("진공압 정상 범위 알려줘", I.KNOWLEDGE),
    ("지금 진공압 얼마야?", I.LIVE_STATUS),
    ("지금 왜 장비 멈췄어?", I.LIVE_DIAGNOSIS),
    ("박스 못 집는데 어떻게 확인해?", I.TROUBLESHOOTING),
    ("지난번에도 VAC-203 발생했어?", I.HISTORY),
    # spec §17
    ("지금 진공압이 정상인지 확인해줘.", I.LIVE_DIAGNOSIS),
    ("진공압 정상 범위가 얼마야?", I.KNOWLEDGE),
    # spec §18
    ("지금 박스를 왜 못 집어?", I.LIVE_DIAGNOSIS),
    # sample dataset expectations
    ("VIS-301 알람이 떴어. 뭘 확인해?", I.TROUBLESHOOTING),
    ("박스 흡착 실패 원인이 뭐야?", I.TROUBLESHOOTING),
    ("빨판이 박스를 못 잡아", I.TROUBLESHOOTING),
    ("카메라가 응답이 없어", I.TROUBLESHOOTING),
    ("로봇이 자꾸 멈추고 서보 과부하래", I.TROUBLESHOOTING),
    ("박스가 있는데 비전에서 검출이 안 돼", I.TROUBLESHOOTING),
    ("로봇이 박스 위치에서 항상 조금씩 어긋나", I.TROUBLESHOOTING),
    ("진공압 표시가 0으로 고정돼 있어", I.TROUBLESHOOTING),
    ("화면이 먹통이야", I.TROUBLESHOOTING),
    # out-of-scope domain questions stay KNOWLEDGE — retrieval decides they can't be answered (spec §30-5)
    ("로봇 컨트롤러 IP 주소가 뭐야?", I.KNOWLEDGE),
    ("설비 담당자 연락처 알려줘", I.KNOWLEDGE),
    ("VAC-203 지난번에도 있었어?", I.HISTORY),
    ("9월 10일에 VAC-203 어떻게 조치했어?", I.HISTORY),
    ("ROB-105 전에도 발생했어?", I.HISTORY),
    ("VAC-203 알람이 또 났어?", I.HISTORY),
    # past tense about a documented measurement is knowledge, not history
    ("캐시 판정용 중심점이 옆면 쪽으로 얼마나 다르게 밀려나 있었어?", I.KNOWLEDGE),
    ("v0.0.51 릴리즈는 실제로 뭘 바꿨어?", I.KNOWLEDGE),
    ("VAC-203이 며칠에 걸쳐 서서히 나빠지면 어디부터 봐?", I.TROUBLESHOOTING),
    ("캘리브레이션 방법 알려줘", I.KNOWLEDGE),
    ("로봇 payload 한계가 얼마야?", I.KNOWLEDGE),
    ("detection_threshold 기본값이 뭐야?", I.KNOWLEDGE),
    ("처리할 수 있는 박스 최대 크기가 얼마야?", I.KNOWLEDGE),
    ("해당 알람 점검 순서 알려줘", I.TROUBLESHOOTING),
    ("그 알람 원인이 뭐야?", I.KNOWLEDGE),
    # live vs reference disambiguation
    ("현재 알람 뭐 떠 있어?", I.LIVE_STATUS),
    ("로봇 상태 지금 어때?", I.LIVE_STATUS),
    ("진공압은 보통 얼마 이하면 알람이야?", I.KNOWLEDGE),
    # small talk
    ("안녕", I.GENERAL),
    ("고마워!", I.GENERAL),
    ("넌 누구야?", I.GENERAL),
    # knowhow-style developer questions default to knowledge
    ("v0.0.51 릴리즈는 실제로 뭘 바꿨어?", I.KNOWLEDGE),
    ("RF-DETR 학습 때 쓴 증강 파이프라인의 순서는 어떻게 돼?", I.KNOWLEDGE),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_intent_rules(text, expected):
    r = IntentRouter().classify(text)
    assert r.intent == expected, f"{text!r} → {r.intent} (matched {r.matched})"


def test_unmatched_text_defaults_to_knowledge_with_low_confidence():
    r = IntentRouter().classify("dr-depal-sdk 워크스페이스 패키지")
    assert r.intent == I.KNOWLEDGE and r.confidence <= 0.5 and r.matched == ["default"]
