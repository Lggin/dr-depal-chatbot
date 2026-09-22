from app.safety.answer_validator import AnswerValidator, extract_numbers
from app.safety.command_guard import scan_dangerous_commands
from app.safety.source_validator import build_sources, canonicalize_citations, format_citation
from app.schemas.retrieval import RetrievedDoc


def doc(cid, text, **meta):
    return RetrievedDoc(chunk_id=cid, text=text, score=1.0, rank=1, source="rerank", metadata=meta)


MANUAL = doc("VAC_203_001", "[Alarm] VAC-203 [Normal Range] -60 ~ -80 kPa. 공급 공기압 0.5 MPa",
             document_name="Vacuum System Manual", section="6.3", alarm_code="VAC-203", document_type="alarm_manual", trust_level=100)
LOG = doc("MAINT_1", "2026-09-10 VAC-203 발생. 진공압 -45 kPa 정체. 3번 suction cup 호스 교체",
          document_name="Maintenance Log", document_type="maintenance", trust_level=70)


# ---------- numbers ----------

def test_extract_numbers_normalizes_sign_and_leading_zero():
    nums = extract_numbers("-60 ~ -80 kPa, 2026-09-10, 0.5 MPa, 진공압 -09 kPa")
    assert nums >= {"-60", "-80", "2026", "-9", "10", "0.5"}
    assert ".5" not in nums


def test_section_numbers_in_header_or_citation_are_not_flagged():
    v = AnswerValidator().validate("§6.3 에 따르면 정상 범위는 -60 ~ -80 kPa 입니다.\n출처: Vacuum System Manual §6.3",
                                  [MANUAL], "q")
    assert v.unsupported_numbers == [] and v.ok


def test_validator_passes_grounded_answer():
    v = AnswerValidator().validate(
        "정상 범위는 -60 ~ -80 kPa 입니다. 2026 년 9 월 10 일 -45 kPa 정체.\n출처: Vacuum System Manual §6.3",
        [MANUAL, LOG], "VAC-203 정상 범위?",
    )
    assert v.ok and not v.unsupported_numbers and not v.unsupported_codes and not v.invalid_citations
    assert v.cited_documents == ["Vacuum System Manual"]


def test_validator_flags_number_not_in_evidence():
    v = AnswerValidator().validate("정상 범위는 -55 ~ -80 kPa 입니다.", [MANUAL], "VAC-203?")
    assert v.unsupported_numbers == ["-55"] and not v.ok


def test_validator_ignores_small_list_markers_and_question_numbers():
    v = AnswerValidator().validate("1. 호스 2. 패드 3. 발생기. 요청하신 12번 라인은 문서에 없습니다.",
                                  [MANUAL], "12번 라인 알람?")
    assert v.unsupported_numbers == []


def test_validator_flags_alarm_code_not_in_evidence():
    v = AnswerValidator().validate("VAC-203과 유사한 ROB-999도 확인하세요.", [MANUAL], "VAC-203?")
    assert v.unsupported_codes == ["ROB-999"]


def test_validator_flags_citation_of_unprovided_document():
    v = AnswerValidator().validate("답변.\n출처: Robot Manual §1.2; Vacuum System Manual §6.3", [MANUAL], "q")
    assert v.invalid_citations == ["Robot Manual"] and not v.ok


def test_validator_refusal_skips_number_check():
    v = AnswerValidator().validate("현재 제공된 문서에서는 IP 주소 192.168.0.10 을 확인할 수 없습니다.", [MANUAL], "IP?")
    assert v.is_refusal and v.unsupported_numbers == [] and v.ok


def test_validator_uses_machine_state_as_evidence():
    v = AnswerValidator().validate("현재 진공압 -43 kPa 로 부족합니다.", [MANUAL], "지금 진공압?",
                                  extra_evidence="vacuum_pressure: -43")
    assert v.ok


# ---------- command guard ----------

def test_guard_flags_control_instructions():
    assert "servo on" in scan_dangerous_commands("문제가 없으면 서보를 on 하세요.")
    assert "plc reset" in scan_dangerous_commands("PLC를 리셋 후 재확인")
    assert "restart" in scan_dangerous_commands("장비를 재시작하면 해결됩니다.")
    assert "e-stop release" in scan_dangerous_commands("비상정지를 해제하고")


def test_guard_ignores_negated_and_descriptive_mentions():
    assert scan_dangerous_commands("임의로 PLC를 리셋하지 마세요. 담당자에게 연락하세요.") == []
    assert scan_dangerous_commands("VAC-203은 진공압 저하 알람입니다.") == []


# ---------- sources ----------

def test_build_sources_narrows_to_cited_when_resolvable():
    srcs = build_sources([MANUAL, LOG], cited_names=["Maintenance Log"])
    assert [s.chunk_id for s in srcs] == ["MAINT_1"]
    assert build_sources([MANUAL, LOG], cited_names=["Nope"]) and len(build_sources([MANUAL, LOG], cited_names=["Nope"])) == 2


def test_format_citation_uses_metadata_only():
    line = format_citation(build_sources([MANUAL, LOG]))
    assert line == "출처: Vacuum System Manual §6.3 (VAC-203); Maintenance Log"


def test_canonicalize_replaces_model_citation_line():
    answer = "네, 이력이 있습니다.\n\n출처: Maintenance Log §2026-09-10 DP-01 VAC-203 발생"
    out = canonicalize_citations(answer, build_sources([LOG]), is_refusal=False)
    assert out == "네, 이력이 있습니다.\n\n출처: Maintenance Log"


def test_canonicalize_removes_citation_on_refusal():
    out = canonicalize_citations("현재 제공된 문서에서는 확인할 수 없습니다.\n출처: X", build_sources([MANUAL]), is_refusal=True)
    assert "출처" not in out
