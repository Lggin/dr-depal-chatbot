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


def test_citation_names_with_commas_dashes_and_parentheses_are_not_fragmented():
    tl = doc("t", "8월 한 달 릴리즈 12회.", document_name="02. 타임라인 — 2026년 3월 ~ 8월",
             section="8월 — 정밀도 추격 (406 커밋, v0.0.46 → v0.0.57)", document_type="manual")
    ans = ("8월에 12회 릴리즈.\n출처: 02. 타임라인 — 2026년 3월 ~ 8월 §8월 — 정밀도 추격 (406 커밋, v0.0.46 → v0.0.57); "
           "[SOURCE 2] 02. 타임라인 — 2026년 3월 ~ 8월 §활동량 요약 (커밋 수)")
    v = AnswerValidator().validate(ans, [tl], "8월 릴리즈 몇 번?")
    assert v.cited_documents == ["02. 타임라인 — 2026년 3월 ~ 8월"] * 2
    assert v.invalid_citations == [] and v.ok


def test_machine_state_in_citation_line_is_not_an_invalid_citation():
    v = AnswerValidator().validate("현재 진공압은 -43 kPa 입니다.\n출처: Vacuum System Manual §6.3; [MACHINE STATE]",
                                  [MANUAL], "지금 진공압?", extra_evidence="vacuum_pressure: -43")
    assert v.invalid_citations == [] and v.cited_documents == ["Vacuum System Manual"] and v.ok


def test_validator_flags_citation_of_unprovided_document():
    v = AnswerValidator().validate("답변.\n출처: Robot Manual §1.2; Vacuum System Manual §6.3", [MANUAL], "q")
    assert v.invalid_citations == ["Robot Manual"] and not v.ok


def test_validator_refusal_skips_number_check():
    v = AnswerValidator().validate("현재 제공된 문서에서는 IP 주소 192.168.0.10 을 확인할 수 없습니다.", [MANUAL], "IP?")
    assert v.is_refusal and v.unsupported_numbers == [] and v.ok


def test_number_followed_by_paren_or_period_counts_as_evidence():
    src = doc("v", "detection_threshold 값 확인(기본 0.5). 소프트웨어 v2.3.", document_name="Vision Manual", document_type="alarm_manual")
    v = AnswerValidator().validate("기본값은 0.5 이고 버전은 2.3 입니다.", [src], "q")
    assert v.unsupported_numbers == [] and v.ok


def test_chinese_characters_flagged_as_foreign_script():
    v = AnswerValidator().validate("정상 가동을恢复了습니다.", [MANUAL], "q")
    assert v.foreign_script and not v.ok
    assert not AnswerValidator().validate("정상 가동을 회복했습니다.", [MANUAL], "q").foreign_script


def test_negated_mention_of_missing_source_type_is_not_fabrication():
    v = AnswerValidator().validate("정상 범위는 -60 ~ -80 kPa 입니다. 과거 유지보수 기록은 제공된 문서에 없습니다.", [MANUAL], "q")
    assert v.fabricated_source_types == [] and v.ok


def test_spaced_out_citation_name_still_matches_document():
    tl = doc("t", "8월 릴리즈 12회.", document_name="02. 타임라인 — 2026년 3월 ~ 8월", section="8월", document_type="manual")
    v = AnswerValidator().validate("8월에 12회 릴리즈.\n출처: 02. 타임라인 — 2026 년 3 월 ~ 8 월 §8월", [tl], "q")
    assert v.invalid_citations == [] and v.ok
    assert [s.chunk_id for s in build_sources([tl, MANUAL], cited_names=["02. 타임라인 — 2026 년 3 월 ~ 8 월"])] == ["t"]


def test_claimed_maintenance_or_engineer_note_without_such_source_is_flagged():
    ans = "확인 순서: LED, ping, 케이블.\n참고: 과거 유지보수 이력에 따르면 LED가 꺼져 있었고, 현장 엔지니어 메모에 따르면 케이블 마모가 잦습니다."
    v = AnswerValidator().validate(ans, [MANUAL], "VIS-301?")          # MANUAL is an official alarm manual only
    assert v.fabricated_source_types == ["maintenance", "engineer_note"] and not v.ok
    ok = AnswerValidator().validate("과거 유지보수 이력: 2026-09-10 호스 교체.", [MANUAL, LOG], "이력?")
    assert ok.fabricated_source_types == []


def test_hedged_opening_followed_by_grounded_answer_is_not_a_refusal():
    text = ("서보 과부하 알람 (ROB-105) 이 발생하는 원인은 현재 제공된 문서에서는 확인할 수 없습니다.\n\n"
            "제공된 [Robot Controller Manual] §8.5 에 따르면 ROB-105 의 가능한 원인은 다음과 같습니다: "
            "1. Payload 초과: 현재 설정된 payload 가 30 kg 을 초과하는지 확인해야 합니다. 2. 축 기계적 간섭: 축을 수동으로 회전하여 확인. "
            "3. 감속기 이상: 소음이나 발열 확인.\n\n현재 문서에는 실제 로봇 상태가 포함되어 있지 않으므로 구체적인 상태는 판단할 수 없습니다.")
    v = AnswerValidator().validate(text, [MANUAL], "서보 과부하래")
    assert not v.is_refusal


def test_true_refusal_is_short_and_uncited():
    assert AnswerValidator().validate("현재 제공된 문서에서는 로봇 컨트롤러 IP 주소를 확인할 수 없습니다.\n\n"
                                      "제공된 문서에는 알람 코드와 절차만 있고 네트워크 설정은 없습니다.", [MANUAL], "IP?").is_refusal


def test_refusal_phrase_mid_answer_is_a_caveat_not_a_refusal():
    text = ("카메라 응답이 없어 VIS-301 알람 가능성이 높습니다.\n\n현재 제공된 문서에서는 카메라의 실제 상태를 확인할 수 없으므로 "
            "다음 순서로 점검합니다: 1. LED 2. ping 3. 케이블")
    v = AnswerValidator().validate(text, [MANUAL], "카메라가 응답이 없어")
    assert not v.is_refusal


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


def test_select_used_sources_drops_unrelated_docs_but_keeps_top1():
    from app.safety.source_validator import select_used_sources
    rob = doc("ROB_105_001", "[Alarm] ROB-105 Servo Overload. Payload 30 kg 이하. 감속기 소음 확인",
              document_name="Robot Controller Manual", section="8.5", document_type="alarm_manual")
    vis = doc("VIS_301_001", "[Alarm] VIS-301 Camera Communication Timeout. GigE 케이블, ping",
              document_name="Vision System Manual", section="5.1", document_type="alarm_manual")
    answer = "VAC-203은 진공압 저하 알람이며 정상 범위는 -60 ~ -80 kPa 입니다. 원인은 hose 누설, suction cup 손상입니다.\n출처: Vacuum System Manual §6.3"
    kept = [d.chunk_id for d in select_used_sources(answer, [MANUAL, LOG, rob, vis])]
    assert kept[0] == "VAC_203_001" and "ROB_105_001" not in kept and "VIS_301_001" not in kept
    # a doc holding an anchor no other doc has (-45) counts as used; a shared year (2026) does not
    assert "MAINT_1" in [d.chunk_id for d in select_used_sources("2026-09-10 사례에서 진공압 -45 kPa 정체", [rob, LOG])]
    # top-ranked doc is always kept, even with no overlap
    assert [d.chunk_id for d in select_used_sources("전혀 무관한 문장", [vis, rob])] == ["VIS_301_001"]


def test_select_used_sources_shared_anchor_credits_neither_doc():
    from app.safety.source_validator import select_used_sources
    a = doc("a", "v0.0.59에서 캐시 게이트 추가. 반경 5 cm.", document_name="Digest", section="v0.0.59", document_type="manual")
    b = doc("b", "v0.0.59 릴리즈 노트: 세그폴트 수정.", document_name="Timeline", section="8월", document_type="manual")
    # both docs contain v0.0.59; only `a` holds "5 cm" → b is dropped despite the shared version string
    kept = [d.chunk_id for d in select_used_sources("v0.0.59에서 캐시 히트 반경 5 cm 게이트가 추가됐습니다.", [a, b])]
    assert kept == ["a"]


def test_build_sources_with_answer_applies_usage_filter():
    rob = doc("ROB_105_001", "[Alarm] ROB-105 Servo Overload. Payload 30 kg", document_name="Robot Controller Manual", section="8.5", document_type="alarm_manual")
    srcs = build_sources([MANUAL, rob], cited_names=["Vacuum System Manual", "Robot Controller Manual"],
                         answer="정상 진공압은 -60 ~ -80 kPa 이며 hose 누설을 확인합니다.")
    assert [s.chunk_id for s in srcs] == ["VAC_203_001"]


def test_format_citation_uses_metadata_only():
    line = format_citation(build_sources([MANUAL, LOG]))
    assert line == "출처: Vacuum System Manual §6.3 (VAC-203); Maintenance Log"


def test_canonicalize_replaces_model_citation_line():
    answer = "네, 이력이 있습니다.\n\n출처: Maintenance Log §2026-09-10 DP-01 VAC-203 발생"
    out = canonicalize_citations(answer, build_sources([LOG]), is_refusal=False)
    assert out == "네, 이력이 있습니다.\n\n출처: Maintenance Log"


def test_canonicalize_drops_orphan_reference_heading():
    answer = "확인 순서: LED, ping, 케이블.\n\n**참고:**\n*   출처: Vision System Manual §5.1"
    out = canonicalize_citations(answer, build_sources([MANUAL]), is_refusal=False)
    assert "참고" not in out and out.startswith("확인 순서: LED, ping, 케이블.") and out.endswith("출처: Vacuum System Manual §6.3 (VAC-203)")


def test_canonicalize_removes_citation_on_refusal():
    out = canonicalize_citations("현재 제공된 문서에서는 확인할 수 없습니다.\n출처: X", build_sources([MANUAL]), is_refusal=True)
    assert "출처" not in out
