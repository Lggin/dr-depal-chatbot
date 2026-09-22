import pytest

from app.router.alarm_parser import extract_alarm_codes, normalize_alarm_code, strip_alarm_codes


@pytest.mark.parametrize("raw", ["VAC-203", "VAC203", "VAC_203", "vac 203", "Vac-203"])
def test_normalize_variants_to_canonical(raw):
    assert normalize_alarm_code(raw) == "VAC-203"


def test_normalize_short_prefix_and_leading_zero():
    assert normalize_alarm_code("E203") == "E-203"
    assert normalize_alarm_code("SV032") == "SV-032"


def test_normalize_rejects_non_codes():
    with pytest.raises(ValueError):
        normalize_alarm_code("hello")


def test_extract_dedup_preserves_order():
    text = "VAC203 떴고, 그 다음 rob-105, 다시 VAC-203"
    assert extract_alarm_codes(text) == ["VAC-203", "ROB-105"]


def test_extract_ignores_measurements_and_counts():
    assert extract_alarm_codes("진공압 -60 kPa, 1.0 mm, 9개 포인트, 0.5 MPa") == []


def test_extract_word_boundary():
    # "ABCD1234X" is not a code; but "GigE" has no digits.
    assert extract_alarm_codes("ABCD1234X GigE 케이블") == []


def test_strip_removes_code_spans():
    assert "VAC" not in strip_alarm_codes("VAC203 알람") and "알람" in strip_alarm_codes("VAC203 알람")
