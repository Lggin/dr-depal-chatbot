from ingestion.chunker import Chunker, kebab
from ingestion.markdown_parser import INTRO_TITLE, parse_markdown
from ingestion.metadata_generator import build_metadata, detect_components, infer_system, latest_date

MD = """# 03. 파지 자세 추정 노하우

도입 문단입니다.

## 1. 윗면 평면 적합 — IMU vs RANSAC

섹션 1 서문.

### 두 경로의 역사

RANSAC 전환은 v0.0.49 (#533, 2026-09-01)에 이루어졌다.

```python
# 코드 안의 ## 가짜 헤더
x = 1
```

### `plane_angle_max`는 게이트

| 값 | 의미 |
|---|---|
| 5 | 이전 |
| 10 | 현재 |

## 2. OBB

짧음.

### 2-1

아주 짧음.

### 2-2

이것도 짧음.
"""


def chars(s):  # 1 token per char — deterministic for tests
    return len(s)


def test_parser_titles_paths_and_code_fence_headers_ignored():
    doc = parse_markdown(MD)
    assert doc.title == "03. 파지 자세 추정 노하우"
    titles = [s.title for s in doc.sections]
    assert titles[0] == INTRO_TITLE
    assert "두 경로의 역사" in titles and "# 코드 안의 ## 가짜 헤더" not in " ".join(titles)
    hist = next(s for s in doc.sections if s.title == "두 경로의 역사")
    assert hist.path == ["1. 윗면 평면 적합 — IMU vs RANSAC", "두 경로의 역사"]
    assert "```python" in hist.text and "x = 1" in hist.text


def test_chunker_merges_short_intro_into_first_child_and_tiny_siblings():
    doc = parse_markdown(MD)
    chunks = Chunker(chars, max_tokens=5000, min_tokens=40).chunk_document(doc, "docs/03.md")
    ids = [c.chunk_id for c in chunks]
    assert all(c.n_parts == 1 for c in chunks) and len(ids) == len(set(ids))
    assert ids[0].startswith("docs-03-") and "03md" not in ids[0]

    # doc 도입부 (9 chars) → "## 1." intro (8 chars) → first "###" child collapse into one chunk
    hist = next(c for c in chunks if "두 경로의 역사" in c.section)
    assert hist.merged_from == [INTRO_TITLE, "1. 윗면 평면 적합 — IMU vs RANSAC", "두 경로의 역사"]
    assert "도입 문단입니다." in hist.text and "섹션 1 서문." in hist.text and "RANSAC 전환" in hist.text
    assert hist.search_text.startswith("03. 파지 자세 추정 노하우 > 도입부 · 1. 윗면 평면 적합 — IMU vs RANSAC · 두 경로의 역사")
    # "## 2. OBB" intro + two tiny children collapse into one chunk
    obb = next(c for c in chunks if c.section.startswith("2. OBB"))
    assert obb.merged_from == ["2. OBB", "2-1", "2-2"]
    assert "아주 짧음" in obb.text and "이것도 짧음" in obb.text
    # a section that is big enough stays alone
    gate = next(c for c in chunks if "plane_angle_max" in c.section)
    assert gate.merged_from == [] and "| 10 | 현재 |" in gate.text


def test_chunker_splits_oversize_section_on_blocks_with_overlap_and_atomic_table():
    body = "\n\n".join(f"문단 {i} " + "가" * 60 for i in range(8)) + "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    md = f"# T\n\n## 큰 섹션\n\n{body}\n"
    chunks = Chunker(chars, max_tokens=220, min_tokens=10, overlap_tokens=70).chunk_document(parse_markdown(md), "x.md")
    assert len(chunks) >= 3 and all(c.n_parts == len(chunks) for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].chunk_id.endswith("-p1") and "(1/" in chunks[0].search_text
    table_chunks = [c for c in chunks if "| 1 | 2 |" in c.text]
    assert len(table_chunks) == 1 and "| 3 | 4 |" in table_chunks[0].text     # table not split
    assert all(c.tokens <= 220 for c in chunks)
    # overlap: last block of part 1 reappears at the start of part 2
    last_block_p1 = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(last_block_p1)


def test_kebab_id_is_deterministic_and_clean():
    assert kebab("docs/03-grasp.md", "1. 윗면", "두 경로") == "docs-03-grasp-1-윗면-두-경로"
    assert kebab("a b", "c__d") == "a-b-c-d"


def test_metadata_extraction():
    doc = parse_markdown(MD)
    hist = next(c for c in Chunker(chars, 5000, 1).chunk_document(doc, "docs/03.md") if "두-경로의-역사" in c.chunk_id)
    m = build_metadata(hist, {"document_type": "manual", "trust_level": 90, "system": "geometry", "tags": ["x"]},
                       {"machine": "depal", "manufacturer": "DR", "alarm_codes": False})
    assert set(m["components"]) == {"ransac", "imu"} and m["component"] == "ransac" and m["system"] == "geometry"
    assert "v0.0.49" in m["tags"] and "#533" in m["tags"] and m["tags"][0] == "x"
    assert m["software_version"] == "v0.0.49" and m["as_of_date"] == "2026-09-01"
    assert m["alarm_code"] is None and m["document_name"] == "03. 파지 자세 추정 노하우"
    assert m["trust_level"] == 90 and m["search_text"] == hist.search_text


def test_component_and_system_inference():
    comps = detect_components("RF-DETR과 SAM으로 라벨링, TensorRT로 배포")
    assert set(comps) == {"rf-detr", "sam", "tensorrt"}
    assert infer_system(["rf-detr", "sam"], default="general") == "vision"
    assert infer_system(["tensorrt"], default="training") == "training"     # single hit doesn't override an explicit default
    assert infer_system([], default="config") == "config"
    assert latest_date("2026-08-20 그리고 2026.09.01") == "2026-09-01" and latest_date("없음") is None
