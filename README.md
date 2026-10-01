# Depalletizer Chatbot

디팔레타이징 시스템 작업자/엔지니어를 위한 **완전 로컬** 기술 지원 챗봇.
매뉴얼·알람·SOP·유지보수 이력을 검색(RAG)하고, 로컬 LLM이 **출처를 붙여** 설명합니다. 인터넷·클라우드 LLM 불필요.

> **개인 보관본 안내** — 이 저장소는 사내 프로젝트의 코드·아키텍처·평가 코드만 담은 개인 포트폴리오용 사본입니다.
> 실제 운영 코퍼스(사내 노하우 문서 621청크, `corpus/processed/knowhow_*.json`)와 그 문서에서 만든
> 평가셋·결과(`evaluation/datasets/knowhow_*.json`, `evaluation/results/latest_*knowhow*`)는 사내 자료라 제외했습니다.
> 공개 샘플 코퍼스 12청크(`corpus/processed/sample_chunks.json`)와 샘플 평가셋(`test_questions.json`)으로 전 과정이 재현됩니다.
> README/DEVELOPMENT.md의 knowhow 관련 수치는 제외된 코퍼스에서 측정한 기록입니다.

```
질문 → Alarm 파서 / Glossary → Dense(BGE-M3) ∥ Sparse(BM25+Kiwi) → RRF → Reranker(gap 컷)
     → Prompt(토큰 예산) → Qwen3.5-9B Q4 (llama.cpp) → 검증(수치·코드·출처·위험명령) → 답변 + 출처
```

| 구성 | 선택 | 비고 |
|---|---|---|
| LLM | `Qwen/Qwen3.5-9B` Q4_K_M GGUF (`unsloth/Qwen3.5-9B-GGUF`) | llama.cpp, 전 레이어 GPU, ~5.9GB VRAM, ~70 tok/s |
| 임베딩 | `BAAI/bge-m3` (1024d, 8192 tok) | |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | 순위 간 격차(gap) 기준 컷 |
| 벡터DB | Qdrant **v1.19.1** | 클라이언트·서버 버전 일치 필수 |
| 백엔드 | FastAPI + SQLite(로그) | |
| GPU | 16GB (RTX 4090 Laptop 검증) | 임베딩·리랭커는 VRAM<20GB면 CPU 자동 |

## 1. 설치

```bash
cd ~/depalletizer_chatbot
python3.10 -m venv venv && source venv/bin/activate
pip install --index-url https://pypi.org/simple/ -r requirements.txt   # 사내 Nexus 401 우회

# llama-cpp-python은 CUDA 빌드가 필요 (Qwen3.5 = 'qwen35' 아키텍처, 2026-08 이후 llama.cpp)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 \
  pip install --no-cache-dir llama-cpp-python==0.3.35

cp .env.example .env   # 필요 시 수정
```

> **항상 `./run.sh <명령>`으로 실행하세요.** 이 랩 노트북은 쉘 프로필이 ROS2를 source해서 `PYTHONPATH`에 ROS 패키지가 venv 앞에 끼어듭니다. `run.sh`가 이를 제거하고 `PYTHONNOUSERSITE=1`을 켭니다.

## 2. Qdrant 실행

```bash
docker compose up -d qdrant          # qdrant/qdrant:v1.19.1, 포트 6333
curl -s localhost:6333/healthz       # "healthz check passed"
```

## 3. 모델 준비

```bash
# LLM GGUF (5.3GB) → models/gguf/  (git 무시됨)
./run.sh python3 -c "from huggingface_hub import hf_hub_download as d; \
  d('unsloth/Qwen3.5-9B-GGUF','Qwen3.5-9B-Q4_K_M.gguf',local_dir='models/gguf')"

# 임베딩·리랭커는 첫 실행 시 HF 캐시로 자동 다운로드 (각 ~2.2GB)
./run.sh python3 scripts/test_llm.py        # LLM 로드·생성 스모크 테스트
```

## 4. 코퍼스 추가 및 색인

Phase 1은 `corpus/processed/sample_chunks.json`(12청크, 명세 §5/§6 형식)을 사용합니다.
청크 하나 = `chunk_id, document_type, document_name, section, alarm_code, symptoms, trust_level, text, …`.
PDF/Markdown → 청크 자동 변환 파이프라인은 Phase 2(`ingestion/`)에서 제공됩니다.

```bash
./run.sh python3 scripts/ingest_sample.py   # 임베딩 → Qdrant 컬렉션 재생성 (payload 인덱스 포함)
```

용어 사전 `corpus/glossary/glossary.json`은 **재현된 검색 실패가 있는 항목만** 등재합니다. 후보는 `glossary_candidates.json`.

## 5. API 실행

```bash
./run.sh uvicorn app.main:app --host 0.0.0.0 --port 8000
# 모델 3개 로드 ~8초 후 /api/ready → 200
```

| 엔드포인트 | 설명 |
|---|---|
| `GET  /api/health` | 프로세스 생존 |
| `GET  /api/ready` | llm/retrieval/qdrant/database 준비 상태 (미준비 503) |
| `POST /api/session` → `DELETE /api/session/{id}` | 세션 생성/삭제 (히스토리 1턴 유지, TTL 30분) |
| `POST /api/chat` | 질의응답 |
| `POST /api/feedback` | `{log_id, feedback: up\|down, note}` |
| `GET  /api/admin/documents` | 색인된 문서/청크 통계 |
| `GET  /api/admin/logs?limit=&session_id=` | 질의 로그(§25) |
| `POST /api/admin/ingest`, `/reindex` | Phase 2까지 501 |

### Chat 예제

```bash
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"session_id": "test01", "message": "VAC-203이 뭐야?"}'
```

```json
{
  "answer": "VAC-203은 진공압이 정상 범위에 도달하지 못해 발생하는 'Vacuum Pressure Low' 알람입니다. ...\n\n출처: Vacuum System Manual §6.3 (VAC-203)",
  "intent": "knowledge",
  "sources": [{"chunk_id": "VAC_203_001", "document": "Vacuum System Manual", "section": "6.3", "alarm_code": "VAC-203", "trust_level": 100, "document_type": "alarm_manual"}],
  "validation": {"ok": true, "is_refusal": false, "unsupported_numbers": [], "unsupported_codes": [], "invalid_citations": [], "danger_flags": []},
  "alarm_codes": ["VAC-203"],
  "retrieval": {"documents_found": 4, "reranked": true, "top_k": 4, "sources": [...], "scores": [...]},
  "timing_ms": {"retrieval": 1618.0, "generation": 1916.5, "validation": 0.5},
  "latency_ms": 3535.0,
  "log_id": 1
}
```

같은 `session_id`로 `"그 알람 원인이 뭐야?"`를 보내면 직전 턴의 `VAC-203`이 자동으로 이어집니다(`retrieval_query` 필드에 표시).
문서에 없는 질문은 `"현재 제공된 문서에서는 해당 내용을 확인할 수 없습니다."`로 답하고 `sources`는 비어 있습니다.

`sources`는 LLM 텍스트가 아니라 **검색 메타데이터**에서 생성되며, 답변 끝의 `출처:` 줄도 그것으로 교체됩니다.

### 의도별 라우팅 (Phase 3)

질문은 규칙 기반 라우터(`app/router/intent_router.py`, LLM 미사용)로 분류되어 `intent`/`route` 필드에 표시됩니다.

| 질문 예 | intent | 동작 |
|---|---|---|
| `VAC-203이 뭐야?`, `진공압 정상 범위 알려줘` | knowledge | 문서 검색 → 답변 + 출처 |
| `VAC-203 지난번에도 있었어?` | history | 유지보수 기록 우선 검색(`history.filtered`), 없으면 전체 |
| `지금 진공압 얼마야?` | live_status | 설비 도구 스냅샷 → 현재값 + 문서 기준값 비교 (도구 없으면 미연결 안내) |
| `지금 박스를 왜 못 집어?` | live_diagnosis | 현재 알람·값 vs 기준값 → 우선 점검 순서 (도구 없으면 ⚠️ 표시 후 문서만) |
| `박스를 못 집어.` | troubleshooting | **단계별 진단 시작**(아래) |
| `박스 흡착 실패 원인이 뭐야?` | troubleshooting | 설명 요구 → 문서로 바로 답변 |
| `안녕` | general | 도우미 소개 |

### 단계별 진단 (Troubleshooting)

증상을 말하면 결정트리(`configs/troubleshooting.yaml`)가 한 번에 하나씩 예/아니오로 확인하고, 조건이 모이면 문서 근거로 점검 순서를 제시합니다. 세션에 상태가 유지됩니다.

```
U: 박스를 못 집어.
A: '박스 흡착/픽업 실패' 증상을 단계적으로 확인하겠습니다.
   비전 화면에서 박스가 정상적으로 검출되고 있나요?        ← 결정론, 0ms
U: 응
A: 확인했습니다. 로봇이 박스 상면 중앙까지 정상적으로 내려오나요?
U: 응
A: 확인했습니다. 흡착 시 진공압이 정상 범위(-60 kPa 이상)에 도달하나요?
U: 아니, -45 정도에서 안 올라가
A: (check_vacuum 단계) 1. 진공압 확인 … 2. 호스 누설 … 3. suction cup … 4. vacuum generator …
   출처: Vacuum System Manual §6.3 (VAC-203); …            ← RAG + LLM
   점검 후 결과를 알려 주세요. 해결되면 "해결됐어"라고 답해 주시면 됩니다.
U: 해결됐어
A: 문제가 해결되었다고 하셨습니다. …
```

응답의 `troubleshooting` 필드에 `issue / facts / pending_fact / current_step`이 담깁니다. 진단 중 다른 주제를 질문하면 진단 상태는 폐기됩니다.

### 설비 데이터 도구 (Phase 4, READ ONLY)

`지금 …` 질문과 진단 트리는 설비 조회 도구(`app/tools/`)를 사용합니다. V1은 **읽기 전용**이며 `ToolRegistry`가 쓰기 도구 등록 자체를 거부합니다(명세 §22). 기본 백엔드는 **Mock**(`TOOLS_BACKEND=mock`) — `configs/machine.yaml`의 시나리오를 읽습니다.

```bash
curl -s localhost:8000/api/admin/tools                       # 도구 11종 목록, read_only, 현재 시나리오
curl -s localhost:8000/api/admin/machine                     # 현재 스냅샷 (HOLD / VAC-203 / -43 kPa / ...)
curl -s -X POST "localhost:8000/api/admin/mock/scenario?name=camera_timeout"   # vac203_low | normal | camera_timeout
```

| 질문 | 도구 | 문서 | 응답 |
|---|---|---|---|
| `지금 진공압 얼마야?` | ✓ | ✓(기준값) | "현재 -43 kPa, 정상 -60~-80보다 높음" |
| `진공압 정상 범위가 얼마야?` | ✗ | ✓ | 문서만 |
| `지금 박스를 왜 못 집어?` | ✓ | ✓ | 알람·현재값 vs 기준값 → 정상 계통 제외 → 우선 점검 순서 (명세 §18) |
| `박스를 못 집어.` | ✓ | 권고 턴만 | 도구로 확인된 fact는 건너뛰고 남은 것만 질문 |
| `VAC-203 지난번에도 있었어?` | 유지보수 DB | ✓ | 과거 기록 (현재 원인으로 단정하지 않음) |

응답의 `tools_used`(호출 도구), `machine_state`(답변에 허용된 현재 값)로 어떤 값이 실제 조회된 것인지 추적할 수 있습니다. 실기기 연결은 `app/tools/mock.py`의 각 클래스를 같은 이름·인터페이스의 PLC/로봇/비전 어댑터로 교체하면 됩니다(`TOOLS_BACKEND=none`이면 도구 없이 동작).

**트리 추가/수정**: `configs/troubleshooting.yaml`에 `issues.<id>`를 추가합니다 — `triggers`(질문 매칭 문자열), `facts`(예/아니오 질문, Phase 4용 `tool` 키), `order`(질문 순서), `steps`(`when` 조건이 모두 맞으면 권고, `doc_query`로 근거 검색), `escalate`. 로드 시 `when`/`order`가 미정의 fact를 참조하면 기동이 실패합니다. 서버 재시작 필요.

## 6. 테스트·검증

```bash
./run.sh pytest tests/ -q                            # 86 tests, 모델 불필요, ~3s
./run.sh python3 scripts/test_hybrid_retrieval.py    # 검색 베이스라인 (샘플 10문항 R@1/R@3)
./run.sh python3 scripts/test_chat.py                # 검색→프롬프트→LLM→검증 E2E + 2턴 세션
```

### 평가 (`scripts/run_eval.py`)

```bash
# 검색만 (LLM 없음, 초 단위): Hit@k / Recall@k / MRR / nDCG / 알람 파싱 정확도
./run.sh python3 scripts/run_eval.py --mode retrieval --dataset evaluation/datasets/knowhow_questions.json --collection knowhow_docs

# 전체 (LLM): must_include·거절·인용·라우트·도구 선택 정확도 + 검증기 플래그 + 지연
./run.sh python3 scripts/run_eval.py --mode full --collection depalletizer_docs --show-failures

# + LLM judge: 문장별 groundedness, reference 대비 completeness/contradiction, 과잉 인용
./run.sh python3 scripts/run_eval.py --mode full --collection knowhow_docs --dataset evaluation/datasets/knowhow_questions.json --judge --show-ungrounded
```

결과는 `evaluation/results/latest_<mode>_<dataset>__<collection>.json`에 저장되고, 이전 실행 대비 하락(answer_correct·citation·refusal_recall·tool_selection 0 허용, groundedness 0.03, MRR 0.02)이 있으면 **종료코드 1**(회귀 게이트).

| 데이터셋 | 문항 | 성격 |
|---|---|---|
| `test_questions.json` | 39 | 샘플 알람 코퍼스용. 의도·라우트·도구·후속질문·범위밖·LIVE 포함 |
| `knowhow_questions.json` | 40 | 이전 프로젝트 held-out(dev 25·operator 15). **코퍼스 문장과 독립적인 유일한 세트** |
| `knowhow_generated.json` | ~300 | `evaluation/generate_questions.py`로 청크당 생성(`reviewed: false`). 검색 recall·groundedness용 — 정답률의 절대 지표로 쓰지 말 것 |

**Judge 신뢰성**: `./run.sh python3 evaluation/judge_calibration.py` — 숫자 조작·파생 계산·원인 날조·누락·정직한 "문서에 없음"·과잉 인용 7방향을 모두 맞혀야 통과(현재 7/7). judge 프롬프트를 바꾸면 반드시 재실행.

**실사용 질문 회수**: `./run.sh python3 scripts/harvest_logs.py --since-days 7 --feedback down` → `/api/chat` 로그(§25)를 평가 후보 JSON으로. reviewer가 `must_include`·`expected_source`를 채우고 `reviewed: true`로 바꾸면 데이터셋에 편입.

## 7. 설정

`app/config.py`(pydantic-settings) ← `.env` / 환경변수. 주요 값:

| 키 | 기본 | 의미 |
|---|---|---|
| `LLM_MODEL_PATH` | `models/gguf/Qwen3.5-9B-Q4_K_M.gguf` | |
| `LLM_ENABLE_THINKING` | `false` | Qwen 생각모드 (그라운딩 환경에선 지연만 추가) |
| `AUX_DEVICE` | `auto` | 임베딩·리랭커 위치 (`auto`=VRAM≥20GB면 cuda) |
| `DENSE_TOP_K / SPARSE_TOP_K / FUSION_TOP_K / RERANK_TOP_K` | 20/20/20/4 | |
| `RERANK_GAP_CUTOFF` | 3.0 | 인접 리랭커 점수 격차 컷 (top_k 상한보다 먼저 적용) |
| `EXACT_PIN_MIN_TRUST` | 85 | 알람코드 exact 히트 중 상단 고정할 최소 신뢰도 |
| `CONTEXT_MAX_TOKENS / CONTEXT_MAX_DOCS` | 3000 / 5 | |
| `HISTORY_TURNS` | 1 | 더 길면 예전 답변을 반복하는 부작용 |
| `SKIP_MODEL_LOAD` | `false` | 모델 없이 API만 기동 (테스트) |

## 8. 프로젝트 구조

```
app/
  api/            health · chat · session · admin
  llm/            embedding_client · client(llama.cpp) · prompt · generator(ChatPipeline)
  rag/            qdrant_client · dense · sparse(BM25+Kiwi) · hybrid(RRF+exact) · reranker · retriever · query_expander · tokenizer
  router/         alarm_parser (regex 정규화)
  safety/         command_guard · answer_validator · source_validator
  memory/         session (인메모리, TTL)
  database/       SQLAlchemy 모델 · 로그 저장소
  schemas/        chat · document · machine · retrieval · tool
  config.py · dependencies.py(build_chat_pipeline) · main.py
corpus/processed/sample_chunks.json · corpus/glossary/
scripts/  ingest_sample · test_embedding · test_dense/sparse/hybrid_retrieval · test_llm · test_prompt · test_chat
evaluation/sample_cases.py
tests/   (86)
DEVELOPMENT.md   단계별 실측 기록·결정 근거
```

## 9. 문제 해결

- **`/api/chat` 503** — 모델 미로드. `/api/ready`로 어느 서비스가 false인지 확인.
- **Qdrant 응답 파싱 오류** — 클라이언트/서버 버전 불일치. 둘 다 1.19.1인지 확인.
- **`ModuleNotFoundError` 또는 이상한 패키지 버전** — `./run.sh` 없이 실행했을 가능성(ROS PYTHONPATH 누출).
- **CUDA OOM** — `AUX_DEVICE=cpu`(기본 auto가 이미 16GB에선 cpu). LLM만 ~6GB.
- **llama-cpp-python 빌드 실패** — `nvcc --version` 확인. 이 노트북 드라이버는 CUDA 12.4 상한이라 cu13x 프리빌드 휠은 동작하지 않음.

## 로드맵

- ✅ Phase 1 — Offline RAG MVP (검색 Hit@1 0.939 / 답변 37/37 on 샘플)
- ✅ Phase 2 — Corpus Pipeline: `dr-depal-knowhow` 마크다운 → 621청크, `scripts/ingest.py` / `POST /api/admin/reindex` (검색 Hit@1 0.821, 답변 0.975 on 40 held-out)
- ✅ Phase 3 — Intent Router, Troubleshooting 결정트리, 세션 진단 상태 (intent/route 정확도 1.0 on 샘플)
- ✅ Phase 4 — Machine Tools: Mock 11종(READ ONLY), LIVE 라우트·트리 facts·이력에 도구 결과 주입, 시나리오 전환
- ✅ Phase 5 — LLM judge(캘리브레이션 7/7), 도구 선택 정확도, 300문항 반자동 생성, 로그 회수
- Phase 6 — LoRA/SFT (답변 형식·용어, 지식은 계속 RAG)
