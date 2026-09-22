# Depalletizer Chatbot

디팔레타이징 시스템 작업자/엔지니어를 위한 **완전 로컬** 기술 지원 챗봇.
매뉴얼·알람·SOP·유지보수 이력을 검색(RAG)하고, 로컬 LLM이 **출처를 붙여** 설명합니다. 인터넷·클라우드 LLM 불필요.

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

## 6. 테스트·검증

```bash
./run.sh pytest tests/ -q                            # 86 tests, 모델 불필요, ~3s
./run.sh python3 scripts/test_hybrid_retrieval.py    # 검색 베이스라인 (샘플 10문항 R@1/R@3)
./run.sh python3 scripts/test_chat.py                # 검색→프롬프트→LLM→검증 E2E + 2턴 세션
```

Evaluation 스크립트(`scripts/run_eval.py`, Recall@5·MRR·Intent 정확도)는 Step 9에서 추가됩니다.

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

- **Phase 1 — Offline RAG MVP**: Step 1~8 완료 → Step 9 Evaluation
- Phase 2 — Corpus Pipeline (PDF/Markdown 파싱, semantic chunking, 메타데이터, `scripts/ingest.py`)
- Phase 3 — Intent Router, Troubleshooting Rule Engine, Multi-turn 진단 상태
- Phase 4 — Machine Tools (Mock → 실제 PLC/Robot/Vision), READ ONLY
- Phase 5 — Evaluation 자동화 (300+ 문항)
- Phase 6 — LoRA/SFT (답변 형식·용어, 지식은 계속 RAG)
