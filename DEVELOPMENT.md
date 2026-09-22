# Development Progress

## Phase 1: Offline RAG MVP

### ✅ Step 1: Embedding & Indexing (COMPLETE)

**파일:**
- `app/llm/embedding_client.py` - BGE-M3 임베딩 클라이언트
- `app/rag/qdrant_client.py` - Qdrant 벡터DB 클라이언트
- `scripts/test_embedding.py` - 통합 테스트 스크립트

**구현된 기능:**
- BGE-M3 모델 로드 (CUDA/CPU 자동 선택)
- 대량 텍스트 임베딩 (배치 처리)
- Qdrant 연결 및 컬렉션 관리
- 문서 업서트 (메타데이터 포함)
- 벡터 검색

**테스트 방법:**

```bash
docker compose up -d qdrant
./run.sh python3 scripts/test_embedding.py   # 통합(실서버)
./run.sh pytest tests/ -v                    # 단위(:memory:, Docker 불필요)
```

**2026-09-21 실측 결과:** BGE-M3 로드 ~10s(1024d), 5건 업서트, 질의 '진공 시스템 문제' →
흡착/진공 문서 0.694 > VAC-203 0.654 > 영어 vacuum 문서 0.523 (교차언어 매칭 OK).

**이 단계에서 잡은 함정 (재발 방지):**
- `sentence-transformers 2.2.2` → `huggingface_hub>=0.20`에서 `cached_download` 제거로 import 실패. **2.7.0**으로 핀.
- `qdrant-client 1.7.0` vs 서버 `latest`(1.19.1) → 응답 파싱 실패. **클라이언트·서버 버전 일치**(둘 다 1.19.1로 핀, docker-compose는 `latest` 금지).
- 이 노트북은 쉘 프로필이 ROS2를 source해 `PYTHONPATH`에 ROS 패키지가 venv 앞에 끼어듦 → **항상 `./run.sh`로 실행**(`env -u PYTHONPATH PYTHONNOUSERSITE=1`).
- 포인트 id를 배치 내 순번으로 쓰면 배치 간 덮어쓰기 → `uuid5(chunk_id)`로 결정론적 id(재색인 시 중복 대신 갱신).

**예상 출력:**
```
[1a] Loading embedding model...
✅ Model loaded. Dimension: 1024

[1b] Generating embeddings...
✅ Generated 5 embeddings
   Shape: (5, 1024)

[1c] Connecting to Qdrant...
✅ Connected to Qdrant. Version: ...

[1d] Creating collection...
✅ Collection created: depalletizer_docs

[1e] Upserting documents...
✅ Upserted 5 documents

[1f] Collection info...
   Points count: 5
   Vectors count: 5

[1g] Testing search...
   Query: '진공 시스템 문제'
   Results:
   1. Score: 0.8234
      ID: doc_000
      Text: VAC-203: 진공압이 정상 범위에 도달하지 못함
   ...

✅ Step 1 Test Completed Successfully!
```

---

### ✅ Step 2: Dense Retriever (COMPLETE, 2026-09-21)

**파일:**
- `app/schemas/retrieval.py` — `RetrievedDoc`/`RetrievalResult` (dense/sparse/hybrid/rerank 공용)
- `app/rag/metadata_filter.py` — dict → Qdrant `Filter` (exact / match-any / range), 인덱스 필드 목록
- `app/rag/dense_retriever.py` — 질의 임베딩 → Qdrant ANN → `RetrievalResult`
- `app/dependencies.py` — `resolve_aux_device()`: 총 VRAM < 20GB면 임베딩·리랭커 CPU(과거 16GB OOM 재현 방지), `AUX_DEVICE`로 override
- `app/rag/qdrant_client.py` — `search(filters=)`, `scroll_by_filter()`(벡터 없는 exact 조회), 컬렉션 생성 시 payload 인덱스 7개
- `corpus/processed/sample_chunks.json` — 명세서 §5/§6 형식 샘플 12청크(alarm 5, manual 1, sop 2, troubleshooting 1, maintenance 1, faq 1, engineer_note 1)
- `scripts/ingest_sample.py`, `scripts/test_dense_retrieval.py`, `tests/test_dense_retriever.py`(6 tests, fake embedder)

**실행:**
```bash
./run.sh python3 scripts/ingest_sample.py          # 샘플 12청크 색인 (aux=cpu 자동 선택, ~20s)
./run.sh python3 scripts/test_dense_retrieval.py   # dense 단독 베이스라인
./run.sh pytest tests/ -q                          # 10 passed
```

**Dense 단독 베이스라인 (10문항): R@1 = 8/10, R@3 = 10/10**

| 질의 | 순위 | 1등 | 해석 |
|---|---|---|---|
| `VAC203이 뭐야?` (비정규 코드) | 2 | ENG_NOTE(trust 50) | 공식 매뉴얼(trust 100)과 0.497 vs 0.49x 근접 — dense는 신뢰도를 모름 → **Alarm 파서 exact-match 경로 필요**(명세 §7) |
| `빨판이 박스를 못 잡아` (구어체) | 2 | FAQ_BOX_TYPES | "빨판"이 코퍼스에 없음 → **glossary 확장 필요**(§8). 실패가 재현됐으므로 `빨판→suction cup` 등재는 정당 |
| 나머지 8건 | 1 | — | 필터 2건(document_type, trust_level 범위) 포함 정상 |

**확인된 사실:** BGE-M3 `max_seq_length=8192`(e5-large 512 한계 없음). VRAM 15.5GB → aux 모델 CPU 자동 배치. 로컬(`:memory:`) Qdrant에선 payload 인덱스가 무시됨(경고만, 테스트엔 영향 없음).

---

### ✅ Step 3: Sparse Retriever + Alarm Parser (COMPLETE, 2026-09-21)

**파일:**
- `app/router/alarm_parser.py` — regex 정규화 `VAC203/VAC_203/vac 203/E203/SV032 → VAC-203/E-203/SV-032`, `extract_alarm_codes()`, `strip_alarm_codes()`. 결정론적(LLM·벡터 미사용, 명세 §7·§30-7)
- `app/rag/tokenizer.py` — `KoreanTokenizer`: 알람코드를 **먼저 뽑아 단일 정규화 토큰**으로 보존 → 나머지를 Kiwi 형태소 분석, 내용어 태그(NNG/NNP/NNB/NR/SL/SN/SH/VV/VA/XR)만 유지, 소문자화. Kiwi가 `VAC203`→`VAC`+`203`으로 쪼개는 문제를 이렇게 우회
- `app/rag/sparse_retriever.py` — `SparseRetriever.build(chunks)` / `.from_qdrant(store)`(Qdrant payload에서 색인 → dense와 동일 청크 보장), BM25Okapi, 0점 제거, 메타데이터 post-filter(`_matches`: build_filter와 동일 의미)
- `app/rag/qdrant_client.py` — `scroll_all()` 페이지네이션
- `evaluation/sample_cases.py` — 10문항 공용 케이스 + `report()` (dense/sparse/hybrid 비교용)
- 테스트 20개 추가 (alarm_parser 7, tokenizer 3, sparse 10) → **30 passed**

**실행:**
```bash
./run.sh python3 scripts/test_sparse_retrieval.py   # 임베딩 모델 불필요, <2s
```

**Sparse 단독 베이스라인 (10문항): R@1 = 6/10, R@3 = 9/10** — dense(8/10, 10/10)와 상보적

| 질의 | dense | sparse | 해석 |
|---|---|---|---|
| `VAC-203이 뭐야?` / `VAC203이 뭐야?` | 1 / 2 | 3 / 3 | 정규화는 성공(`tokens=['vac-203']`, 두 표기 동일 결과). 그러나 BM25 길이 정규화로 **짧은 유지보수 로그·엔지니어 노트가 긴 공식 매뉴얼을 이김**. dense도 sparse도 신뢰도를 모름 → **Step 4에서 exact alarm_code 매치 + trust 정렬 단계를 최우선으로**(명세 §7 우선순위 1) |
| `빨판이 박스를 못 잡아` | 2 | **no hits** | "빨판"이 코퍼스에 전무 → BM25는 원리적으로 불가. dense가 2등으로 살려둠 → RRF로 회수 + glossary(`빨판→suction cup`)로 1등 목표 |
| `진공 문제 공식 매뉴얼만` | 1 | 2 | 1등이 VAC_MANUAL_SPEC_001(공식 사양) — 사실상 정답 동급, 케이스의 기대값이 좁은 것 |
| 나머지 7건 | 1 | 1 | — |

**Step 4 설계에 반영할 결론:** (1) 코드 질의는 exact 경로가 없으면 dense/sparse 어느 쪽도 trust를 못 반영 (2) 구어체는 glossary 없이는 sparse가 0건 (3) 두 검색기의 실패가 겹치지 않아 RRF 융합 가치가 실측으로 확인됨.

---

### ✅ Step 4: Hybrid + Reranker (COMPLETE, 2026-09-21)

**파일:**
- `app/rag/query_expander.py` — `GlossaryExpander`: 질의에 glossary 키가 있으면 동의어를 **원문 뒤에 덧붙임**(원문 보존)
- `app/rag/hybrid_retriever.py` — `rrf_fuse()`(k=60) + `HybridRetriever`: ①`extract_alarm_codes` → `scroll_by_filter(alarm_code)` **exact 히트를 trust 내림차순으로 최상단 고정** ②glossary 확장 ③dense∥sparse → RRF → fusion top-20
- `app/rag/reranker.py` — `bge-reranker-v2-m3` 크로스인코더. `gap_cut()`: **후보 전체 채점 → 인접 점수 격차>3.0에서 절단 → 그 다음 top_k=4 상한**(순서 뒤바뀌면 gap 규칙이 top_k 너머를 못 보는 과거 버그). 청크 앞에 `document_name/section/alarm_code`를 붙여 채점
- `app/rag/retriever.py` — `RetrievalPipeline.search()`: hybrid → rerank → exact 재고정(코드 질의는 리랭커 의견과 무관하게 공식문서 우선)
- config: `rrf_k=60`, `rerank_gap_cutoff=3.0`, `rerank_max_length=1024`(모델 한계 8194), `glossary_path`
- 테스트 11개 추가(RRF, gap_cut, glossary, exact 고정) → **41 passed**. exact 단계의 `rank=0` 스키마 위반을 테스트가 잡아냄

**실행:**
```bash
./run.sh python3 scripts/test_hybrid_retrieval.py   # hybrid / hybrid+rerank 베이스라인 + 상세 2건
```

**베이스라인 (10문항)** — 단계별 누적 효과:

| 단계 | R@1 | R@3 | 비고 |
|---|---|---|---|
| dense 단독 | 8 | 10 | |
| sparse 단독 | 6 | 9 | `빨판` 0건 |
| hybrid(exact+glossary+RRF), 리랭커 전 | 9 | 9 | exact 고정으로 `VAC-203`/`VAC203` 둘 다 공식매뉴얼 1등 확정 |
| **hybrid + reranker** | **9** | **10** | `빨판` 5등→**1등**(리랭커가 회수, 2등과 격차 2.7) |

- 남은 1건 `진공 문제 공식 매뉴얼만`: 리랭커가 VAC-201(센서 이상)을 VAC-203보다 앞에 둠 — 둘 다 trust 100 공식 진공 알람, 질의 자체가 모호. 평가셋 기대값 문제로 분류, 미수정
- `VAC203이 뭐야?` 최종 컨텍스트: `[exact] 공식매뉴얼(100) → 유지보수로그(70) → 엔지니어노트(50) → [rerank] 트러블슈팅(85)` — LLM이 공식/이력/경험담을 구분해 받음(명세 §21)
- **latency: median 1.4s / query (CPU)** — 임베딩+BM25+리랭커 12~20건. GPU면 수백 ms 예상. Phase 1 허용

**glossary 교훈 재확인:** 스켈레톤 때 명세서 예시 그대로 넣은 10개 항목 중 `박스→box carton case package`가 `빨판` 질의를 리랭커 전 **2등→6등으로 악화**시킴(BM25가 FAQ의 carton에 쏠림). 재현된 실패가 있는 `빨판` 1개만 `glossary.json`에 남기고 나머지는 `glossary_candidates.json`으로 격리(승격 조건 명시). 리랭커 전 R@1 8→9로 회복.

---

### 🔄 Step 5: Local LLM Client (IN PROGRESS, 2026-09-22)

**사전 조사 결과 (결정 근거):**
- `Qwen/Qwen3.5-9B` 존재. GGUF: `unsloth/Qwen3.5-9B-GGUF` `Qwen3.5-9B-Q4_K_M.gguf` **5.29GB** 채택(bartowski 5.75GB 대안)
- GGUF `general.architecture = qwen35` (HF `model_type=qwen3_5`, `Qwen3_5ForConditionalGeneration`, ctx 262144). **`qwen3`과 다른 신규 아키텍처** → 2026-08 이후 llama.cpp 필요
- `llama-cpp-python 0.3.35`(2026-08-17)가 내장한 llama.cpp `4df29be`(08-16)에 `LLM_ARCH_QWEN35` 포함 ✅
- 이 노트북 드라이버 550.144 → **CUDA 12.4 상한**. 프리빌드 휠: cu132(0.3.35)는 드라이버 불가, cu121은 0.3.19까지(qwen35 미지원) → **소스 빌드**: `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 pip install llama-cpp-python==0.3.35` (로컬 nvcc 12.6, sm_89 SASS라 minor-version 호환 기대)
- 폴백: Ollama(미설치, 시스템 서비스 설치 필요해 보류) — 라이브러리에 `qwen3.5:*` 태그 존재. 2차 폴백: transformers+bitsandbytes 4bit(단, transformers 4.36은 qwen3_5 미지원 → torch/transformers 상향 필요)
- `transformers 4.36.2` 유지 가능 이유: LLM 경로가 llama.cpp라 transformers는 리랭커(XLM-R)에만 쓰임

**설계 (`app/llm/client.py`):**
- `LLMClient` Protocol(load/generate/stream/count_tokens) — 백엔드 교체 지점
- `LlamaCppClient`: GGUF 내장 chat_template을 **Jinja2로 직접 렌더**(`enable_thinking=False` 전달 목적 — `create_chat_completion`은 템플릿 kwargs를 못 넘김). 생각모드 OFF는 이전 프로젝트 실측 결론(few-shot·그라운딩 환경에선 정확도 이득 없이 지연만 추가). 모델이 그래도 `<think>`를 내면 `_strip_think`
- stop = `<|im_end|>`, `<|endoftext|>`; repeat_penalty 1.05
- config: `llm_model_path`, `llm_n_gpu_layers=-1`, `llm_enable_thinking=False`

**검증 스크립트:** `./run.sh python3 scripts/test_llm.py` — 로드 시간, VRAM, 렌더된 프롬프트 tail, tok/s, 그라운딩 체크(정상범위 -60~-80 + 출처 인용)

### 📋 이후: Step 6 Prompt Builder → 7 Chat Pipeline → 8 FastAPI
- BM25 인덱싱
- 키워드 기반 검색

**Step 4: Hybrid & Reranker** (app/rag/hybrid_retriever.py, app/rag/reranker.py)
- RRF (Reciprocal Rank Fusion)
- BGE-Reranker-v2-m3 재정렬

---

## 개발 규칙

### 코드 스타일
- Type hints 필수
- Pydantic 모델 사용
- 로깅 적용
- 예외처리 필수

### 테스트
- 각 Step마다 테스트 스크립트 제공
- pytest 단위 테스트도 함께 작성

### 문서
- 함수 docstring 필수
- 복잡한 로직은 주석 추가
- DEVELOPMENT.md에 진행 상황 기록

---

## 아키텍처 참고

```
User Input
    ↓
Query Processor (Step 1에서 아직 미구현)
    ↓
┌─────────────────────────────────┐
│  Dense Search (Step 2)          │  ← Embedding Client 사용
│  + Sparse Search (Step 3)       │  ← BM25 사용
└──────────┬──────────────────────┘
           ↓
    RRF Fusion (Step 4)
           ↓
    Reranker (Step 4)
           ↓
    Top 4 Documents
           ↓
    LLM Generator (Step 5 이후)
```
