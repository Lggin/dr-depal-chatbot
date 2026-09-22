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

### ✅ Step 5: Local LLM Client (COMPLETE, 2026-09-22)

**실측 (`scripts/test_llm.py`):**
```
LLM loaded in 1.3s: Qwen3.5-9B arch=qwen35 n_ctx=8192 gpu_layers=-1 (all on GPU)
VRAM: 15 → 5,931 MiB (생성 중 5,983 MiB)          ← 16GB 중 ~10GB 여유
prompt 170 + completion 69 tokens in 969ms → 71.2 tok/s
rendered tail: <|im_start|>assistant\n<think>\n</think>\n   ← thinking OFF 정상 (빈 think 블록)
answer: "VAC-203 은 진공압이 정상 범위에 도달하지 못하는 알람입니다. 정상 진공압 범위는 -60 ~ -80 kPa 입니다.
         출처: [SOURCE 1] Vacuum System Manual, Section: 6.3, Alarm: VAC-203"
grounding check ✅
```
- 소스 빌드 바이너리(nvcc 12.6)가 CUDA 12.4 드라이버에서 정상 동작 — minor-version 호환 우려 해소
- **VRAM 여유 ~10GB** → Step 7 통합 시 임베딩(~1.1GB fp16)+리랭커(~1.1GB fp16)를 GPU에 올려도 될 가능성. 단, 현재 `aux_gpu_min_vram_gb=20` 규칙은 Qwen3-14B(≈10GB) 기준이었으므로 **세 모델 동시 상주 상태에서 실측 후** 조정(지금은 유지)
- 커밋: Step 5

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

### ✅ Step 6: Prompt Builder (COMPLETE, 2026-09-22)

**파일:** `app/llm/prompt.py` — `SYSTEM_PROMPT`(명세 §15 10규칙 + 검증된 추가 4규칙), `format_source()`/`trust_label()`(§13 `[SOURCE n]` 블록, 비공식 출처는 "not an official manual" 명시 §21), `ContextBuilder`(토큰 예산 3000·최대 5문서, 파이프라인 순서 보존, 예산 초과 문서는 건너뛰고 첫 문서만 초과 시 절단), `PromptBuilder`(system → 직전 1턴 history → `[MACHINE STATE]`/`[TROUBLESHOOTING STATE]`(옵션)/`[DOCUMENTS]`/`[QUESTION]`). 문서 0건이면 명시 공지 삽입. 테스트 8개 → **53 passed**

**E2E (`scripts/test_prompt.py [질의...]`)** — 검색→프롬프트→LLM, 세 모델 동시 상주 VRAM **6.0GB**(aux CPU), 질의당 총 3~5s:

| 질의 | 결과 |
|---|---|
| `VAC203이 뭐야?` | 정확, `출처: Vacuum System Manual §6.3` ✅ |
| `빨판이 박스를 못 잡아. 뭘 확인해야 해?` | 4단계 점검 순서 그대로, `출처: Troubleshooting Guide §4.1` ✅ (gap 컷으로 1문서만 전달) |
| `로봇 컨트롤러 IP 주소가 뭐야?` (범위 밖) | "현재 제공된 문서에서는 … 확인할 수 없습니다" + 없는 정보 명시 ✅ |
| `VAC-203 지난번에도 있었어?` | 1차: "과거 유지보수 이력" 라벨은 정확했으나 **유지보수 로그 조치에 엔지니어 노트의 "3축 케이블 덕트" 세부를 혼합**, Section 없는 출처에 본문을 §로 오용 → **rule 14(출처 간 사실 혼합 금지)·rule 6(헤더 그대로 인용, Section 없으면 Document만) 추가** → 2차: 로그 사실만 사용, `출처: Maintenance Log` ✅ |

**추가 규칙 근거(이전 프로젝트 실측):** 11 직접·짧게 먼저(문서별 나열 방지), 12 유사기호 원문대조(l/1), 13 영어 음차 금지("서셉션 컵" 실제 발생 → 수정), 14 문서 간 혼합 금지(rule9 재현). 범위 밖 거절 문구는 명세 §30-5 그대로 고정.

**rule 14 안정화 과정:** 추상 지시만으로는 불안정(2차 수정 후 재실행에서 다시 혼합) → **구체적 WRONG/RIGHT worked example** 추가 후 동일 질의 3회 반복 **3/3 혼합 없음**(1회는 엔지니어 메모를 "참고 사항" 절로 분리 표기 — 의도한 패턴). ⚠️ 예시는 처음에 샘플 코퍼스 내용을 그대로 썼다가 **코퍼스에 없는 가상 사례(CAM-410 케이블)**로 교체 — 평가 데이터가 프롬프트에 새는 트레이드오프 회피. 잔여: Section 없는 문서에 본문 첫 줄을 §로 붙이는 경향(3회 중 2회) → Step 7에서 출처를 검색 메타데이터로 구조 생성해 LLM 텍스트 의존 제거 예정.

### ✅ Step 7: Chat Pipeline + Answer Validator (COMPLETE, 2026-09-22)

**파일:**
- `app/safety/command_guard.py` — §22 위험 명령(robot start/reset, servo on, PLC reset, motor move, e-stop 해제, 재시작) 한/영 명령형 감지, 부정문("리셋하지 마세요") 제외. 감지 시 답변 끝에 `DANGER_NOTICE` 부착(V1은 차단 대신 경고)
- `app/safety/answer_validator.py` — §23 결정론적 검사: ①답변 수치가 컨텍스트·질문·장비상태·SOURCE 헤더(section/alarm)에 존재 ②알람코드 존재 ③인용 문서명이 제공 문서 중 하나 ④고정 거절문구 감지. 플래그만(차단 없음). 1~9 정수(목록 번호)는 무시, 인용 줄은 검사에서 제외
- `app/safety/source_validator.py` — **`sources`는 검색 메타데이터로 생성**(LLM 텍스트 아님). LLM의 `출처:` 줄은 "어느 문서를 썼는지" 선택에만 사용 후 **정규 인용줄로 교체** → Step 6 잔여 문제(§에 본문 첫 줄 붙임) 구조적으로 해소. 거절이면 sources 비움
- `app/memory/session.py` — `SessionStore`(인메모리, TTL 30분, history 1턴), 스레드 안전
- `app/llm/generator.py` — `ChatPipeline.answer(session_id, question, filters, machine_state) → ChatResponse`. **`contextualize()`**: 질의에 알람코드가 없고 지시어("그 알람", "해당 원인", "아까 그거")가 있으면 직전 턴의 알람코드를 검색 질의와 프롬프트 질문에 결정론적으로 이어붙임(LLM 미사용). `close()`
- `app/dependencies.py` — `build_chat_pipeline()`: 설정에서 3모델+검색스택 전체 배선(스크립트·FastAPI 공용), 7.3s
- `app/rag/hybrid_retriever.py` — **`pin_min_trust=85`**: exact 히트 중 공식 문서만 상단 고정, 유지보수 로그·엔지니어 노트는 후보 풀에 넣어 리랭커와 경쟁
- 스키마: `ChatResponse.validation/alarm_codes/timing_ms`, `DocumentSource.document_type`
- 테스트 25개 추가(validator 10, guard 2, sources 4, pipeline 8, hybrid 3 갱신) → **78 passed**

**E2E (`scripts/test_chat.py`)** — 5문항 + 2턴 세션, 전부 `validation.ok=true`, 질의당 2.6~4.5s(검색 1.4~1.5s + 생성 1.1~3.0s):

| 질의 | 결과 |
|---|---|
| `VAC203이 뭐야?` | 매뉴얼만 사용, `출처: Vacuum System Manual §6.3 (VAC-203)` — 이전엔 로그·노트가 함께 고정되어 혼합 유발 |
| `진공압 정상 범위 알려줘` | -60~-80 kPa + 사양 §3.2, 이력은 "과거 유지보수 이력" 라벨로 분리 ✅ (VAC-201 §6.1은 실제 미사용인데 인용 — 과잉 인용, 경미) |
| `빨판이 박스를 못 잡아…` | 4단계 순서 ✅ |
| `VAC-203 지난번에도 있었어?` | 로그 사실만, `출처: Maintenance Log (VAC-203)` ✅ |
| `로봇 컨트롤러 IP 주소가 뭐야?` | 거절 문구 + **`sources: []`** ✅ |
| 세션: `VAC-203이 뭐야?` → **`그 알람 원인이 뭐야?`** | 1차 시도에선 "알람 코드가 명시되지 않음"으로 실패 → `contextualize()` 도입 후 `codes=['VAC-203'] (carried)`, 원인 3가지 정확 ✅ |

**이 단계에서 잡은 것:** ①검증기가 `§6.3`·`3.2` 섹션 번호를 근거 없는 수치로 오탐(+`0.5`→`.5` 정규화 버그) — 단위테스트가 먼저 잡음 ②거절에도 sources 4건 노출 ③지시대명사 후속질문 실패 ④정의 질문에 비공식 exact 히트 강제 주입. 잔여: 미사용 문서 과잉 인용(검증기는 "인용"과 "사용"을 구분 못 함), 원인 항목에 원문에 없는 짧은 설명 덧붙임("공기가 새어 들어와 압력이 낮아짐") — 수치가 아니라 검증기 미포착, Phase 5 평가셋에서 groundedness로 측정 예정.

### ✅ Step 8: FastAPI + 로깅 (COMPLETE, 2026-09-22)

**파일:**
- `app/main.py` — lifespan에서 DB 초기화 + `build_chat_pipeline()` 1회(7.4s), 종료 시 `close()`. `SKIP_MODEL_LOAD=1`이면 모델 없이 기동(테스트)
- `app/api/chat.py` — `POST /api/chat`: 스레드풀 실행 + **`threading.Lock`으로 생성 직렬화**(llama.cpp 컨텍스트는 단일). 응답 후 로그 기록 → `log_id`. `POST /api/feedback`(`up|down`, note)
- `app/api/session.py` — `POST /api/session`(201) / `GET` / `DELETE`(204)
- `app/api/health.py` — `/ready`: llm·retrieval·qdrant·database 각각 확인, 하나라도 false면 503
- `app/api/admin.py` — `GET /admin/documents`(문서별·타입별 청크 수), `GET /admin/logs`, `ingest`/`reindex`는 Phase 2까지 **501**
- `app/database/` — SQLAlchemy 2.0, `ChatLog`(§25: timestamp·session·question·normalized_query·intent·alarm_codes·retrieved_chunk_ids·reranker_scores·sources·answer·validation·timing·latency·user_feedback). SQLite `data/chatbot.db`; `:memory:`는 StaticPool
- 스키마: `ChatResponse.retrieval_query/log_id`, `RetrievalInfo.scores`, `SessionInfo`, `FeedbackRequest`
- 테스트 8개(`tests/test_api.py`, 가짜 파이프라인 주입·인메모리 DB) → **86 passed**

**실서버 검증 (uvicorn + curl):** `/ready` 200 → 세션 생성 → `VAC-203이 뭐야?` 3.5s(`log_id=1`) → 후속 `그 알람 원인이 뭐야?` → `retrieval_query="… VAC-203"`, `alarm_codes=['VAC-203']` → 피드백 `up` → `/admin/documents` 12청크/9문서 → `/admin/logs`에 청크 id·피드백 반영 → 세션 삭제 204

**잡은 함정:** `httpx 0.28`이 `Client(app=)`를 제거해 Starlette 0.35 TestClient가 깨짐 → `httpx==0.27.2` 핀(qdrant-client 요구범위 내)

### ✅ Step 9: Evaluation (COMPLETE, 2026-09-22) — Phase 1 종료

**파일:**
- `evaluation/datasets/test_questions.json` — §26 스키마 **37문항**, 9카테고리(alarm_definition 7 · normal_value 7 · procedure 4 · colloquial 4 · symptom_to_alarm 3 · history 3 · faq 2 · out_of_scope 4 · follow_up 3). `history` 필드로 후속질문, `expect_refusal`로 범위 밖
- `evaluation/metrics.py` — Recall@k / Hit@k / MRR / nDCG (binary relevance, 정답 없는 케이스는 NaN 제외)
- `evaluation/retrieval_eval.py` — LLM 없이 검색만(follow-up은 `contextualize` 적용). fused(리랭커 전)·final(리랭커 후) 두 단계 + 알람 파싱 정확도. **~30s**
- `evaluation/answer_eval.py` — ChatPipeline E2E: must_include/must_not, 거절 정확도(recall·false rate), 인용 적중(expected chunk ∈ sources), unsupported claim rate, intent, 지연. **~3분**
- `evaluation/report.py` — `results/<ts>_<mode>.json` + `latest_<mode>.json`(커밋됨), Markdown 표(이전 대비 Δ), **회귀 게이트**(retrieval: hit@5/recall@5 하락 0 허용, mrr 0.02; full: answer_correct/citation/refusal_recall 하락 0 허용) → 종료코드 1
- `scripts/run_eval.py --mode retrieval|full [--show-failures] [--no-gate]`
- 테스트 6개 → **95 passed**

**Phase 1 최종 지표 (샘플 코퍼스 12청크 / 37문항):**

| 검색 (33건 채점) | fused | final(rerank) |
|---|---|---|
| Hit@1 | 0.879 | **0.939** |
| Hit@3 / Recall@3 | 0.970 | **1.000** |
| Hit@5 / Recall@5 | 1.000 | **1.000** |
| MRR | 0.925 | **0.970** |
| nDCG@5 | — | 0.978 |
| 알람 파싱 정확도 | 1.000 | |

| 답변 (37건) | 1차 | **최종** |
|---|---|---|
| answer_correct | 0.892 | **1.000** |
| citation_accuracy | 0.892 | **1.000** |
| refusal_recall (범위 밖 4건) | 1.000 | **1.000** |
| false_refusal_rate | 0.121 | **0.000** |
| unsupported_claim_rate | 0.091 | **0.000** |
| danger_flag_rate | 0 | 0 |
| intent_accuracy | 0.541 | 0.541 (라우터 없음 — Phase 3 베이스라인) |
| latency median / p95 | 3.2s / 4.5s | **2.9s / 4.6s** |

**1차 → 최종 사이에 잡은 것 (평가가 아니었으면 못 봤을 것들):**
1. **거짓 거절 4건** — 사용자가 증상을 말하면("카메라가 응답이 없어") 답은 정확한데 rule 7 때문에 "현재 제공된 문서에서는 실제 상태를 확인할 수 없으므로…"를 덧붙이고, 검증기가 그 문구로 거절 판정 → sources 비움. **프롬프트 rule 16**(사용자 보고 증상은 전제로) + **검증기: 문구가 첫 문장에 있을 때만 거절**
2. **프롬프트 예시 누출** — rule 14의 가상 사례(CAM-410 케이블)를 Q041 답변이 "과거 유지보수 이력"으로 인용. 구체 예시는 혼합을 막지만 9B 모델은 그것을 문서로 오인 → `부품A`/`이유B` 한국어 플레이스홀더 예시로 교체(구조는 구체, 인용은 불가). 이후 누출 0, 혼합 0
3. **검증기 오탐 2종** — `(기본 0.5)`처럼 `)` 뒤 숫자, `v2.3`처럼 문자 뒤 숫자를 근거에서 놓침 → 정규식 수정
4. **한자 누출** — "정상 가동을恢复了습니다" 1건 → `foreign_script` 플래그(결정론적) + rule 17. temperature 0.2→**0.1**(명세 하한)로 실행 간 변동 축소

**평가셋 한계(정직하게):** 12청크 샘플 코퍼스 기준이라 37/37은 "파이프라인이 설계대로 동작한다"는 확인이지 실코퍼스 성능이 아님. 명세 §26의 300문항은 Phase 2에서 실문서가 들어온 뒤 의미가 있음. 또 must_include 토큰이 느슨해(예: "진공") 답변 품질 상한을 재지 못함 — Phase 5에서 LLM judge 또는 사람 채점 필요.

**Phase 1 완료.** 명세 §32 목표(질문 → Query Processor → Hybrid RAG → Reranker → Local LLM → 출처 포함 답변)와 §28 Phase 1 예시 질문 4종("VAC-203이 뭐야?", "진공압 정상 범위", "캘리브레이션 방법", "박스 흡착 실패 원인") 전부 정답.

---

## Phase 2: Corpus Pipeline (2026-09-22 ~)

**결정:** 실코퍼스 = `dr-depal-knowhow`(자동 갱신 저장소, `/knowhow-update`). 574 V2 JSON을 1회 변환하는 대신 **저장소 마크다운을 직접 읽는 파이프라인**을 만들어 갱신 시 `scripts/ingest.py` 1회로 재색인. 574 V2는 `--kind v2json`으로 별도 컬렉션(`knowhow_v2`)에 넣어 새 청커와 A/B 비교.

### ✅ 2.1 소스 선언 + 파서 + 청커 + 메타데이터 + 파이프라인

- `configs/corpus.yaml` — 29소스(knowhow docs 14 · depal-al README 11 · 프로젝트 노트 4). 소스별 document_type/trust/system/tags. `alarm_codes: false`(이 코퍼스엔 VAC-203류 코드 없음 → `sm_89`, `cu121` 같은 오탐 방지). 노트 4개는 `corpus/raw/notes/`로 복사(프로젝트 소유)
- `ingestion/markdown_parser.py` — `#`=문서 제목, `##`/`###`→섹션(heading path 보존), 코드펜스 안 `#` 무시, 첫 `##` 앞 텍스트=`도입부`
- `ingestion/chunker.py` — 토큰 기준(임베딩 토크나이저 주입). 단위=섹션. **min 60 미만은 병합**: 그룹 내 어느 섹션의 자식·형제면 흡수(`도입부 → ## 서문 → ### 첫 자식` 체인이 한 청크로). **max 800 초과는 블록 분할**(문단 경계, 코드펜스·표는 원자 보존, overlap 80, `(i/n)`). chunk_id = kebab(source + heading path)[-pN], 결정론적. `search_text` = `문서 > h2 > h3\n\n본문`(임베딩·BM25 대상), `text` = 본문(LLM 컨텍스트)
- `ingestion/metadata_generator.py` — 소스 설정 + 규칙 추출: components(키워드 33개 → 시스템 추론), 버전 `v0.0.NN`·이슈 `#NNN` → tags, 최신 날짜 → as_of_date, 수치 사양 여부. LLM 미사용
- `ingestion/v2_import.py` — 574 V2 → 새 스키마(구 document_type 6종 → 신 7종 매핑, 소스별 trust 재부여)
- `ingestion/pipeline.py` — yaml → 청크 JSON + **manifest**(토큰 분포·타입/시스템/trust 분포·소스별 청크/분할/병합 수) → 임베딩(search_text) → Qdrant(컬렉션 재생성). 소스 라벨은 V2·held-out과 동일(`docs/03-…md`, `depal-al/README.md`, `notes/…`)
- `scripts/ingest.py --dry-run | [--kind v2json --collection X] [--device]` — 인제스트는 LLM 미상주라 **GPU 기본**(CPU는 16청크/분으로 너무 느렸음)
- `app/rag/sparse_retriever.py` — BM25가 `search_text`를 색인(섹션 제목 매칭)
- 테스트 6개 → **101 passed**

**dry-run 실측 (knowhow_docs):** 621청크 / 29소스, 토큰 **p50 264 · p90 619 · max 792 · 초과 0**, 분할 70(03-grasp 12, 10-open-issues 16, user_notes 13), 병합 48. V2(574)보다 많은 건 단위가 더 작기 때문(sanity: 526 원본 섹션 → 병합/분할 후 621)

### ✅ 2.2 평가셋 변환 + 결과 분리

- `evaluation/build_knowhow_dataset.py` — 이전 프로젝트 held-out 40문항(dev 25 · operator 15)을 §26 스키마로. `source: "docs/x.md > h2 — h3"`를 라벨+제목 조각 매칭(`>` 분리 후 ` — `로 재분리)으로 청크 id에 해소 → **39/40 매핑**(1건은 표 제목 → `expected_source: null`=unknown). must_include는 정답문에서 버전/따옴표 용어/단위 수치 자동 시드(17건) — 수작업 보강 대상
- 스키마: `expected_source: null`(정답 위치 미상, 지표 제외) vs `[]`(범위 밖) 구분
- `evaluation/report.py` — 결과 파일을 `(mode, dataset)`별 `latest_<mode>_<tag>.json`으로 분리(샘플 코퍼스 결과가 knowhow 결과에 덮어써지거나 게이트가 다른 데이터셋을 비교하던 문제). `run_eval.py --collection --dataset`

### ✅ 2.3 `/api/admin/reindex|ingest` 실구현
- `ChatPipeline.reindex(config_path)` — 서빙 중인 컬렉션으로 재빌드·재색인 + **BM25 재구성**(서버 재시작 불필요). 생성 락 안에서 실행(재색인 중 생성 차단). `ingest`는 소스가 선언적(yaml)이라 reindex와 동일

### ✅ 2.4 색인 + 검색 A/B (2026-09-22)

**색인 함정 2건:** ① CPU 임베딩 16청크/분 → 인제스트는 GPU 기본 ② V2 임포트 **CUDA OOM이 두 번**(배치 64: 36GB 요청, 배치 16: 9.4GB 요청 — 파일 끝에 몰린 2k토큰 청크) → 마지막 30개가 조용히 누락(544/574)됐고 `grep|tail` 필터가 트레이스백·종료코드를 숨김. 수정: **토큰 예산 적응형 배치**(`token_batches`: 개수 16 AND 합계 6000토큰) + **색인 후 정확 카운트 검증**(`store.count()` 불일치 시 예외). 재실행 574/574 ✅. 교훈: 배치는 "개수"가 아니라 "합계 토큰"으로 잡고, 색인은 반드시 카운트로 검증

**검색 A/B — 같은 40문항(각 컬렉션 id로 해소), 39건 채점:**

| | knowhow_docs (새 청커 621) | knowhow_v2 (레거시 574) |
|---|---|---|
| final Hit@1 | **0.821** | 0.769 |
| final Hit@3 | **0.974** | 0.949 |
| final Hit@5 | **0.974** | 0.949 |
| final MRR | **0.880** | 0.842 |
| final nDCG@5 | 0.844 | 0.864 |
| fused→final Hit@1 (리랭커 효과) | 0.641→0.821 | — |
| miss@5 | KO012 | KD015, KO012 |

새 청커가 Hit·MRR 전부 우세. Recall@5는 V2가 높게(0.929 vs 0.890) 보이지만 V2 청크가 커서 기대 집합이 작아진 효과. KO012("라벨링한 데이터가 학습에 반영되려면 다음에 뭘 해?" → §저장 위치·회수)는 양쪽 공통 미스 — fused top-10에도 없음, 어휘 격차("반영"↔"회수"). glossary 후보(실패 재현됨).

⚠️ 첫 A/B 시도는 expected id를 새 청커 형식으로만 해소해 V2가 Hit@5 0.256으로 나왔음 — 같은 문서를 1등으로 가져와도 id 불일치로 미스 처리. **컬렉션별로 평가셋을 따로 해소**(`build_knowhow_dataset.py --chunks --out`)해야 공정

### ✅ 2.5 knowhow full 평가 — Phase 2 종료 (2026-09-22)

**aux 모델 GPU 이동 효과:** knowhow 청크(p90 619tok)에서 CPU 리랭커가 질의당 ~15s → GPU에서 **중위 2.8s / p95 5.0s**. 3모델 동시 상주 VRAM ~8.5GB(16GB 중). `aux_gpu_min_vram_gb` 20→12

**40문항(dev 25 · operator 15) full 평가 — 3회 반복:**

| | 1차 | 2차 (rule 11 완전성 + 인용 파서 수정) | **3차 (must_include 6건 교정)** |
|---|---|---|---|
| answer_correct | 0.750 | 0.875 | **0.975** |
| citation_accuracy | 0.949 | 0.974 | **0.974** |
| unsupported_claim_rate | 0.350 | 0.025 | **0.025** |
| false_refusal_rate | 0 | 0 | **0** |
| dev / operator | 0.64 / 0.93 | 0.80 / 1.00 | **0.96 / 1.00** |

- **1차→2차:** ①`unsupported 0.35`는 전부 **인용 파서 버그** — `출처:` 줄을 `,`로도 분리해 "(2026-08-21, v0.0.59 #470)"류 괄호·쉼표가 든 knowhow 문서명·섹션이 `v0.0.46 → v0.0.57)`, `SOURCE 3` 조각으로 잘려 "미제공 문서 인용"으로 오판. `;`만 분리, `[SOURCE n]`·꼬리 괄호 제거 → 0.025. ②**내용 누락 5건**(KD006 히트 1.74cm만 말하고 미스 9.90/47.9cm 생략, KD012 leaf 239 생략 등) → rule 11에 "질문 항목의 원문 수치는 전부 포함, 비교 질문은 양쪽 수치" 추가 → KD006이 양쪽 수치를 다 답함
- **2차→3차:** 남은 5건은 전부 **자동 시드 must_include의 오류**(질문이 묻지 않은 이력 버전·부수 설명·배너 문구를 요구) → 질문 요지 토큰으로 수작업 교정(`auto_generated: false`, 교정 사유 기록). 모델 답변은 5건 모두 정확·완전했음
- **남은 1건 KD017**(테스트 수 8월 추이): 9개 데이터 포인트 중 마지막 2개만 답하고 `2,688−2,634=54`를 **계산해 제시**(원문에 없는 파생 수치 → 검증기가 잡음, unsupported 0.025의 정체). rule 2 회색지대 — 파생 계산 허용 여부는 Phase 5에서 결정
- 평가 heuristic 한계: must_include는 공백·마크다운 무시 비교(`2,000 줄`↔`2,000줄`)로 보정했으나 여전히 "정답 판정"이 아니라 "핵심 토큰 포함 여부". LLM judge는 Phase 5

**Phase 2 완료.** 실코퍼스(621청크) 기준 검색 Hit@1 0.821 / Hit@3 0.974 / MRR 0.880, 답변 정확도 0.975, 재색인 `scripts/ingest.py` 1회 또는 `POST /api/admin/reindex`.

---

## Phase 3: Intent Router · Troubleshooting Engine · Multi-turn (2026-09-22 ~)

### ✅ 3.1 Intent Router
- `app/router/intent_router.py` — 규칙 기반, LLM 미사용. 신호: `_NOW`(지금·현재), `_REFERENCE`(정상 범위·기준·정상인지), `_WHY_NOW`, `_HISTORY`, `_TROUBLE`(증상·점검 어휘), `_KNOWLEDGE`, `_SMALLTALK`, `_MACHINE_NOUN`. 우선순위 **LIVE_DIAGNOSIS > LIVE_STATUS > HISTORY > TROUBLESHOOTING > KNOWLEDGE**, GENERAL=명시적 잡담만(30자 미만·장비 명사 없음), 미매칭=KNOWLEDGE(confidence 0.5 — 코퍼스가 답할 수 있는지는 검색이 판단)
- 핵심 분기: "지금 진공압 **얼마야**"→LIVE_STATUS / "진공압 **정상 범위**가 얼마야"→KNOWLEDGE / "지금 진공압이 **정상인지** 확인해줘"→LIVE_DIAGNOSIS(now+reference+machine, 명세 §17)
- 34문항 규칙 테스트(명세 §9·§17·§18 예시 + 샘플 평가셋 기대값) 전부 통과. 잡은 함정: 한글은 전부 `\w`라 **`\b`가 한글 사이에서 절대 안 걸림**("넌 누구야", "나와요") — 이 프로젝트 정규식 3곳에서 반복된 실수

### ✅ 3.2 Troubleshooting Engine (명세 §19–§20)
- `configs/troubleshooting.yaml` — 결정트리 3종: **pickup_failure**(명세 §19 그대로: vision→robot_pose→vacuum→box), **detection_failure**(카메라→실시간모드→조명→임계값; 오퍼레이터 체크리스트 노트 기반), **position_offset**(일정한 어긋남→캘리브레이션 / 불규칙→검출 품질). issue = `triggers`(트리거 최장 일치) + `facts`(yes/no 질문, Phase 4용 `tool` 키) + `order` + `steps`(`when` 조건 전부 충족 시 권고, `doc_query`로 검색) + `escalate`
- `app/troubleshooting/rules.py` — 로드·검증(`when`/`order`가 미정의 fact 참조 시 예외)
- `app/troubleshooting/state.py` — 세션별 `TroubleshootingState`(issue·facts·pending_fact·current_step·resolved)
- `app/troubleshooting/engine.py` — `next()`: 조건 충족 step 있으면 **recommend**, 없으면 `order` 순으로 **아직 관련 있는**(어떤 step이 여전히 필요로 하는) 미확인 fact 하나를 **ask**, 다 알면 **escalate**. `apply_reply()`: yes/no 파싱(`parse_yes_no`), "해결됐어"→resolved, "그만/다른 질문"→cancelled, 모호→unclear(재질문). `is_reply()`: 명시적 yes/no/해결/취소는 40자까지, 그 외 25자 이하 & 새 질문 표지(뭐야·알려·얼마·알람코드) 없음. `prefill_from_tools()`: Phase 4 훅
- vision=false로 답하면 robot/vacuum 질문을 건너뛰고 바로 check_vision 권고(관련 없는 fact는 묻지 않음). 테스트 27건

### ✅ 3.3 ChatPipeline 라우팅 (`app/llm/generator.py` 재작성)
| intent | 동작 | LLM | 검색 |
|---|---|---|---|
| GENERAL | 도우미 소개(결정론) | ✗ | ✗ |
| LIVE_STATUS (도구 없음) | "설비 조회 도구 미연결" 정직 응답, `is_refusal=True` | ✗ | ✗ |
| LIVE_STATUS (machine_state 있음) | `[MACHINE STATE]` 블록 포함 생성 (Phase 4) | ✓ | ✓ |
| LIVE_DIAGNOSIS (도구 없음) | 문서 기준값·점검 순서 + 상단 ⚠️ 미연결 안내 | ✓ | ✓ |
| HISTORY | `document_type=maintenance` 필터 우선, 0건이면 무필터 재검색(`history.filtered`/`history.unfiltered`) | ✓ | ✓ |
| TROUBLESHOOTING + 트리 매칭 + 증상 보고 | **ask**: 결정론 질문(0ms) → 답변마다 갱신 → **recommend**: step의 `doc_query`로 검색, `[TROUBLESHOOTING STATE]`와 함께 생성, 끝에 "해결되면 '해결됐어'" → **resolved/escalate**: 결정론 | ask ✗ / recommend ✓ | 동일 |
| TROUBLESHOOTING + "원인이 뭐야?"(설명 요구) 또는 트리 없음 | 일반 RAG(`_ASKS_EXPLANATION`으로 구분 — "박스 흡착 실패 원인이 뭐야?"는 트리를 시작하지 않음) | ✓ | ✓ |
| KNOWLEDGE | Phase 1 경로 | ✓ | ✓ |
- sticky: 진단 중 짧은 답변은 intent와 무관하게 엔진으로; 새 주제 질문이 오면 진단 상태 폐기. `ChatResponse.route`/`troubleshooting` 필드 추가
- E2E(`scripts/test_troubleshooting.py`, 샘플 코퍼스): 라우트 5종 정상, **명세 §20 대화 재현** — "박스를 못 집어." → 비전? "응" → 로봇? "응" → 진공? "아니, -45…" → check_vacuum 권고(호스→패드→발생기, 이미 정상인 항목 재점검 금지 준수, 4문서 근거) → "해결됐어" → 종료

**이 단계에서 잡은 날조:** VIS-301 답변이 코퍼스에 없는 "과거 유지보수 이력"·"현장 엔지니어 메모" 절을 지어냄(공식 매뉴얼 1건만 컨텍스트에 있었음, 수치가 없어 검증기 미포착 — rule 14 예시의 "현장 엔지니어 메모:" 형식을 흉내 낸 것으로 추정). → **rule 18**(해당 Trust의 SOURCE가 있을 때만 언급) + **검증기 `fabricated_source_types`**(답변이 이력/메모를 주장하는데 컨텍스트에 그 타입 문서가 없으면 플래그, `ok=False`). 3회 재실행 날조 0. 잔여 코스메틱(빈 "참고:" 제목)은 `canonicalize_citations`에서 제거

### ✅ 3.4 평가 — Phase 3 종료 (2026-09-22)

샘플 평가셋 갱신: `expected_route` 11건(트리 시작 4건은 `expected_source: null` + 첫 진단 질문 키워드로 채점, LIVE 1건 `live.no_tools`, HISTORY 2건 `history.filtered`), 범위 밖 3건의 기대 intent를 general→**knowledge**로(GENERAL은 잡담 전용, 범위 밖은 검색이 거절을 결정하는 설계). `answer_eval`에 `route_accuracy`·`fabricated_source_types`·`foreign_script` 기록 추가.

| 샘플 37문항 | Phase 1 최종 | Phase 3 1차 | **Phase 3 최종** |
|---|---|---|---|
| answer_correct | 1.000 | 0.973 | **1.000** |
| intent_accuracy | 0.541 (고정 KNOWLEDGE) | 0.892 | **1.000** |
| route_accuracy | — | 0.909 | **1.000** |
| citation_accuracy | 1.000 | 1.000 | **1.000** |
| unsupported_claim_rate | 0.000 | 0.061 | **0.000** |
| false_refusal_rate | 0 | 0 | **0** |
| latency median | 2.9s | 1.7s | **1.4s** |

1차→최종에서 잡은 것: ① Q041 "진공압 표시가 0으로 **고정돼** 있어" → 증상 어휘(고정/표시/먹통/깜빡/끊겨/느려) 라우터 보강 ② Q033 답변이 "원인은 현재 제공된 문서에서는 확인할 수 없습니다"로 **열고 나서** §8.5 근거로 원인 3가지를 다 제시 → 거절로 오분류되어 sources 비움. **rule 16에 WRONG/RIGHT 오프닝 예시**(플레이스홀더) + **거절 판정 = 첫 문장 문구 AND (본문 ≤220자 OR 섹션·문서 참조 없음)** — 진짜 거절(~120자, 인용 없음)과 헤지형 정답(300자+, 인용 있음) 분리 ③ 1차의 unsupported 2건은 재실행에서 재현 안 됨(temp 0.1 변동) — 이후 평가 행에 플래그 종류를 기록해 다음엔 식별 가능

**knowhow 40문항 회귀 게이트:** 1차 실행 **0.950으로 게이트 실패** → 원인 3개 모두 실코퍼스가 드러낸 결함: ① 과거형 어미 "…밀려나 **있었어**?"가 `_HISTORY`에 걸려 KD006·KD016이 HISTORY로 오라우팅 → 이력 힌트 때문에 모델이 빈 "과거 유지보수 이력: … 확인할 수 없습니다" 절을 붙임(금지 문구) — 라우터를 "발생/재발/조치" 표지 기반으로 재작성, 무필터 폴백 시 힌트도 "기록 없으면 한 문장만" ② 검증기 `fabricated_source_types`가 "유지보수 기록은 **없습니다**"(정직한 부정문)까지 잡음 → 문장 단위 부정어 제외 ③ 모델이 문서명에 공백 삽입("2026 년 3 월")해 인용 불일치 → 공백 무시 비교. 재실행 **1.000 / unsupported 0.000, 게이트 통과**. intent 0.950은 knowhow 평가셋의 기대 intent가 전부 자동 "knowledge"라서(오퍼레이터 증상 질문 2건이 troubleshooting으로 분류된 것은 타당)

**Phase 3 완료.** 라우터·트리·세션 상태·라우팅 전부 결정론적 Python(명세 §30-7); LLM은 권고 단계의 근거 설명과 KNOWLEDGE/HISTORY/LIVE_DIAGNOSIS 답변만 담당. Phase 4 훅 준비됨: `facts.tool` → `prefill_from_tools()`, `machine_state` 슬롯(LIVE 라우트가 도구 결과를 받으면 LLM 경로로 전환).

### 📋 다음: Phase 4 Machine Tools — `app/tools/base.py` 인터페이스 + Mock(알람·PLC·로봇·비전·센서·유지보수), `ToolRegistry`, LIVE_STATUS/LIVE_DIAGNOSIS 라우트에 도구 결과 주입, 트리 facts 자동 채움, READ ONLY 강제
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
