# Depalletizer Chatbot

완전 로컬 환경 기반 디팔레타이징 시스템 기술 지원 챗봇.

## 개요

- **언어**: Python 3.10+
- **LLM**: Qwen3.5-9B (4-bit 양자화)
- **임베딩**: BAAI/bge-m3
- **리랭커**: BAAI/bge-reranker-v2-m3
- **벡터DB**: Qdrant
- **백엔드**: FastAPI
- **GPU 필요**: 16GB VRAM

## 프로젝트 구조

```
depalletizer_chatbot/
├── app/                          # 애플리케이션 코드
│   ├── schemas/                  # Pydantic 모델
│   ├── api/                      # FastAPI 라우터
│   ├── llm/                      # LLM 클라이언트
│   ├── rag/                      # RAG 엔진
│   ├── router/                   # 의도 라우팅
│   ├── tools/                    # 외부 도구 인터페이스
│   ├── config.py                 # 설정
│   └── main.py                   # FastAPI 앱
├── corpus/                       # 문서 및 코퍼스
├── configs/                      # YAML 설정 파일
├── scripts/                      # 유틸리티 스크립트
├── evaluation/                   # 평가 도구
├── tests/                        # 유닛 테스트
├── requirements.txt              # 의존성
├── .env.example                  # 환경변수 템플릿
└── README.md                     # 이 파일
```

## 빠른 시작

### 1. 환경 설정

```bash
cd /home/dr/depalletizer_chatbot

# 가상환경 생성
python3.10 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
```

### 2. Qdrant 실행

```bash
# Docker로 Qdrant 실행 (권장)
docker run -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 또는 이미 Docker Compose가 있다면:
docker compose up qdrant
```

### 3. LLM 모델 준비

```bash
# Qwen3.5-9B 모델 다운로드 (HuggingFace)
# llama.cpp 또는 Ollama를 통해 실행할 준비
# Phase 2에서 상세 구현 예정
```

### 4. FastAPI 실행

```bash
# 개발 서버
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는
python app/main.py
```

## API 사용 예제

### Health Check

```bash
curl http://localhost:8000/api/health
```

응답:
```json
{
  "status": "ok",
  "service": "depalletizer_chatbot"
}
```

### Chat (Phase 1 Placeholder)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_001",
    "message": "VAC-203이 뭐야?",
    "language": "ko"
  }'
```

응답:
```json
{
  "session_id": "test_session_001",
  "message": "VAC-203이 뭐야?",
  "answer": "[PLACEHOLDER] Received: VAC-203이 뭐야?",
  "intent": "general",
  "sources": [],
  "tools_used": [],
  "retrieval": {
    "documents_found": 0,
    "reranked": false,
    "top_k": 0,
    "sources": []
  },
  "latency_ms": 1.234
}
```

## 개발 로드맵

### Phase 1: Offline RAG MVP ✅ (현재)
- [x] 프로젝트 Skeleton
- [x] Config 시스템
- [x] Pydantic 스키마
- [x] FastAPI 기본 구조
- [ ] Embedding 로드
- [ ] Qdrant 연결
- [ ] Dense/Sparse 검색
- [ ] Reranker
- [ ] Local LLM
- [ ] 통합 파이프라인

### Phase 2: Corpus Pipeline
- 문서 파싱 (PDF, Markdown)
- Semantic Chunking
- 메타데이터 자동 추출
- Index 재구성

### Phase 3: Troubleshooting Assistant
- Intent Router
- Decision Tree Engine
- Multi-turn 대화
- 상태 추적

### Phase 4: Machine Tools
- Alarm Tool
- PLC I/O
- Robot State
- Vision Detection
- Sensor Reading
- Maintenance History

### Phase 5: Evaluation
- 자동 평가 스크립트
- 성능 지표 (Recall, MRR 등)
- 회귀 테스트

### Phase 6: Optimization
- Fine-tuning (LoRA)
- Synthetic QA Dataset
- Domain SFT

## 설정 파일

### `configs/app.yaml`
- 앱 설정
- LLM/임베딩 모델
- Retrieval 하이퍼파라미터
- 데이터베이스 설정

### `.env`
- 환경 변수
- 모델 경로
- API 키 (필요 시)

## 로깅

로그는 `logs/chatbot.log`에 저장됩니다.

로그 레벨은 환경변수로 제어:
```bash
LOG_LEVEL=DEBUG uvicorn app.main:app
```

## 테스트

```bash
# 유닛 테스트 실행
pytest tests/ -v

# 카버리지 확인
pytest tests/ --cov=app
```

## 성능 목표 (Phase 1)

```
응답 시간: < 5초
VRAM 사용: < 15GB
Recall@5: > 0.8
```

## 주요 설계 원칙

1. **정확성 우선** - 추측하지 말고 근거를 검색할 것
2. **분리된 역할** - RAG = 지식, Tools = 상태, LLM = 판단
3. **검색 우선** - LLM 메모리에 의존하지 말 것
4. **출처 명시** - 모든 답변은 출처를 포함할 것
5. **실패 처리** - 정보 부족 시 명시적으로 표현할 것

## 문제 해결

### Qdrant 연결 안 됨
```bash
# Qdrant 상태 확인
curl http://localhost:6333/health

# 다시 시작
docker restart qdrant
```

### CUDA 메모리 부족
```bash
# GPU 사용 비활성화
CUDA_VISIBLE_DEVICES= python app/main.py
```

### 모델 로드 실패
- HuggingFace 캐시 확인
- 네트워크 연결 확인
- 디스크 용량 확인

## 참고 문서

- [Qwen 문서](https://github.com/QwenLM/Qwen)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Qdrant 문서](https://qdrant.tech/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

## 라이선스

프로젝트 정의 필요

## 연락처

개발팀: TBD
