# Face Blur Service

AI 기반 얼굴 블러 서비스 - 영상에서 본인 얼굴을 제외한 나머지 얼굴을 자동으로 블러 처리합니다.

## 주요 기능

- **자동 얼굴 탐지**: YOLOv8 기반 실시간 얼굴 탐지
- **얼굴 인식**: ArcFace 기반 얼굴 임베딩으로 동일 인물 클러스터링
- **선택적 블러**: 본인 사진을 업로드하면 자동으로 제외
- **다양한 블러 옵션**: Gaussian, Mosaic, Blackout
- **프라이버시 보호**: 24시간 후 자동 삭제, 서버에 임베딩 미저장

## 기술 스택

### Backend
- FastAPI (Python 3.11+)
- PyTorch + Ultralytics (YOLOv8)
- InsightFace (ArcFace)
- OpenCV

### Frontend
- React 18 + TypeScript
- Vite
- Tailwind CSS
- Zustand (상태 관리)

## 빠른 시작

### Docker로 실행

```bash
# 프로젝트 클론
git clone <repository-url>
cd blur_ai

# Docker Compose로 실행
docker-compose up -d

# 접속
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

### 로컬 개발 환경

#### Backend

```bash
cd backend

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

## 사용 방법

1. **영상 업로드**: MP4/MOV 파일을 드래그앤드롭 (최대 500MB, 10분)
2. **본인 사진 업로드**: 1-5장의 본인 사진 업로드
3. **분석 대기**: AI가 영상을 분석하고 얼굴을 탐지
4. **얼굴 선택**: 블러 처리할 얼굴 선택 (본인은 자동 제외)
5. **블러 처리**: 블러 스타일/강도 설정 후 처리
6. **다운로드**: 완료된 영상 다운로드

## API 문서

서버 실행 후 `http://localhost:8000/docs`에서 Swagger UI로 API 문서 확인

### 주요 엔드포인트

- `POST /api/v1/upload/video` - 영상 업로드
- `POST /api/v1/upload/reference` - 참조 이미지 업로드
- `POST /api/v1/analyze` - 영상 분석 시작
- `GET /api/v1/analyze/{id}/result` - 분석 결과 조회
- `POST /api/v1/process/blur` - 블러 처리 시작
- `GET /api/v1/process/{id}/download` - 처리된 영상 다운로드

## 프로젝트 구조

```
blur_ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/     # API 엔드포인트
│   │   ├── core/           # 보안, 설정
│   │   ├── models/         # Pydantic 스키마
│   │   └── services/       # AI 서비스 로직
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/     # React 컴포넌트
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # API 클라이언트
│   │   └── types/          # TypeScript 타입
│   ├── package.json
│   └── Dockerfile
├── models/                 # AI 모델 파일
├── docker-compose.yml
├── PRD.md
└── TRD.md
```

## 환경 변수

Backend `.env`:
```
DEBUG=false
UPLOAD_DIR=uploads
MODEL_DIR=models
OUTPUT_DIR=outputs
MAX_VIDEO_SIZE_MB=500
RETENTION_HOURS=24
```

## 라이선스

MIT License
