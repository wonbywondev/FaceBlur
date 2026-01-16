# 얼굴 블러 서비스 TRD (Technical Requirements Document)

**버전**: 1.0
**작성일**: 2026.01.16
**기반 문서**: PRD.md v1.0
**목적**: 개인화 얼굴 보호 영상 편집 서비스의 기술 설계 및 구현 명세

---

## 1. 시스템 아키텍처

### 1.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Upload Zone │  │ Face Grid   │  │ Video Preview Player    │ │
│  │ (Drag&Drop) │  │ (Checkbox)  │  │ (Canvas Overlay)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Upload API  │  │ Process API │  │ Background Tasks        │ │
│  │ /upload     │  │ /analyze    │  │ (Video Processing)      │ │
│  │ /reference  │  │ /blur       │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI Pipeline (PyTorch)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ YOLOv8-face │  │ ArcFace     │  │ OpenCV                  │ │
│  │ (Detection) │  │ (Embedding) │  │ (Blur Processing)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 디렉토리 구조

```
blur_ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI 앱 진입점
│   │   ├── config.py            # 설정 관리
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── upload.py    # 업로드 API
│   │   │   │   ├── analyze.py   # 분석 API
│   │   │   │   └── process.py   # 블러 처리 API
│   │   │   └── deps.py          # 의존성 주입
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── security.py      # 보안 유틸
│   │   │   └── tasks.py         # Background Tasks
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py       # Pydantic 스키마
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── face_detector.py # YOLOv8 얼굴 탐지
│   │       ├── face_embedder.py # ArcFace 임베딩
│   │       ├── face_matcher.py  # 유사도 매칭
│   │       └── video_processor.py # 영상 처리
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── UploadZone.tsx   # 드래그앤드롭 업로드
│   │   │   ├── FaceGrid.tsx     # 얼굴 목록 체크박스
│   │   │   ├── VideoPreview.tsx # 미리보기 플레이어
│   │   │   └── ProgressBar.tsx  # 진행률 표시
│   │   ├── hooks/
│   │   │   └── useVideoProcessor.ts
│   │   ├── services/
│   │   │   └── api.ts           # API 클라이언트
│   │   └── types/
│   │       └── index.ts
│   ├── package.json
│   └── vite.config.ts
├── models/                       # AI 모델 파일
│   ├── yolov8n-face.pt
│   └── arcface_r100.pth
├── docker-compose.yml
└── README.md
```

---

## 2. 백엔드 상세 설계

### 2.1 기술 스택

| 구성요소 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| Framework | FastAPI | 0.115+ | REST API 서버 |
| Runtime | Python | 3.11+ | 백엔드 런타임 |
| ML Framework | PyTorch | 2.1+ | AI 모델 추론 |
| Face Detection | Ultralytics | 8.1+ | YOLOv8 얼굴 탐지 |
| Face Embedding | InsightFace | 0.7+ | ArcFace 임베딩 |
| Video Processing | OpenCV | 4.9+ | 영상 처리 |
| Task Queue | FastAPI BackgroundTasks | - | 비동기 처리 |
| Validation | Pydantic | 2.0+ | 데이터 검증 |

### 2.2 API 엔드포인트

#### 2.2.1 영상 업로드 API

```python
# POST /api/v1/upload/video
# Request: multipart/form-data
{
    "file": UploadFile  # MP4/MOV, max 500MB
}

# Response: 200 OK
{
    "video_id": "uuid-string",
    "filename": "original_name.mp4",
    "duration": 120.5,  # seconds
    "resolution": {"width": 1920, "height": 1080},
    "status": "uploaded"
}
```

#### 2.2.2 참조 이미지 업로드 API

```python
# POST /api/v1/upload/reference
# Request: multipart/form-data
{
    "video_id": "uuid-string",
    "files": List[UploadFile]  # 1-5장, JPG/PNG
}

# Response: 200 OK
{
    "reference_id": "uuid-string",
    "face_count": 1,
    "embeddings_generated": true
}
```

#### 2.2.3 영상 분석 API

```python
# POST /api/v1/analyze
{
    "video_id": "uuid-string",
    "reference_id": "uuid-string"
}

# Response: 200 OK
{
    "analysis_id": "uuid-string",
    "status": "processing",
    "estimated_time": 60  # seconds
}

# WebSocket /ws/analysis/{analysis_id}
# Progress updates
{
    "progress": 45,  # percentage
    "faces_detected": 12,
    "frames_processed": 1500
}
```

#### 2.2.4 분석 결과 조회 API

```python
# GET /api/v1/analyze/{analysis_id}/result
# Response: 200 OK
{
    "faces": [
        {
            "face_id": "face-001",
            "thumbnail": "base64-encoded-image",
            "first_appearance": 5.2,  # seconds
            "appearances": [
                {"start": 5.2, "end": 8.5},
                {"start": 15.0, "end": 22.3}
            ],
            "similarity_to_reference": 92.5,  # percentage
            "is_reference": true,
            "blur_enabled": false
        },
        {
            "face_id": "face-002",
            "thumbnail": "base64-encoded-image",
            "first_appearance": 12.1,
            "appearances": [...],
            "similarity_to_reference": 15.2,
            "is_reference": false,
            "blur_enabled": true  # 기본값: 본인 외 블러
        }
    ],
    "total_faces": 5,
    "reference_matches": 1
}
```

#### 2.2.5 블러 처리 API

```python
# POST /api/v1/process/blur
{
    "analysis_id": "uuid-string",
    "blur_settings": {
        "type": "gaussian",  # gaussian | mosaic | blackout
        "intensity": 25,     # 1-50
        "face_ids": ["face-002", "face-003"]  # 블러 대상
    }
}

# Response: 202 Accepted
{
    "process_id": "uuid-string",
    "status": "processing"
}

# GET /api/v1/process/{process_id}/download
# Response: video/mp4 stream
```

### 2.3 데이터 모델 (Pydantic Schemas)

```python
# backend/app/models/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from uuid import UUID

class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"

class VideoUploadResponse(BaseModel):
    video_id: UUID
    filename: str
    duration: float
    resolution: dict
    status: str

class FaceAppearance(BaseModel):
    start: float
    end: float

class DetectedFace(BaseModel):
    face_id: str
    thumbnail: str  # base64
    first_appearance: float
    appearances: List[FaceAppearance]
    similarity_to_reference: float = Field(ge=0, le=100)
    is_reference: bool
    blur_enabled: bool

class AnalysisResult(BaseModel):
    faces: List[DetectedFace]
    total_faces: int
    reference_matches: int

class BlurSettings(BaseModel):
    type: BlurType = BlurType.GAUSSIAN
    intensity: int = Field(default=25, ge=1, le=50)
    face_ids: List[str]

class ProcessRequest(BaseModel):
    analysis_id: UUID
    blur_settings: BlurSettings
```

---

## 3. AI 파이프라인 설계

### 3.1 얼굴 탐지 (YOLOv8)

```python
# backend/app/services/face_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple

class FaceDetector:
    def __init__(self, model_path: str = "models/yolov8n-face.pt"):
        self.model = YOLO(model_path)
        # MPS (Apple Silicon) 최적화
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def detect_faces(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """프레임에서 얼굴 탐지"""
        results = self.model(frame, conf=conf_threshold, device=self.device)

        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence
                })
        return faces

    def process_video(self, video_path: str, sample_rate: int = 5):
        """영상 전체 프레임 처리 (sample_rate 프레임마다 탐지)"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        all_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                faces = self.detect_faces(frame)
                for face in faces:
                    face["timestamp"] = timestamp
                    face["frame_number"] = frame_count
                all_detections.append(faces)

            frame_count += 1

        cap.release()
        return all_detections
```

### 3.2 얼굴 임베딩 (ArcFace)

```python
# backend/app/services/face_embedder.py

import insightface
from insightface.app import FaceAnalysis
import numpy as np
from typing import List

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """얼굴 이미지에서 512차원 임베딩 추출"""
        faces = self.app.get(face_image)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def get_embeddings_from_reference(self, images: List[np.ndarray]) -> np.ndarray:
        """참조 이미지들에서 평균 임베딩 계산"""
        embeddings = []
        for img in images:
            emb = self.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("No faces detected in reference images")

        # 평균 임베딩 (더 robust한 매칭을 위해)
        return np.mean(embeddings, axis=0)
```

### 3.3 얼굴 매칭

```python
# backend/app/services/face_matcher.py

import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple

class FaceMatcher:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """코사인 유사도 계산 (0-100%)"""
        similarity = 1 - cosine(emb1, emb2)
        return max(0, min(100, similarity * 100))

    def is_same_person(self, emb1: np.ndarray, emb2: np.ndarray) -> bool:
        """동일 인물 여부 판단"""
        similarity = self.calculate_similarity(emb1, emb2)
        return similarity >= (self.threshold * 100)

    def cluster_faces(self, face_embeddings: List[dict]) -> List[dict]:
        """탐지된 얼굴들을 동일 인물 기준으로 클러스터링"""
        clusters = []

        for face in face_embeddings:
            matched = False
            for cluster in clusters:
                if self.is_same_person(face["embedding"], cluster["representative"]):
                    cluster["appearances"].append({
                        "timestamp": face["timestamp"],
                        "bbox": face["bbox"]
                    })
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "face_id": f"face-{len(clusters):03d}",
                    "representative": face["embedding"],
                    "thumbnail_frame": face["frame"],
                    "appearances": [{
                        "timestamp": face["timestamp"],
                        "bbox": face["bbox"]
                    }]
                })

        return clusters
```

### 3.4 블러 처리

```python
# backend/app/services/video_processor.py

import cv2
import numpy as np
from typing import List, Dict
from enum import Enum

class BlurType(Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"

class VideoProcessor:
    def __init__(self):
        pass

    def apply_blur(
        self,
        frame: np.ndarray,
        bbox: List[int],
        blur_type: BlurType,
        intensity: int = 25
    ) -> np.ndarray:
        """단일 프레임에 블러 적용"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if blur_type == BlurType.GAUSSIAN:
            # 커널 크기는 홀수여야 함
            kernel_size = intensity * 2 + 1
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        elif blur_type == BlurType.MOSAIC:
            # 모자이크: 축소 후 확대
            scale = max(1, intensity // 5)
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (w // scale, h // scale))
            blurred_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif blur_type == BlurType.BLACKOUT:
            blurred_roi = np.zeros_like(roi)

        frame[y1:y2, x1:x2] = blurred_roi
        return frame

    def process_video(
        self,
        input_path: str,
        output_path: str,
        blur_targets: List[Dict],  # face_id별 appearances 정보
        blur_type: BlurType,
        intensity: int,
        progress_callback=None
    ):
        """전체 영상 블러 처리"""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # 현재 프레임에서 블러 적용할 얼굴 찾기
            for target in blur_targets:
                for appearance in target["appearances"]:
                    if appearance["start"] <= current_time <= appearance["end"]:
                        frame = self.apply_blur(
                            frame,
                            appearance["bbox"],
                            blur_type,
                            intensity
                        )

            out.write(frame)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx / total_frames * 100)

        cap.release()
        out.release()
```

---

## 4. 프론트엔드 상세 설계

### 4.1 기술 스택

| 구성요소 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| Framework | React | 18+ | UI 프레임워크 |
| Build Tool | Vite | 5+ | 빌드 도구 |
| Language | TypeScript | 5+ | 타입 안전성 |
| Styling | Tailwind CSS | 3+ | 스타일링 |
| State | Zustand | 4+ | 상태 관리 |
| HTTP Client | Axios | 1+ | API 통신 |

### 4.2 핵심 컴포넌트

#### 4.2.1 UploadZone

```tsx
// frontend/src/components/UploadZone.tsx

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadZoneProps {
  onVideoUpload: (file: File) => void;
  onReferenceUpload: (files: File[]) => void;
}

export const UploadZone = ({ onVideoUpload, onReferenceUpload }: UploadZoneProps) => {
  const [uploadType, setUploadType] = useState<'video' | 'reference'>('video');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (uploadType === 'video') {
      onVideoUpload(acceptedFiles[0]);
    } else {
      onReferenceUpload(acceptedFiles);
    }
  }, [uploadType, onVideoUpload, onReferenceUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: uploadType === 'video'
      ? { 'video/mp4': ['.mp4'], 'video/quicktime': ['.mov'] }
      : { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] },
    maxFiles: uploadType === 'video' ? 1 : 5,
    maxSize: uploadType === 'video' ? 500 * 1024 * 1024 : 10 * 1024 * 1024
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button
          onClick={() => setUploadType('video')}
          className={`px-4 py-2 rounded ${uploadType === 'video' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          영상 업로드
        </button>
        <button
          onClick={() => setUploadType('reference')}
          className={`px-4 py-2 rounded ${uploadType === 'reference' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        >
          본인 사진 업로드
        </button>
      </div>

      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
        `}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-blue-600">파일을 놓으세요...</p>
        ) : (
          <p className="text-gray-600">
            {uploadType === 'video'
              ? '영상 파일을 드래그하거나 클릭하세요 (MP4/MOV, 최대 500MB)'
              : '본인 사진을 드래그하거나 클릭하세요 (1-5장)'}
          </p>
        )}
      </div>
    </div>
  );
};
```

#### 4.2.2 FaceGrid

```tsx
// frontend/src/components/FaceGrid.tsx

import { useState } from 'react';

interface Face {
  face_id: string;
  thumbnail: string;
  first_appearance: number;
  similarity_to_reference: number;
  is_reference: boolean;
  blur_enabled: boolean;
}

interface FaceGridProps {
  faces: Face[];
  onBlurToggle: (faceId: string, enabled: boolean) => void;
  onBlurAll: () => void;
}

export const FaceGrid = ({ faces, onBlurToggle, onBlurAll }: FaceGridProps) => {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">감지된 얼굴 ({faces.length}명)</h3>
        <button
          onClick={onBlurAll}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          본인 외 모두 블러
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {faces.map((face) => (
          <div
            key={face.face_id}
            className={`
              relative rounded-lg overflow-hidden border-2
              ${face.is_reference ? 'border-green-500' : 'border-gray-200'}
            `}
          >
            <img
              src={`data:image/jpeg;base64,${face.thumbnail}`}
              alt={`Face ${face.face_id}`}
              className="w-full aspect-square object-cover"
            />

            <div className="absolute bottom-0 left-0 right-0 bg-black/70 p-2">
              <div className="flex items-center justify-between">
                <span className="text-white text-sm">
                  {face.similarity_to_reference.toFixed(1)}% 유사
                </span>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={face.blur_enabled}
                    onChange={(e) => onBlurToggle(face.face_id, e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-white text-sm">블러</span>
                </label>
              </div>
              <p className="text-gray-300 text-xs mt-1">
                첫 등장: {face.first_appearance.toFixed(1)}초
              </p>
            </div>

            {face.is_reference && (
              <div className="absolute top-2 left-2 bg-green-500 text-white text-xs px-2 py-1 rounded">
                본인
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
```

---

## 5. 보안 및 프라이버시

### 5.1 데이터 처리 정책

| 항목 | 정책 | 구현 |
|------|------|------|
| 영상 저장 | 처리 완료 후 24시간 내 삭제 | Cron job + S3 lifecycle |
| 얼굴 임베딩 | 세션 종료 시 즉시 삭제 | In-memory only |
| 참조 이미지 | 처리 완료 후 즉시 삭제 | Temp file cleanup |
| 로그 | 개인식별정보 제외 | Structured logging |

### 5.2 GDPR/CCPA 준수

```python
# backend/app/core/security.py

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

class DataRetentionManager:
    def __init__(self, base_path: str, retention_hours: int = 24):
        self.base_path = Path(base_path)
        self.retention_hours = retention_hours

    def cleanup_expired(self):
        """만료된 파일 삭제"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        for item in self.base_path.iterdir():
            if item.stat().st_mtime < cutoff.timestamp():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def delete_session_data(self, session_id: str):
        """세션 데이터 즉시 삭제"""
        session_path = self.base_path / session_id
        if session_path.exists():
            shutil.rmtree(session_path)

    def log_deletion(self, item_type: str, item_id: str):
        """삭제 감사 로그"""
        # 개인식별정보 없이 삭제 기록만 남김
        return {
            "event": "data_deletion",
            "type": item_type,
            "id_hash": hash(item_id),  # ID 해시화
            "timestamp": datetime.now().isoformat()
        }
```

---

## 6. 성능 요구사항

### 6.1 목표 지표

| 지표 | 목표값 | 측정 방법 |
|------|--------|----------|
| 영상 분석 시간 | <2분/10분 영상 | Backend latency |
| 블러 처리 시간 | <1분/10분 영상 | Backend latency |
| 얼굴 탐지 정확도 | >95% | YOLOv8 mAP |
| 동일 인물 매칭 정확도 | >90% | ArcFace TAR@FAR |
| 동시 처리 | 10 요청 | Load test |

### 6.2 최적화 전략

```python
# MPS (Apple Silicon) 최적화 설정
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# 프레임 샘플링으로 처리 시간 단축
FRAME_SAMPLE_RATE = 5  # 5프레임마다 탐지 (6fps 영상 기준 ~1.2초 간격)

# 배치 처리로 효율성 향상
BATCH_SIZE = 8  # GPU 메모리에 따라 조절
```

---

## 7. 배포 구성

### 7.1 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - upload_data:/app/uploads
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_PATH=/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  upload_data:
```

### 7.2 Backend Dockerfile

```dockerfile
# backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8. 개발 마일스톤

### Phase 1: 기반 구축
- [ ] 프로젝트 구조 설정
- [ ] FastAPI 서버 기본 설정
- [ ] 파일 업로드 API 구현
- [ ] React 프론트엔드 기본 UI

### Phase 2: AI 파이프라인
- [ ] YOLOv8 얼굴 탐지 통합
- [ ] ArcFace 임베딩 추출
- [ ] 얼굴 클러스터링 및 매칭

### Phase 3: 핵심 기능
- [ ] 블러 처리 엔진
- [ ] 영상 출력 파이프라인
- [ ] 실시간 진행률 WebSocket

### Phase 4: UI/UX 완성
- [ ] FaceGrid 컴포넌트
- [ ] 미리보기 플레이어
- [ ] 다크모드 지원

### Phase 5: 최적화 및 배포
- [ ] MPS/GPU 최적화
- [ ] Docker 배포 설정
- [ ] 성능 테스트

---

## 9. 의존성 목록

### 9.1 Backend (requirements.txt)

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
pydantic>=2.0.0
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.1.0
insightface>=0.7.0
opencv-python>=4.9.0
numpy>=1.24.0
scipy>=1.11.0
pillow>=10.0.0
python-dotenv>=1.0.0
```

### 9.2 Frontend (package.json)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-dropzone": "^14.2.0",
    "axios": "^1.6.0",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "tailwindcss": "^3.4.0",
    "@types/react": "^18.2.0"
  }
}
```

---

*문서 끝*
