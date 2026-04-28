from typing import Dict, List
from pydantic import BaseModel, HttpUrl


class Landmark(BaseModel):
    """MediaPipe Face Mesh 랜드마크 1개 (정규화 좌표)."""
    x: float
    y: float
    z: float


class RawFrame(BaseModel):
    """
    프론트엔드가 보내는 MediaPipe raw frame 1개.

    필드:
      t_ms            : WAV 녹음 시작 기준 경과 시간 (ms)
      face_landmarks  : 478개 NormalizedLandmark (x, y, z)
      face_blendshapes: 52개 blendshape {categoryName: score}
                        ex) {"jawOpen": 0.72, "mouthFunnel": 0.15, ...}

    출처:
      - MediaPipe Face Landmarker JS: FaceLandmarkerResult.faceLandmarks,
        FaceLandmarkerResult.faceBlendshapes
      - https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js
    """
    t_ms: float
    face_landmarks: List[Landmark]
    face_blendshapes: Dict[str, float]


class AnalyzeRequest(BaseModel):
    """
    POST /analyze 의 JSON 요청 본문.

    백엔드(Spring Boot) 흐름:
      1) 프론트가 보낸 WAV를 백엔드가 S3에 PUT 업로드
      2) S3 GET용 presigned HTTPS URL 발급
      3) {word, audio_url, frames} 를 application/json 으로 AI 서버 POST /analyze
    """
    word: str
    audio_url: HttpUrl
    frames: List[RawFrame]
