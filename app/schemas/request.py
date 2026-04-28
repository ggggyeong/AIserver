from typing import Dict, List
from pydantic import BaseModel


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
    /analyze 및 /analyze-direct 엔드포인트 입력 구조 (문서화용).

    실제 전송 형식: multipart/form-data
      word        : str
      audio_file  : WAV 파일
      frames_json : JSON 문자열 (List[RawFrame])

    프론트엔드 → Spring Boot → AI 서버 경로:
      프론트가 MediaPipe raw frames + WAV를 Spring Boot에 보내면
      Spring Boot가 그대로 AI 서버 /analyze 로 포워딩.
    """
    word: str
    frames: List[RawFrame]
