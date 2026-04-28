from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from app.services.visual_viseme_scorer import extract_canonical_frame_features


def raw_frames_to_canonical_frames(
    raw_frames: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    프런트가 보낸 raw MediaPipe frame DTO를
    scorer가 기대하는 canonical feature frame으로 변환한다.

    입력 raw frame 예시:
    {
      "t_ms": 1234,
      "face_landmarks": [{"x":..., "y":..., "z":...}, ...],
      "face_blendshapes": {"jawOpen": 0.3, ...}
    }
    """
    canonical_frames: List[Dict[str, Any]] = []

    for frame in raw_frames:
        t_ms = float(frame["t_ms"])
        face_landmarks = frame.get("face_landmarks", [])
        face_blendshapes = frame.get("face_blendshapes", {})

        # 얼굴이 검출되지 않은 프레임은 skip
        if not face_landmarks or not face_blendshapes:
            continue

        canonical = extract_canonical_frame_features(
            face_blendshapes=face_blendshapes,
            face_landmarks=face_landmarks,
        )
        canonical["t_ms"] = t_ms
        canonical_frames.append(canonical)

    return canonical_frames
