from typing import Any, Dict, List, Mapping, Sequence


def slice_frames_by_time(
    frames: Sequence[Mapping[str, Any]],
    start_ms: float,
    end_ms: float,
) -> List[Dict[str, Any]]:
    """t_ms가 [start_ms, end_ms] 구간에 포함되는 프레임만 반환."""
    return [
        dict(frame)
        for frame in frames
        if start_ms <= float(frame.get("t_ms", -1)) <= end_ms
    ]


def build_phoneme_windows(
    phoneme_results: Sequence[Mapping[str, Any]],
    canonical_frames: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Azure phoneme 결과 + canonical frames → phoneme별 window 구조.

    window 구조:
    {
        "phoneme":       str,
        "viseme":        str,
        "start_ms":      float,
        "end_ms":        float,
        "duration_ms":   float,
        "frame_count":   int,      ← 추가: 프레임 수 (< 3이면 visual 분석 신뢰 불가)
        "frame_features": [...],   ← canonical feature 프레임 목록
        "audio_accuracy": float | None,
        "error_type":    str,      ← 추가: Azure ErrorType (Mispronunciation 등)
        "nbest_phonemes": [...],   ← 추가: 사용자가 실제로 낸 소리 후보
    }
    """
    windows: List[Dict[str, Any]] = []

    for item in phoneme_results:
        start_ms    = float(item["offset_ms"])
        duration_ms = float(item["duration_ms"])
        end_ms      = start_ms + duration_ms

        window_frames = slice_frames_by_time(canonical_frames, start_ms, end_ms)

        windows.append({
            "phoneme":        item["phoneme"],
            "viseme":         item["viseme"],
            "start_ms":       start_ms,
            "end_ms":         end_ms,
            "duration_ms":    duration_ms,
            "frame_count":    len(window_frames),
            "frame_features": window_frames,
            "audio_accuracy": item.get("accuracy"),
            "error_type":     item.get("error_type", "None"),
            "nbest_phonemes": list(item.get("nbest_phonemes") or []),
        })

    return windows
