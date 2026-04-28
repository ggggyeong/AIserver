from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Sequence

# ── MediaPipe 랜드마크 인덱스 ──────────────────────────────────────────────
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT      = 61
MOUTH_RIGHT     = 291
OUTER_UPPER_LIP = 0
OUTER_LOWER_LIP = 17
INNER_UPPER_LIP = 13
INNER_LOWER_LIP = 14

INNER_LIP_LOOP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 42, 183,
]


# ── 공통 수학 헬퍼 ─────────────────────────────────────────────────────────

def load_profile(path: Path | str) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _avg(mapping: Mapping[str, float], *keys: str) -> float:
    vals = [float(mapping.get(k, 0.0)) for k in keys]
    return sum(vals) / max(len(vals), 1)


def _dist_xy(landmarks: Sequence[Mapping[str, float]], a: int, b: int) -> float:
    ax, ay = float(landmarks[a]["x"]), float(landmarks[a]["y"])
    bx, by = float(landmarks[b]["x"]), float(landmarks[b]["y"])
    return math.hypot(ax - bx, ay - by)


def _polygon_area_xy(
    landmarks: Sequence[Mapping[str, float]], indices: Sequence[int]
) -> float:
    points = [(float(landmarks[i]["x"]), float(landmarks[i]["y"])) for i in indices]
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


# ── Canonical Feature 추출 ────────────────────────────────────────────────

def extract_canonical_frame_features(
    face_blendshapes: Mapping[str, float],
    face_landmarks: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """
    MediaPipe 1 프레임 → canonical feature 17개 (0~1 정규화).

    출처:
      - blendshape 이름: MediaPipe Face Landmarker 공식 문서
      - 랜드마크 인덱스: MediaPipe Face Mesh 478점 명세
      - tongueOut: TalkingHead 오픈소스에서 TH 발음 핵심 feature로 확인
    """
    if len(face_landmarks) <= max(RIGHT_EYE_OUTER, MOUTH_RIGHT, OUTER_LOWER_LIP, INNER_LOWER_LIP):
        raise ValueError("face_landmarks가 478점 MediaPipe face mesh 형식이 아닙니다.")

    interocular = max(_dist_xy(face_landmarks, LEFT_EYE_OUTER, RIGHT_EYE_OUTER), 1e-6)

    mouth_width       = _dist_xy(face_landmarks, MOUTH_LEFT, MOUTH_RIGHT) / interocular
    outer_lip_height  = _dist_xy(face_landmarks, OUTER_UPPER_LIP, OUTER_LOWER_LIP) / interocular
    inner_lip_gap     = _dist_xy(face_landmarks, INNER_UPPER_LIP, INNER_LOWER_LIP) / interocular
    mouth_area        = _polygon_area_xy(face_landmarks, INNER_LIP_LOOP) / (interocular ** 2)

    mouth_width_norm      = _clip(mouth_width / 1.10)
    outer_lip_height_norm = _clip(outer_lip_height / 0.55)
    inner_lip_gap_norm    = _clip(inner_lip_gap / 0.35)
    mouth_area_norm       = _clip(mouth_area / 0.08)
    mouth_aspect_ratio    = _clip((outer_lip_height / max(mouth_width, 1e-6)) / 0.50)

    lip_seal_score = _clip(1.0 - inner_lip_gap_norm)

    stretch_left  = float(face_blendshapes.get("mouthStretchLeft", 0.0))
    stretch_right = float(face_blendshapes.get("mouthStretchRight", 0.0))
    stretch_symmetry = _clip(1.0 - abs(stretch_left - stretch_right))

    roundness_score = _clip(
        (
            1.2 * float(face_blendshapes.get("mouthPucker", 0.0))
            + 1.0 * float(face_blendshapes.get("mouthFunnel", 0.0))
            + 0.8 * (1.0 - mouth_width_norm)
        ) / 3.0
    )

    features = {
        "jawOpen":            float(face_blendshapes.get("jawOpen", 0.0)),
        "mouthClose":         float(face_blendshapes.get("mouthClose", 0.0)),
        "mouthFunnel":        float(face_blendshapes.get("mouthFunnel", 0.0)),
        "mouthPucker":        float(face_blendshapes.get("mouthPucker", 0.0)),
        "mouthStretch":       _avg(face_blendshapes, "mouthStretchLeft", "mouthStretchRight"),
        "mouthPress":         _avg(face_blendshapes, "mouthPressLeft", "mouthPressRight"),
        "upperLipRaise":      _avg(face_blendshapes, "mouthUpperUpLeft", "mouthUpperUpRight"),
        "lowerLipDrop":       _avg(face_blendshapes, "mouthLowerDownLeft", "mouthLowerDownRight"),
        "mouthWidthNorm":     mouth_width_norm,
        "outerLipHeightNorm": outer_lip_height_norm,
        "innerLipGapNorm":    inner_lip_gap_norm,
        "mouthAreaNorm":      mouth_area_norm,
        "mouthAspectRatio":   mouth_aspect_ratio,
        "lipSealScore":       lip_seal_score,
        "roundnessScore":     roundness_score,
        "stretchSymmetry":    stretch_symmetry,
        # TalkingHead 출처: TH(/θ/,/ð/) 발음 시 tongueOut=0.40
        "tongueOut":          _clip(float(face_blendshapes.get("tongueOut", 0.0))),
    }
    return {k: _clip(float(v)) for k, v in features.items()}


# ── Window 통계 ───────────────────────────────────────────────────────────

def summarize_feature_window(
    frame_features: Sequence[Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """프레임 시퀀스 → feature별 통계 (mean, median, max, min, p90, std)."""
    if not frame_features:
        raise ValueError("frame_features가 비어있습니다.")

    feature_names = sorted(frame_features[0].keys())
    summary: Dict[str, Dict[str, float]] = {}

    for name in feature_names:
        values  = [float(frame.get(name, 0.0)) for frame in frame_features]
        ordered = sorted(values)
        avg     = mean(values)
        p90_idx = min(len(ordered) - 1, max(0, math.ceil(0.90 * len(ordered)) - 1))

        summary[name] = {
            "mean":   avg,
            "median": median(values),
            "max":    max(values),
            "min":    min(values),
            "p90":    ordered[p90_idx],
            "std":    math.sqrt(mean([(v - avg) ** 2 for v in values])) if len(values) > 1 else 0.0,
        }
    return summary


# ── Gaussian 헬퍼 ─────────────────────────────────────────────────────────

def _gaussian_match(value: float, target: float, tolerance: float) -> float:
    """목표값에 가까울수록 1.0에 가까운 0~1 점수 반환 (대칭)."""
    tolerance = max(float(tolerance), 1e-6)
    z = (float(value) - float(target)) / tolerance
    return math.exp(-0.5 * z * z)


def _gaussian_match_directional(
    value: float, target: float, tolerance: float, direction: str = "exact"
) -> float:
    """
    단방향 Gaussian.
      "max" : value >= target 이면 1.0, 아니면 Gaussian 패널티
              (개모음 jawOpen처럼 '최소한 이만큼은 열어야' 할 때)
      "min" : value <= target 이면 1.0, 아니면 Gaussian 패널티
              (고모음 jawOpen처럼 '이 이상 열리면 안 될 때')
      "exact": 대칭 Gaussian (기본)
    """
    v, t = float(value), float(target)
    if direction == "max" and v >= t:
        return 1.0
    if direction == "min" and v <= t:
        return 1.0
    return _gaussian_match(v, t, tolerance)


# ── 분석 방법별 스코어 함수 ───────────────────────────────────────────────

def _make_result(
    viseme: str,
    method: str,
    frame_count: int,
    visual_score: float,
    reliability: float,
    flag: str,
    tip: str | None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "viseme":             viseme,
        "analysis_method":    method,
        "frame_count":        frame_count,
        "visual_score_0_1":   _clip(visual_score),
        "visual_reliability": reliability,
        "effective_score_0_1": _clip(visual_score) * reliability,
        "diagnosis_flag":     flag,
        "diagnosis_tip":      tip,
    }
    if extra:
        result.update(extra)
    return result


def _score_skip(viseme: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    diag = spec.get("diagnosis", {})
    return _make_result(
        viseme=viseme,
        method="skip",
        frame_count=0,
        visual_score=0.0,
        reliability=0.0,
        flag=diag.get("fail_flag", "NOT_VISUALLY_DETECTABLE"),
        tip=diag.get("fail_tip_ko"),
    )


def _score_insufficient_frames(viseme: str, spec: Mapping[str, Any], frame_count: int) -> Dict[str, Any]:
    """프레임 3개 미만 → 통계 신뢰 불가."""
    diag = spec.get("diagnosis", {})
    return _make_result(
        viseme=viseme,
        method=spec.get("analysis_method", "unknown"),
        frame_count=frame_count,
        visual_score=0.0,
        reliability=0.0,
        flag="INSUFFICIENT_FRAMES",
        tip=f"프레임 수 부족({frame_count}개)으로 시각 분석 불가.",
    )


def _score_pattern_detection(
    viseme: str,
    frames: Sequence[Mapping[str, float]],
    stats: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    PP 전용: lipSealScore의 peak가 threshold 이상인지 확인.
    점수 = peak 값 자체 (0~1). threshold 미달 시 점수 낮게 유지.
    """
    key     = spec["key_feature"]          # "lipSealScore"
    thresh  = float(spec["peak_threshold"])
    rely    = float(spec["visual_reliability"])
    diag    = spec.get("diagnosis", {})

    peak = float(stats.get(key, {}).get("max", 0.0))

    # secondary check: jawOpen이 낮아야 함
    secondary = spec.get("secondary_check", {})
    jaw_ok = True
    if secondary:
        jaw_max = float(stats.get(secondary["feature"], {}).get("max", 1.0))
        jaw_ok  = jaw_max < float(secondary.get("should_be_below", 0.15))

    detected = peak >= thresh and jaw_ok
    score    = peak  # peak 값이 점수

    flag = diag.get("pass_flag" if detected else "fail_flag", "")
    tip  = None if detected else diag.get("fail_tip_ko")

    return _make_result(
        viseme=viseme,
        method="pattern_detection",
        frame_count=len(frames),
        visual_score=score,
        reliability=rely,
        flag=flag,
        tip=tip,
        extra={
            "lip_closure_peak": round(peak, 4),
            "peak_threshold":   thresh,
            "closure_detected": detected,
        },
    )


def _score_gaussian(
    viseme: str,
    frames: Sequence[Mapping[str, float]],
    stats: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    모음 + FF: profile rules의 target/tolerance 기반 Gaussian matching.
    출처: TalkingHead ARKit 값 기반 target (viseme_feature_profile.json).
    """
    rely  = float(spec["visual_reliability"])
    diag  = spec.get("diagnosis", {})
    rules = spec.get("rules", [])

    weighted_sum  = 0.0
    total_weight  = 0.0
    rule_results: List[Dict[str, Any]] = []

    for rule in rules:
        fname    = rule["feature"]
        stat_key = rule["stat"]
        observed = float(stats.get(fname, {}).get(stat_key, 0.0))
        score    = _gaussian_match_directional(
            observed, rule["target"], rule["tolerance"],
            direction=rule.get("direction", "exact"),
        )

        weighted_sum  += score * rule["weight"]
        total_weight  += rule["weight"]

        rule_results.append({
            "feature":   fname,
            "stat":      stat_key,
            "observed":  round(observed, 4),
            "target":    rule["target"],
            "tolerance": rule["tolerance"],
            "score":     round(score, 4),
            "source":    rule.get("source", ""),
        })

    raw_score = weighted_sum / max(total_weight, 1e-6)

    # 가장 낮은 점수의 rule (주요 feature)가 fail 기준으로 사용
    worst = min(rule_results, key=lambda r: r["score"]) if rule_results else {}
    passed = raw_score >= 0.5

    flag = diag.get("pass_flag" if passed else "fail_flag", "")
    tip  = None if passed else diag.get("fail_tip_ko")

    return _make_result(
        viseme=viseme,
        method="gaussian",
        frame_count=len(frames),
        visual_score=raw_score,
        reliability=rely,
        flag=flag,
        tip=tip,
        extra={"rule_results": rule_results},
    )


def _score_absolute_detection(
    viseme: str,
    frames: Sequence[Mapping[str, float]],
    stats: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    TH 전용: tongueOut 최댓값이 detection_threshold 이상인지 확인.
    출처: TalkingHead tongueOut=0.40, threshold=0.30 (75% 보수적 적용).
    """
    key    = spec["key_feature"]           # "tongueOut"
    thresh = float(spec["detection_threshold"])
    rely   = float(spec["visual_reliability"])
    diag   = spec.get("diagnosis", {})

    max_val  = float(stats.get(key, {}).get("max", 0.0))
    detected = max_val >= thresh
    score    = min(1.0, max_val / max(thresh * 1.5, 1e-6))  # threshold의 1.5배에서 만점

    flag = diag.get("pass_flag" if detected else "fail_flag", "")
    tip  = None if detected else diag.get("fail_tip_ko")

    return _make_result(
        viseme=viseme,
        method="absolute_detection",
        frame_count=len(frames),
        visual_score=score,
        reliability=rely,
        flag=flag,
        tip=tip,
        extra={
            "tongue_out_max":      round(max_val, 4),
            "detection_threshold": thresh,
            "tongue_detected":     detected,
        },
    )


# ── 메인 스코어 함수 ──────────────────────────────────────────────────────

def score_viseme_window(
    viseme: str,
    frame_features: Sequence[Mapping[str, float]],
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    phoneme window 1개 → 시각 점수 + 진단.
    profile의 analysis_method에 따라 분기.
    """
    visemes = profile.get("visemes", {})
    spec    = visemes.get(viseme)

    if not spec or spec.get("analysis_method") == "skip":
        return _score_skip(viseme, spec or {})

    frame_count = len(frame_features)
    if frame_count < 3:
        return _score_insufficient_frames(viseme, spec, frame_count)

    stats  = summarize_feature_window(frame_features)
    method = spec["analysis_method"]

    if method == "pattern_detection":
        return _score_pattern_detection(viseme, frame_features, stats, spec)
    elif method == "gaussian":
        return _score_gaussian(viseme, frame_features, stats, spec)
    elif method == "absolute_detection":
        return _score_absolute_detection(viseme, frame_features, stats, spec)
    else:
        return _score_skip(viseme, spec)


def aggregate_visual_scores(
    scored_visemes: Sequence[Mapping[str, Any]],
    use_duration_weight: bool = True,
) -> Dict[str, Any]:
    """
    viseme 점수 목록 → 단어 전체 시각 점수.
    reliability=0 (skip/insufficient)인 항목은 집계에서 제외.
    가중치: visual_reliability × duration_ms
    """
    numerator   = 0.0
    denominator = 0.0
    items: List[Dict[str, Any]] = []

    for item in scored_visemes:
        raw    = float(item.get("visual_score_0_1", 0.0))
        rely   = float(item.get("visual_reliability", 0.0))
        dur    = max(float(item.get("duration_ms", 0.0)), 1.0) if use_duration_weight else 1.0
        weight = rely * dur

        numerator   += raw * weight
        denominator += weight

        items.append({
            "viseme":            item.get("viseme"),
            "phoneme":           item.get("phoneme"),
            "analysis_method":   item.get("analysis_method"),
            "visual_score_0_1":  raw,
            "visual_reliability": rely,
            "duration_ms":       item.get("duration_ms"),
            "aggregate_weight":  weight,
            "diagnosis_flag":    item.get("diagnosis_flag"),
        })

    visual_score = numerator / max(denominator, 1e-6) if denominator > 0 else 0.0

    return {
        "visual_score_0_1": visual_score,
        "total_weight":     denominator,
        "count":            len(scored_visemes),
        "items":            items,
    }


def score_word_visual_from_windows(
    phoneme_windows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    use_duration_weight: bool = True,
) -> Dict[str, Any]:
    """
    단어 전체 phoneme window → 시각 점수 + 진단 목록.

    각 window 구조:
    {
        "phoneme": "ae",
        "viseme": "aa",
        "duration_ms": 270,
        "frame_features": [{...}, ...],
        "audio_accuracy": 85.0,
    }
    """
    scored_items: List[Dict[str, Any]] = []

    for window in phoneme_windows:
        viseme         = str(window["viseme"])
        frame_features = window.get("frame_features", [])
        duration_ms    = float(window.get("duration_ms", 0.0))

        scored = score_viseme_window(viseme, frame_features, profile)
        scored["phoneme"]        = window.get("phoneme")
        scored["duration_ms"]    = duration_ms
        scored["audio_accuracy"] = window.get("audio_accuracy")

        scored_items.append(scored)

    aggregate = aggregate_visual_scores(scored_items, use_duration_weight=use_duration_weight)

    return {
        "visual_score_0_1": aggregate["visual_score_0_1"],
        "scored_visemes":   scored_items,
        "aggregate":        aggregate,
    }
