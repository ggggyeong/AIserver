from typing import Any, Dict, Mapping, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def score_phoneme_audio(phoneme_item: Mapping[str, Any]) -> Dict[str, Any]:
    """
    phoneme 단위 audio score.

    error_type 처리:
      - Omission: 음소를 생략했으므로 점수 강제 0
      - Mispronunciation / 그 외: Azure AccuracyScore 그대로 사용
    """
    error_type = str(phoneme_item.get("error_type") or "None")

    if error_type == "Omission":
        accuracy = 0.0
    else:
        accuracy = _safe_float(phoneme_item.get("accuracy"), 0.0)

    return {
        "phoneme": phoneme_item.get("phoneme"),
        "viseme": phoneme_item.get("viseme"),
        "duration_ms": _safe_float(phoneme_item.get("duration_ms"), 0.0),
        "error_type": error_type,
        "nbest_phonemes": list(phoneme_item.get("nbest_phonemes") or []),
        "audio_score_0_100": accuracy,
        "audio_score_0_1": accuracy / 100.0,
    }


def parse_azure_overall_scores(nbest_item: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Azure NBest[0]["PronunciationAssessment"] 에서 overall score 추출.
    """
    pa = nbest_item.get("PronunciationAssessment") or {}

    return {
        "accuracy_score": _safe_float(pa.get("AccuracyScore"), None),
        "fluency_score": _safe_float(pa.get("FluencyScore"), None),
        "completeness_score": _safe_float(pa.get("CompletenessScore"), None),
        "prosody_score": _safe_float(pa.get("ProsodyScore"), None),
        "pron_score": _safe_float(pa.get("PronScore"), None),
    }


def score_word_audio_from_phonemes(
    phoneme_results: Sequence[Mapping[str, Any]],
    fluency_score: float | None = None,
    completeness_score: float | None = None,
) -> Dict[str, Any]:
    """
    word 단위 audio score 산출.

    방식:
      1) phoneme accuracy duration-weighted average
      2) fluency / completeness 약한 보정 (각 15%)
    """
    weighted_sum = 0.0
    duration_sum = 0.0
    phoneme_scores = []

    for item in phoneme_results:
        scored = score_phoneme_audio(item)
        phoneme_scores.append(scored)

        dur = max(_safe_float(scored["duration_ms"], 0.0), 1.0)
        score = _safe_float(scored["audio_score_0_100"], 0.0)

        weighted_sum += score * dur
        duration_sum += dur

    phoneme_weighted_accuracy = weighted_sum / max(duration_sum, 1e-6)

    fluency = _safe_float(fluency_score, phoneme_weighted_accuracy)
    completeness = _safe_float(completeness_score, 100.0)

    custom_audio_score_100 = (
        0.70 * phoneme_weighted_accuracy
        + 0.15 * fluency
        + 0.15 * completeness
    )

    return {
        "custom_audio_score_0_100": custom_audio_score_100,
        "custom_audio_score_0_1": custom_audio_score_100 / 100.0,
        "phoneme_weighted_accuracy_0_100": phoneme_weighted_accuracy,
        "fluency_score_0_100": fluency,
        "completeness_score_0_100": completeness,
        "phoneme_scores": phoneme_scores,
        "total_duration_ms": duration_sum,
    }
