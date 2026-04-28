from typing import Any, Dict, List, Mapping, Sequence


# ── Audio-Visual 일치 검사 ────────────────────────────────────────────────

# 시각 분석 통과로 판단하는 flag 목록
_VISUAL_PASS_FLAGS = {
    "LIP_CLOSURE_DETECTED",
    "TONGUE_PROTRUSION_DETECTED",
    "JAW_OPEN_SUFFICIENT",
    "MID_FRONT_VOWEL_POSTURE",
    "HIGH_FRONT_VOWEL_POSTURE",
    "LIP_ROUNDING_DETECTED",
    "LIP_ROUNDING_STRONG",
    "LABIODENTAL_POSTURE_DETECTED",
}


def _check_av_consistency(
    error_type: str,
    nbest_phonemes: List[Mapping[str, Any]],
    viseme: str,
    analysis_method: str,
    diagnosis_flag: str,
    visual_score: float,
) -> Dict[str, Any]:
    """
    Azure 오디오 결과 vs 시각 분석 결과 일치 여부 판단.

    반환:
      av_flag   : 상황 코드
      av_note_ko: 한국어 설명 (LLM 참고용)
      consistent: bool
    """
    # 시각 분석 불가 케이스 → 일치 검사 의미없음
    if analysis_method in ("skip", "unknown") or diagnosis_flag in (
        "NOT_VISUALLY_DETECTABLE", "SILENCE_SEGMENT", "INSUFFICIENT_FRAMES"
    ):
        return {
            "av_flag": "VISUAL_NOT_AVAILABLE",
            "av_note_ko": "이 음소는 카메라로 확인이 어려워 음성 분석만 적용됩니다.",
            "consistent": True,
        }

    visual_pass = (diagnosis_flag in _VISUAL_PASS_FLAGS) or (visual_score >= 0.5)
    nbest_top   = nbest_phonemes[0]["phoneme"] if nbest_phonemes else None

    # ── 음소 생략 ──────────────────────────────────────────────────────────
    if error_type == "Omission":
        return {
            "av_flag": "OMISSION",
            "av_note_ko": "이 음소를 생략했습니다. 발음 시 빠뜨리지 않도록 주의하세요.",
            "consistent": False,
        }

    # ── 정상 발음 ──────────────────────────────────────────────────────────
    if error_type == "None":
        if visual_pass:
            return {
                "av_flag": "CONSISTENT",
                "av_note_ko": "음성과 입모양 모두 양호합니다.",
                "consistent": True,
            }
        else:
            return {
                "av_flag": "AUDIO_CORRECT_VISUAL_WEAK",
                "av_note_ko": "소리는 정확하지만 입모양이 약합니다. 의식적으로 입 모양을 만들어 보세요.",
                "consistent": True,
            }

    # ── 오발음 (Mispronunciation) ───────────────────────────────────────────
    if error_type == "Mispronunciation":

        # 양순음(PP): 입술 폐쇄 여부로 원인 구분
        if viseme == "PP":
            if visual_pass:
                # 입술은 닫혔는데 소리가 틀림 → 성대 울림 문제(유무성 혼동)
                note = (
                    f"입술은 정확히 닫혔지만 "
                    f"{'/' + nbest_top + '/' if nbest_top else '다른 소리'}처럼 들렸습니다. "
                    "입 모양은 맞으므로 성대 울림 여부를 교정하세요. "
                    "/p/는 무성음(성대 울림 없음), /b/는 유성음(성대 울림 있음)입니다."
                )
                return {
                    "av_flag": "VOICING_MISMATCH",
                    "av_note_ko": note,
                    "consistent": False,
                }
            else:
                return {
                    "av_flag": "FULL_MISMATCH",
                    "av_note_ko": "입술 폐쇄도 감지되지 않고 소리도 틀렸습니다. 입술을 완전히 붙였다가 떼는 동작을 먼저 연습하세요.",
                    "consistent": False,
                }

        # 치간음(TH): 혀 돌출 여부로 원인 구분
        if viseme == "TH":
            if visual_pass:
                return {
                    "av_flag": "ARTICULATION_PARTIAL",
                    "av_note_ko": "혀는 나왔지만 소리가 정확하지 않습니다. 혀를 좀 더 치아 사이에 고정하고 바람을 내보내세요.",
                    "consistent": False,
                }
            else:
                return {
                    "av_flag": "FULL_MISMATCH",
                    "av_note_ko": "혀 돌출도 감지되지 않고 소리도 틀렸습니다. 혀를 윗니와 아랫니 사이로 내밀어야 합니다.",
                    "consistent": False,
                }

        # 모음 / FF: Gaussian 점수로 판단
        if visual_pass:
            return {
                "av_flag": "AUDIO_MISMATCH_VISUAL_OK",
                "av_note_ko": "입 모양은 비교적 맞지만 소리가 정확하지 않습니다. 발성 자체를 교정해 보세요.",
                "consistent": False,
            }
        else:
            return {
                "av_flag": "FULL_MISMATCH",
                "av_note_ko": "소리와 입 모양 모두 교정이 필요합니다.",
                "consistent": False,
            }

    # ── 그 외 ErrorType (UnexpectedBreak, Monotone 등) ─────────────────────
    return {
        "av_flag": f"AUDIO_ISSUE_{error_type.upper()}",
        "av_note_ko": f"발음 오류 유형: {error_type}. 음성 분석 결과를 참고하세요.",
        "consistent": False,
    }


# ── Word 단위 Fusion ──────────────────────────────────────────────────────

def fuse_word_audio_visual(
    audio_scoring: Mapping[str, Any],
    visual_scoring: Mapping[str, Any],
    audio_weight: float = 0.75,
    visual_weight: float = 0.25,
) -> Dict[str, Any]:
    """
    word 단위 fused score.
    시각 신뢰도가 낮은 단어(skip 음소 많음)는 visual_weight가 자연히 낮아짐.
    """
    audio_score  = float(audio_scoring.get("custom_audio_score_0_1", 0.0))
    visual_score = float(visual_scoring.get("visual_score_0_1", 0.0))

    fused_score = audio_weight * audio_score + visual_weight * visual_score

    return {
        "audio_score_0_1":   audio_score,
        "visual_score_0_1":  visual_score,
        "audio_weight":      audio_weight,
        "visual_weight":     visual_weight,
        "fused_score_0_1":   fused_score,
        "fused_score_0_100": fused_score * 100.0,
    }


# ── Phoneme 단위 Fusion ───────────────────────────────────────────────────

def fuse_phoneme_level(
    phoneme_audio_scores: Sequence[Mapping[str, Any]],
    scored_visemes: Sequence[Mapping[str, Any]],
    audio_weight: float = 0.75,
    visual_weight: float = 0.25,
) -> Dict[str, Any]:
    """
    phoneme 단위 fused score + Audio-Visual 일치 검사.

    fusion 가중치:
      effective_visual_weight = visual_weight × visual_reliability
      → skip/insufficient 음소(reliability=0)는 오디오 100%
      → PP(reliability=0.95)는 오디오 76%, 시각 24%

    phoneme_audio_scores와 scored_visemes는 순서가 대응되어야 함.
    """
    items: List[Dict[str, Any]] = []

    for audio_item, visual_item in zip(phoneme_audio_scores, scored_visemes):
        audio_score  = float(audio_item.get("audio_score_0_1", 0.0))
        visual_score = float(visual_item.get("visual_score_0_1", 0.0))
        reliability  = float(visual_item.get("visual_reliability", 0.0))

        # 신뢰도 기반 동적 가중치: skip 음소는 visual 기여 0
        eff_visual_w = visual_weight * reliability
        eff_audio_w  = 1.0 - eff_visual_w
        fused        = eff_audio_w * audio_score + eff_visual_w * visual_score

        # Audio-Visual 일치 검사
        av = _check_av_consistency(
            error_type     = str(audio_item.get("error_type", "None")),
            nbest_phonemes = list(audio_item.get("nbest_phonemes") or []),
            viseme         = str(visual_item.get("viseme", "")),
            analysis_method= str(visual_item.get("analysis_method", "skip")),
            diagnosis_flag = str(visual_item.get("diagnosis_flag", "")),
            visual_score   = visual_score,
        )

        items.append({
            "phoneme":              audio_item.get("phoneme"),
            "viseme":               audio_item.get("viseme"),
            "duration_ms":          audio_item.get("duration_ms"),
            "error_type":           audio_item.get("error_type", "None"),
            "nbest_phonemes":       list(audio_item.get("nbest_phonemes") or []),
            "audio_score_0_1":      audio_score,
            "visual_score_0_1":     visual_score,
            "visual_reliability":   reliability,
            "effective_audio_w":    round(eff_audio_w, 4),
            "effective_visual_w":   round(eff_visual_w, 4),
            "fused_score_0_1":      fused,
            "fused_score_0_100":    fused * 100.0,
            "diagnosis_flag":       visual_item.get("diagnosis_flag"),
            "diagnosis_tip":        visual_item.get("diagnosis_tip"),
            "av_flag":              av["av_flag"],
            "av_note_ko":           av["av_note_ko"],
            "av_consistent":        av["consistent"],
        })

    return {"items": items}
