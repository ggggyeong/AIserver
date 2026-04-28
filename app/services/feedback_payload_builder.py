from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _score_band(score_0_100: float) -> str:
    if score_0_100 >= 95:
        return "excellent"
    if score_0_100 >= 85:
        return "good"
    if score_0_100 >= 70:
        return "needs_attention"
    return "weak"


def _viseme_hint(viseme: str) -> str:
    return {
        "PP": "양 입술이 완전히 닫혀야 하는 계열",
        "FF": "윗니가 아랫입술에 닿아야 하는 계열",
        "TH": "혀를 치아 사이로 내밀어야 하는 계열",
        "DD": "혀 끝이 윗잇몸에 닿는 계열 (카메라 감지 불가)",
        "kk": "혀 뒤쪽이 연구개에 닿는 계열 (카메라 감지 불가)",
        "CH": "후치경 파찰음 계열 (카메라 감지 불가)",
        "SS": "치경 치찰음 계열 (카메라 감지 불가)",
        "nn": "n/l 계열, 혀 끝 접촉 (카메라 감지 불가)",
        "RR": "r 계열, 혀 말림 (카메라 감지 불가)",
        "aa": "턱을 최대로 내려 입을 크게 여는 개모음 계열",
        "E":  "중간 높이, 입술을 옆으로 당기는 중전설모음 계열",
        "ih": "턱을 낮게 유지하며 입술을 옆으로 펼치는 고전설모음 계열",
        "oh": "입술을 둥글게 오므리는 중후설원순모음 계열",
        "ou": "입술을 앞으로 내밀어 최대한 둥글게 만드는 고후설원순모음 계열",
        "sil": "무음/침묵 구간",
    }.get(viseme, "해당 viseme 정보 없음")


def build_feedback_payload(
    word: str,
    audio_scoring: Mapping[str, Any],
    visual_scoring: Mapping[str, Any],
    fused_word: Mapping[str, Any],
    fused_phonemes: Mapping[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    LLM 피드백 생성용 구조화 payload.

    Spring Boot는 이 payload를 받아:
      1. llm_context.instructions 에서 유저 설정(순한맛/매운맛) instruction 선택
      2. key_issues + praise_point 를 GPT 프롬프트에 포함
      3. GPT가 2~3문장 한국어 피드백 생성
      4. Azure TTS로 음성 재생
    """

    # ── phoneme별 진단 통합 ────────────────────────────────────────────────
    phoneme_items: List[Dict[str, Any]] = []

    for fused in fused_phonemes.get("items", []):
        phoneme        = str(fused.get("phoneme", ""))
        viseme         = str(fused.get("viseme", ""))
        audio_s        = _safe_float(fused.get("audio_score_0_1"))
        visual_s       = _safe_float(fused.get("visual_score_0_1"))
        fused_s        = _safe_float(fused.get("fused_score_0_1"))
        duration_ms    = _safe_float(fused.get("duration_ms"))
        reliability    = _safe_float(fused.get("visual_reliability"))
        error_type     = str(fused.get("error_type") or "None")
        nbest_phonemes = list(fused.get("nbest_phonemes") or [])
        av_flag        = str(fused.get("av_flag") or "VISUAL_NOT_AVAILABLE")
        av_note_ko     = fused.get("av_note_ko")
        diagnosis_tip  = fused.get("diagnosis_tip")
        av_consistent  = bool(fused.get("av_consistent", True))

        phoneme_items.append({
            "phoneme":            phoneme,
            "viseme":             viseme,
            "viseme_hint":        _viseme_hint(viseme),
            "duration_ms":        duration_ms,
            "audio_score_0_100":  round(audio_s * 100.0, 1),
            "visual_score_0_100": round(visual_s * 100.0, 1),
            "fused_score_0_100":  round(fused_s * 100.0, 1),
            "fused_score_0_10":   round(fused_s * 10.0, 1),
            "visual_reliability": reliability,
            "band":               _score_band(fused_s * 100.0),
            "error_type":         error_type,
            "nbest_phonemes":     nbest_phonemes,
            "av_flag":            av_flag,
            "av_note_ko":         av_note_ko,
            "diagnosis_tip":      diagnosis_tip,
            "av_consistent":      av_consistent,
        })

    # ── 약한/강한/불일치 분류 ──────────────────────────────────────────────
    weakest   = sorted(phoneme_items, key=lambda x: x["fused_score_0_100"])[:top_k]
    strongest = sorted(phoneme_items, key=lambda x: x["fused_score_0_100"], reverse=True)[:top_k]

    # av_consistent=False인 음소 → 오디오-시각 불일치, 가장 구체적인 피드백 가능
    mismatches = [p for p in phoneme_items if not p["av_consistent"]]

    # ── 전체 점수 ──────────────────────────────────────────────────────────
    overall_audio  = round(_safe_float(audio_scoring.get("custom_audio_score_0_1")) * 100.0, 1)
    overall_visual = round(_safe_float(fused_word.get("visual_score_0_1")) * 100.0, 1)
    overall_fused  = round(_safe_float(fused_word.get("fused_score_0_1")) * 100.0, 1)
    overall_band   = _score_band(overall_fused)

    # ── GPT용 핵심 이슈 추출 (상위 2개 약한 음소) ─────────────────────────
    key_issues: List[Dict[str, Any]] = []
    for p in weakest[:2]:
        if p["band"] in ("weak", "needs_attention"):
            nbest_str = (
                f"/{p['nbest_phonemes'][0]['phoneme']}/"
                if p["nbest_phonemes"] else None
            )
            key_issues.append({
                "phoneme":    p["phoneme"],
                "viseme":     p["viseme"],
                "score":      p["fused_score_0_100"],
                "error_type": p["error_type"],
                "heard_as":   nbest_str,       # "이 소리처럼 들렸다"
                "av_note":    p["av_note_ko"], # 오디오-시각 진단 설명
                "how_to_fix": p["diagnosis_tip"],  # 교정 방법
            })

    # ── 잘한 부분 요약 ─────────────────────────────────────────────────────
    praise_phonemes = [
        p["phoneme"] for p in strongest[:2] if p["band"] in ("excellent", "good")
    ]
    praise_point = (
        f"{', '.join(['/' + ph + '/' for ph in praise_phonemes])} 발음은 잘 됐습니다."
        if praise_phonemes else None
    )

    # ── LLM context ────────────────────────────────────────────────────────
    llm_context = {
        "instructions": {
            "순한맛": (
                "한국어로 2~3문장의 따뜻하고 격려하는 발음 피드백을 작성하라. "
                "잘한 점(praise_point)을 먼저 언급하고, key_issues의 음소를 중심으로 "
                "교정 방법(how_to_fix)을 친절하고 부드럽게 안내하라. "
                "마지막 문장은 응원으로 마무리하라. 전체 2~3문장 이내."
            ),
            "매운맛": (
                "한국어로 2~3문장의 욕쟁이 할머니 스타일 발음 피드백을 작성하라. "
                "'그따위로 발음하면 어떡해', '이게 발음이야 뭐야' 같은 거친 표현으로 강렬하게 시작하라. "
                "key_issues의 가장 심각한 오류와 heard_as를 활용해 "
                "'~/b/처럼 들린다고~' 식으로 구체적으로 지적하라. "
                "how_to_fix 교정 방법은 반드시 포함하되 직설적으로 표현하라. "
                "'이번엔 제대로 해봐', '한번만 더 해보라고' 같은 도발로 마무리. 전체 2~3문장 이내."
            ),
        },
        "word":          word,
        "overall_score": overall_fused,
        "overall_band":  overall_band,
        "key_issues":    key_issues,
        "praise_point":  praise_point,
        "notes": [
            "av_note와 how_to_fix가 있는 음소는 반드시 해당 내용을 피드백에 포함할 것",
            "heard_as가 있으면 '~처럼 들렸다' 표현 활용 가능",
            "visual_reliability가 0인 음소는 카메라 감지 불가이므로 입모양 피드백 생략",
            "전체 피드백은 반드시 2~3문장 이내로 간결하게 유지",
        ],
    }

    return {
        "word": word,
        "overall_scores": {
            "audio_score_0_100":  overall_audio,
            "visual_score_0_100": overall_visual,
            "fused_score_0_100":  overall_fused,
            "fused_score_0_10":   round(overall_fused / 10, 1),
            "overall_band":       overall_band,
        },
        "summary": {
            "total_phonemes": len(phoneme_items),
            "weak_count":     sum(1 for p in phoneme_items if p["band"] == "weak"),
            "mismatch_count": len(mismatches),
        },
        "weakest_phonemes":    weakest,
        "strongest_phonemes":  strongest,
        "mismatches":          mismatches,
        "phoneme_diagnostics": phoneme_items,
        "llm_context":         llm_context,
    }
