from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.schemas.request import RawFrame
from app.services.azure_pa import analyze_pronunciation
from app.services.frame_aligner import build_phoneme_windows
from app.services.visual_viseme_scorer import load_profile, score_word_visual_from_windows
from app.services.fusion_scorer import fuse_word_audio_visual, fuse_phoneme_level
from app.services.feedback_payload_builder import build_feedback_payload
from app.services.raw_frame_adapter import raw_frames_to_canonical_frames

demo_router = APIRouter()

# profile은 서버 시작 시 한 번만 로드
_PROFILE = load_profile(
    Path(__file__).parent.parent / "data" / "viseme_feature_profile.json"
)


# ── 공통 파이프라인 ────────────────────────────────────────────────────────

async def _run_pipeline(
    word: str,
    raw_frames: list,
    audio_path: str,
) -> Dict[str, Any]:
    """
    /analyze-direct 와 /analyze 가 공유하는 핵심 처리 파이프라인.

    1. raw MediaPipe frames → canonical feature frames
    2. Azure Pronunciation Assessment
    3. phoneme 시간 window 구성
    4. 시각 scoring (viseme_feature_profile 기반)
    5. 오디오 + 시각 fusion (동적 가중치)
    6. LLM용 feedback payload 구성
    """
    # 1) canonical frames
    canonical_frames = raw_frames_to_canonical_frames(raw_frames)
    if not canonical_frames:
        raise HTTPException(
            status_code=400,
            detail="유효한 얼굴 프레임을 추출할 수 없습니다. 카메라와 조명을 확인하세요."
        )

    # 2) Azure 발음 분석
    azure_result = analyze_pronunciation(audio_path, word)
    if azure_result.get("message") != "analyzed":
        raise HTTPException(
            status_code=400,
            detail=f"Azure 발음 분석 실패: {azure_result.get('reason', 'unknown')}"
        )

    phoneme_results = azure_result["phonemes"]
    audio_scoring   = azure_result["audio_scoring"]

    # 3) phoneme 시간 window 구성
    phoneme_windows = build_phoneme_windows(
        phoneme_results=phoneme_results,
        canonical_frames=canonical_frames,
    )

    # 4) 시각 scoring
    visual_result = score_word_visual_from_windows(
        phoneme_windows=phoneme_windows,
        profile=_PROFILE,
        use_duration_weight=True,
    )

    # 5) fusion (동적 가중치: visual_reliability 기반)
    fused_word = fuse_word_audio_visual(
        audio_scoring=audio_scoring,
        visual_scoring=visual_result,
        audio_weight=0.75,
        visual_weight=0.25,
    )

    fused_phonemes = fuse_phoneme_level(
        phoneme_audio_scores=audio_scoring["phoneme_scores"],
        scored_visemes=visual_result["scored_visemes"],
        audio_weight=0.75,
        visual_weight=0.25,
    )

    # 6) LLM용 feedback payload
    feedback_payload = build_feedback_payload(
        word=word,
        audio_scoring=audio_scoring,
        visual_scoring=visual_result,
        fused_word=fused_word,
        fused_phonemes=fused_phonemes,
        top_k=3,
    )

    return {
        "azure_result":    azure_result,
        "audio_scoring":   audio_scoring,
        "visual_result":   visual_result,
        "fused_word":      fused_word,
        "fused_phonemes":  fused_phonemes,
        "feedback_payload": feedback_payload,
        "raw_frame_count":       len(raw_frames),
        "canonical_frame_count": len(canonical_frames),
    }


async def _save_audio_temp(audio_file: UploadFile) -> str:
    """UploadFile을 임시 파일로 저장하고 경로 반환."""
    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await audio_file.read())
        return f.name


def _validate_audio(audio_file: UploadFile) -> None:
    allowed = {"audio/wav", "audio/wave", "audio/x-wav"}
    if audio_file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="WAV 형식의 오디오 파일만 지원합니다.")


def _parse_frames(frames_json: str) -> list:
    try:
        raw_frames = json.loads(frames_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"frames_json JSON 파싱 오류: {e}")

    if not isinstance(raw_frames, list) or len(raw_frames) == 0:
        raise HTTPException(status_code=400, detail="frames_json은 비어있지 않은 배열이어야 합니다.")

    # 첫 프레임만 샘플 검증 → 형식 오류를 명확한 메시지로 잡음
    try:
        RawFrame.model_validate(raw_frames[0])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"frames 형식 오류 (1번 프레임): {e}")

    return raw_frames


# ── 엔드포인트 ─────────────────────────────────────────────────────────────

@demo_router.post("/analyze-direct")
async def analyze_direct(
    word: str = Form(...),
    frames_json: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    로컬 HTML 데모용 엔드포인트.
    Spring Boot 없이 프론트 → AI 서버 직접 호출.
    디버그용 중간 결과 포함해서 반환.
    """
    temp_path = None
    try:
        _validate_audio(audio_file)
        raw_frames = _parse_frames(frames_json)
        temp_path  = await _save_audio_temp(audio_file)

        result = await _run_pipeline(word, raw_frames, temp_path)

        return {
            "message":               "analyzed",
            "word":                  word,
            "raw_frame_count":       result["raw_frame_count"],
            "canonical_frame_count": result["canonical_frame_count"],
            "azure_overall":         result["azure_result"].get("azure_overall"),
            "audio_scoring":         result["audio_scoring"],
            "visual_scoring":        result["visual_result"],
            "fused_word":            result["fused_word"],
            "fused_phonemes":        result["fused_phonemes"],
            "feedback_payload":      result["feedback_payload"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@demo_router.post("/analyze")
async def analyze(
    word: str = Form(...),
    frames_json: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Spring Boot 연동용 엔드포인트.
    프론트 → Spring Boot → AI 서버 경로에서 사용.

    Spring Boot는 feedback_payload를 받아:
      1. llm_context.instructions 에서 유저 설정(순한맛/매운맛) instruction 선택
      2. GPT API 호출 → 2~3문장 한국어 피드백 생성
      3. Azure TTS → 음성 재생
    """
    temp_path = None
    try:
        _validate_audio(audio_file)
        raw_frames = _parse_frames(frames_json)
        temp_path  = await _save_audio_temp(audio_file)

        result = await _run_pipeline(word, raw_frames, temp_path)

        return {
            "message":          "analyzed",
            "word":             word,
            "feedback_payload": result["feedback_payload"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
