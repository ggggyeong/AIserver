from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.schemas.request import AnalyzeRequest
from app.services.audio_fetcher import download_audio_to_temp
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


# ── 핵심 파이프라인 ─────────────────────────────────────────────────────────

async def _run_pipeline(
    word: str,
    raw_frames: list,
    audio_path: str,
) -> Dict[str, Any]:
    """
    1. raw MediaPipe frames → canonical feature frames
    2. Azure Pronunciation Assessment
    3. phoneme 시간 window 구성
    4. 시각 scoring (viseme_feature_profile 기반)
    5. 오디오 + 시각 fusion (동적 가중치)
    6. LLM용 feedback payload 구성
    """
    canonical_frames = raw_frames_to_canonical_frames(raw_frames)
    if not canonical_frames:
        raise HTTPException(
            status_code=400,
            detail="유효한 얼굴 프레임을 추출할 수 없습니다. 카메라와 조명을 확인하세요.",
        )

    azure_result = analyze_pronunciation(audio_path, word)
    if azure_result.get("message") != "analyzed":
        raise HTTPException(
            status_code=400,
            detail=f"Azure 발음 분석 실패: {azure_result.get('reason', 'unknown')}",
        )

    phoneme_results = azure_result["phonemes"]
    audio_scoring   = azure_result["audio_scoring"]

    phoneme_windows = build_phoneme_windows(
        phoneme_results=phoneme_results,
        canonical_frames=canonical_frames,
    )

    visual_result = score_word_visual_from_windows(
        phoneme_windows=phoneme_windows,
        profile=_PROFILE,
        use_duration_weight=True,
    )

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

    feedback_payload = build_feedback_payload(
        word=word,
        audio_scoring=audio_scoring,
        visual_scoring=visual_result,
        fused_word=fused_word,
        fused_phonemes=fused_phonemes,
        top_k=3,
    )

    return {"feedback_payload": feedback_payload}


# ── /analyze Swagger 문서 ──────────────────────────────────────────────────

_ANALYZE_DESCRIPTION = """
PRONIMO 발음 분석 — Spring Boot 백엔드 연동 전용 엔드포인트.

### 전체 흐름
1. 프론트가 WAV + MediaPipe frames + word 를 백엔드로 전송
2. **백엔드가 WAV 파일을 S3 에 PUT 업로드**
3. **백엔드가 GET 용 presigned HTTPS URL 발급**
4. 백엔드가 `{word, audio_url, frames}` 를 `application/json` 으로 본 엔드포인트에 POST
5. AI 서버가 `audio_url` 을 GET 으로 다운로드 → Azure Speech + MediaPipe 융합 분석
6. AI 서버가 GPT 프롬프트용 `feedback_payload` 반환
7. 백엔드가 `feedback_payload.llm_context` 로 GPT 호출 → 한국어 피드백 생성

### 요청 본문 (application/json)
| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `word` | string | ✅ | 정답 단어. 예: `"apple"` |
| `audio_url` | string (URL) | ✅ | S3 GET 용 presigned **HTTPS** URL. 응답 Content-Type 은 `audio/*` 또는 `application/octet-stream` 이어야 하며, 파일은 **25MB 이하**. |
| `frames` | array | ✅ | MediaPipe Face Landmarker raw frame 배열. 비어있지 않아야 함. |

`frames[i]` 구조:
```json
{
  "t_ms": 0,
  "face_landmarks": [{"x":0.59,"y":0.48,"z":-0.03}, ... 478개],
  "face_blendshapes": {"jawOpen": 0.72, "mouthFunnel": 0.15, ...}
}
```
- `t_ms` 기준점: WAV 녹음 시작 시점
- 권장 fps: 30

### 응답 (200 OK)
```json
{
  "message": "analyzed",
  "word": "apple",
  "feedback_payload": { ... }
}
```
`feedback_payload` 의 모든 하위 필드 정의는 **`backend_dto/AI_SERVER_CONTRACT.md`** 참고.
주요 가지:
- `overall_scores` — `audio_score_0_100`, `visual_score_0_100`, `fused_score_0_100`, `fused_score_0_10`, `overall_band` (`excellent`/`good`/`needs_attention`/`weak`)
- `summary` — `total_phonemes`, `weak_count`, `mismatch_count`
- `phoneme_diagnostics[]` — 모든 음소 상세 진단 (audio/visual/fused 점수, av_flag, av_note_ko, diagnosis_tip 등)
- `weakest_phonemes` / `strongest_phonemes` / `mismatches` — phoneme_diagnostics 의 정렬·필터 뷰
- `llm_context` — **GPT 프롬프트에 그대로 꽂아쓰는 컨텍스트**:
  - `instructions["순한맛" | "매운맛"]` 중 사용자가 고른 톤의 instruction 문자열
  - `key_issues[]` (약한 음소 최대 2개), `praise_point`, `overall_score`, `overall_band`, `notes[]`

### 백엔드 처리 가이드
1. 사용자 톤 선택값(`"순한맛"` / `"매운맛"`) 으로 `feedback_payload.llm_context.instructions[톤]` 문자열을 꺼낸다.
2. 다음을 GPT user 프롬프트에 동봉: `word`, `overall_score`, `praise_point`, `key_issues`, `notes`.
3. GPT 응답(2~3문장)을 프론트에 내려보냄. 점수는 `overall_scores.fused_score_0_10` 사용 권장.

### 에러 응답 (FastAPI 표준 `{"detail": "..."}`)
| status | 발생 상황 |
| --- | --- |
| 400 | `audio_url` 다운로드 HTTP 4xx/5xx (예: presigned URL 만료) |
| 400 | 다운로드 타임아웃 (connect 5s / read 15s) |
| 400 | 오디오 파일 25MB 초과 |
| 400 | 응답 Content-Type 이 audio/* 도 octet-stream 도 아님 |
| 400 | `frames` 가 비어있거나 얼굴 랜드마크가 한 프레임도 없음 |
| 400 | Azure 발음 분석 실패 |
| 422 | JSON body 검증 실패 (URL 형식 오류, 필수 필드 누락 등) |
| 500 | 그 외 내부 예외 |
"""


_RESPONSE_EXAMPLES: Dict[int, Dict[str, Any]] = {
    200: {
        "description": "분석 성공",
        "content": {
            "application/json": {
                "example": {
                    "message": "analyzed",
                    "word": "apple",
                    "feedback_payload": {
                        "word": "apple",
                        "overall_scores": {
                            "audio_score_0_100": 84.5,
                            "visual_score_0_100": 71.2,
                            "fused_score_0_100": 81.7,
                            "fused_score_0_10": 8.2,
                            "overall_band": "good",
                        },
                        "summary": {
                            "total_phonemes": 4,
                            "weak_count": 1,
                            "mismatch_count": 1,
                        },
                        "phoneme_diagnostics": [
                            {
                                "phoneme": "p",
                                "viseme": "PP",
                                "viseme_hint": "양 입술이 완전히 닫혀야 하는 계열",
                                "duration_ms": 80.0,
                                "audio_score_0_100": 40.0,
                                "visual_score_0_100": 60.0,
                                "fused_score_0_100": 45.0,
                                "fused_score_0_10": 4.5,
                                "visual_reliability": 0.95,
                                "band": "weak",
                                "error_type": "Mispronunciation",
                                "nbest_phonemes": [{"phoneme": "b", "score": 89.0}],
                                "av_flag": "VOICING_MISMATCH",
                                "av_note_ko": "입술은 정확히 닫혔지만 /b/처럼 들렸습니다.",
                                "diagnosis_tip": "입술을 완전히 닫고 성대를 울리지 마세요.",
                                "av_consistent": False,
                            }
                        ],
                        "weakest_phonemes": [],
                        "strongest_phonemes": [],
                        "mismatches": [],
                        "llm_context": {
                            "instructions": {
                                "순한맛": "한국어로 2~3문장의 따뜻하고 격려하는 발음 피드백을 작성하라. ...",
                                "매운맛": "한국어로 2~3문장의 욕쟁이 할머니 스타일 발음 피드백을 작성하라. ...",
                            },
                            "word": "apple",
                            "overall_score": 81.7,
                            "overall_band": "good",
                            "key_issues": [
                                {
                                    "phoneme": "p",
                                    "viseme": "PP",
                                    "score": 45.0,
                                    "error_type": "Mispronunciation",
                                    "heard_as": "/b/",
                                    "av_note": "입술은 정확히 닫혔지만 /b/처럼 들렸습니다.",
                                    "how_to_fix": "입술을 완전히 닫고 성대를 울리지 마세요.",
                                }
                            ],
                            "praise_point": "/ae/ 발음은 잘 됐습니다.",
                            "notes": [
                                "av_note와 how_to_fix가 있는 음소는 반드시 해당 내용을 피드백에 포함할 것",
                                "전체 피드백은 반드시 2~3문장 이내로 간결하게 유지",
                            ],
                        },
                    },
                }
            }
        },
    },
    400: {
        "description": "잘못된 요청 (URL 다운로드 실패, 사이즈 초과, frames 부족, Azure 분석 실패 등)",
        "content": {
            "application/json": {
                "example": {"detail": "audio_url 다운로드 실패 (HTTP 403)"}
            }
        },
    },
    422: {
        "description": "JSON body 검증 실패 (FastAPI 표준)",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "type": "url_parsing",
                            "loc": ["body", "audio_url"],
                            "msg": "Input should be a valid URL",
                            "input": "not-a-url",
                        }
                    ]
                }
            }
        },
    },
    500: {
        "description": "서버 내부 오류",
        "content": {
            "application/json": {
                "example": {"detail": "서버 내부 오류: ..."}
            }
        },
    },
}


# ── 엔드포인트 ─────────────────────────────────────────────────────────────

@demo_router.post(
    "/analyze",
    summary="발음 분석 (Spring Boot 백엔드 전용)",
    description=_ANALYZE_DESCRIPTION,
    response_description="분석 결과 + GPT 프롬프트용 컨텍스트",
    responses=_RESPONSE_EXAMPLES,
)
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    """Spring Boot 백엔드 호출용. 자세한 명세는 위 description 참고."""
    temp_path: str | None = None
    try:
        if not req.frames:
            raise HTTPException(
                status_code=400,
                detail="frames 는 비어있지 않은 배열이어야 합니다.",
            )

        temp_path = await download_audio_to_temp(str(req.audio_url))

        # pydantic 객체 → dict (raw_frame_adapter 가 dict-like 입력을 기대)
        raw_frames = [f.model_dump() for f in req.frames]

        result = await _run_pipeline(req.word, raw_frames, temp_path)

        return {
            "message":          "analyzed",
            "word":             req.word,
            "feedback_payload": result["feedback_payload"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
