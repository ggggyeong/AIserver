"""
S3 presigned URL → 로컬 임시 WAV 파일.

routes.py 의 /analyze 가 받은 audio_url (presigned HTTPS URL) 을
스트리밍으로 다운로드하면서 다음 안전장치를 건다:

  1) 타임아웃         : connect 5s / read 15s
  2) 사이즈 상한      : 25MB (누적 바이트가 넘으면 즉시 중단)
  3) Content-Type    : audio/* 또는 application/octet-stream 만 허용
                        (S3 업로드 시 Content-Type 미지정이면 octet-stream 기본값)
  4) 깔끔한 400 에러 : 타임아웃/네트워크 오류/HTTP 4xx,5xx 모두 사용자 친화 메시지로 변환
"""

from __future__ import annotations

import os
import tempfile

import httpx
from fastapi import HTTPException


MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB
_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)


async def download_audio_to_temp(url: str) -> str:
    """
    presigned HTTPS URL 을 GET 으로 받아 임시 .wav 파일로 저장하고 그 경로를 반환한다.

    임시 파일 삭제 책임은 호출자에게 있다. 호출자는 finally 블록에서 unlink 해야 한다.

    Raises:
        HTTPException: 다운로드 실패 / 사이즈 초과 / 타입 불일치 시 status 400.
    """
    temp_path: str | None = None
    success = False

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"audio_url 다운로드 실패 (HTTP {response.status_code})",
                    )

                content_type = (response.headers.get("content-type") or "").lower().split(";")[0].strip()
                if not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"오디오 파일이 아닙니다. Content-Type: {content_type or 'unknown'}",
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    temp_path = f.name
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_AUDIO_BYTES:
                            raise HTTPException(
                                status_code=400,
                                detail=f"오디오 파일이 너무 큽니다 (최대 {MAX_AUDIO_BYTES // (1024 * 1024)}MB).",
                            )
                        f.write(chunk)

                if total == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="audio_url 에서 빈 파일을 받았습니다.",
                    )

        success = True
        return temp_path  # type: ignore[return-value]

    except httpx.TimeoutException:
        raise HTTPException(status_code=400, detail="audio_url 다운로드 타임아웃 (S3 응답 지연).")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"audio_url 다운로드 네트워크 오류: {e}")
    finally:
        if not success and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
