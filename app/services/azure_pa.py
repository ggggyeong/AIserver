import os
import json
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from app.services.viseme_mapper import map_phoneme_to_viseme
from app.services.audio_scorer import (
    parse_azure_overall_scores,
    score_word_audio_from_phonemes,
)

load_dotenv()

VALID_ERROR_TYPES = {
    "None", "Mispronunciation", "Omission", "Insertion",
    "UnexpectedBreak", "MissingBreak", "Monotone"
}


def _parse_nbest_phonemes(pa_dict: dict) -> list:
    """
    Azure NBestPhonemes 파싱.
    반환: [{"phoneme": "b", "score": 87.0}, ...]
    ex) /p/ 자리에서 /b/처럼 발음했다면 → b가 1순위로 옴
    """
    raw = pa_dict.get("NBestPhonemes") or []
    return [
        {"phoneme": item["Phoneme"], "score": float(item.get("Score", 0.0))}
        for item in raw
        if "Phoneme" in item
    ]


def analyze_pronunciation(audio_path: str, word: str) -> dict:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region,
    )
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=word,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
    )

    # NBestPhonemes + Prosody: SDK 버전에 따라 from_json 또는 메서드 방식 시도
    try:
        pa_config_json = json.dumps({
            "referenceText": word,
            "gradingSystem": "HundredMark",
            "granularity": "Phoneme",
            "phonemeAlphabet": "SAPI",
            "NBestPhonemeCount": 5,
            "EnableProsodyAssessment": True,
        })
        pronunciation_config = speechsdk.PronunciationAssessmentConfig.from_json(pa_config_json)
    except AttributeError:
        try:
            pronunciation_config.enable_prosody_assessment()
        except AttributeError:
            pass

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        detail_str = ""
        try:
            cancellation = speechsdk.CancellationDetails.from_result(result)
            detail_str = f"reason={cancellation.reason}, error={cancellation.error_details}"
        except Exception:
            try:
                detail = speechsdk.NoMatchDetails.from_result(result)
                detail_str = str(detail.reason)
            except Exception:
                pass

        return {
            "message": "failed",
            "word": word,
            "audio": audio_path,
            "reason": str(result.reason),
            "detail": detail_str,
        }

    raw_json = result.properties.get(
        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
    )
    data = json.loads(raw_json)

    nbest = data["NBest"][0]
    overall_scores = parse_azure_overall_scores(nbest)

    phoneme_results = []

    for w in nbest.get("Words", []):
        for p in w.get("Phonemes", []):
            phoneme = p["Phoneme"]
            viseme = map_phoneme_to_viseme(phoneme)
            offset_ms = p["Offset"] / 10000
            duration_ms = p["Duration"] / 10000

            pa = p.get("PronunciationAssessment") or {}
            accuracy = pa.get("AccuracyScore")

            error_type = pa.get("ErrorType", "None")
            if error_type not in VALID_ERROR_TYPES:
                error_type = "None"

            nbest_phonemes = _parse_nbest_phonemes(pa)

            phoneme_results.append({
                "word": w["Word"],
                "phoneme": phoneme,
                "viseme": viseme,
                "accuracy": accuracy,
                "error_type": error_type,
                "nbest_phonemes": nbest_phonemes,
                "offset_ms": offset_ms,
                "duration_ms": duration_ms,
            })

    audio_result = score_word_audio_from_phonemes(
        phoneme_results=phoneme_results,
        fluency_score=overall_scores["fluency_score"],
        completeness_score=overall_scores["completeness_score"],
    )

    return {
        "message": "analyzed",
        "word": word,
        "audio": audio_path,
        "phonemes": phoneme_results,
        "azure_overall": overall_scores,
        "audio_scoring": audio_result,
    }
