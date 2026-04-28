# PRONIMO AI 서버 연동 스키마

전체 파이프라인: **프론트 → Spring Boot → AI 서버 → Spring Boot → GPT → 프론트 → Azure TTS**

---

## 전체 흐름 요약

```
[프론트]
  웹캠 (MediaPipe) + 마이크 (WAV 녹음)
      │
      │ ① word + WAV + frames_json
      ▼
[Spring Boot :8080]
  DB에서 정답 word 확인
      │
      │ ② word + WAV + frames_json (그대로 포워딩)
      ▼
[AI 서버 :8000]
  발음 분석 (Azure Speech + MediaPipe 융합)
      │
      │ ③ feedback_payload (구조화된 분석 결과)
      ▼
[Spring Boot :8080]
  GPT API 호출
      │
      │ ④ feedback_text (2~3문장 한국어 피드백)
      ▼
[프론트]
  점수 표시 + Azure TTS 재생
```

---

## ① 프론트 → Spring Boot

**형식:** `multipart/form-data`

| 필드 | 타입 | 설명 |
|---|---|---|
| `word` | string | 발음할 정답 단어 (예: "apple") |
| `audio_file` | File (WAV) | 사용자 음성 녹음 파일 |
| `frames_json` | string (JSON) | MediaPipe raw frame 배열 |
| `feedback_style` | string | `"순한맛"` 또는 `"매운맛"` |

**frames_json 구조:**
```json
[
  {
    "t_ms": 0,
    "face_landmarks": [
      {"x": 0.597, "y": 0.485, "z": -0.038},
      ...
    ],
    "face_blendshapes": {
      "jawOpen": 0.72,
      "mouthClose": 0.03,
      "mouthFunnel": 0.15,
      "mouthPucker": 0.08,
      "mouthStretchLeft": 0.12,
      "mouthStretchRight": 0.10,
      "mouthPressLeft": 0.05,
      "mouthPressRight": 0.06,
      "mouthUpperUpLeft": 0.20,
      "mouthUpperUpRight": 0.22,
      "mouthLowerDownLeft": 0.30,
      "mouthLowerDownRight": 0.28,
      "mouthRollLower": 0.04,
      "mouthRollUpper": 0.02,
      "tongueOut": 0.0,
      ...
    }
  },
  ...
]
```

> **프론트 주의사항:**
> - `t_ms` 기준점: `wavRecorder.start()` 완료 후 `performance.now()` 로 설정
> - `face_blendshapes`: MediaPipe가 배열로 주는 것을 `{categoryName: score}` 딕셔너리로 변환 후 전송
> - 권장 fps: 30fps

---

## ② Spring Boot → AI 서버

**엔드포인트:** `POST http://localhost:8000/analyze`

**형식:** `multipart/form-data` (프론트에서 받은 것 그대로 포워딩)

| 필드 | 타입 | 설명 |
|---|---|---|
| `word` | string | 정답 단어 |
| `audio_file` | File (WAV) | 음성 파일 |
| `frames_json` | string (JSON) | MediaPipe raw frame 배열 |

---

## ③ AI 서버 → Spring Boot

**HTTP 200 응답 (JSON):**

```json
{
  "message": "analyzed",
  "word": "apple",
  "feedback_payload": {

    "word": "apple",

    "overall_scores": {
      "audio_score_0_100": 84.5,
      "visual_score_0_100": 71.2,
      "fused_score_0_100": 81.7,
      "fused_score_0_10": 8.2,
      "overall_band": "good"
    },

    "summary": {
      "total_phonemes": 4,
      "weak_count": 1,
      "mismatch_count": 1
    },

    "phoneme_diagnostics": [
      {
        "phoneme": "ae",
        "viseme": "aa",
        "fused_score_0_100": 88.0,
        "fused_score_0_10": 8.8,
        "band": "good",
        "error_type": "None",
        "nbest_phonemes": [],
        "av_flag": "CONSISTENT",
        "av_note_ko": "음성과 입모양 모두 양호합니다.",
        "diagnosis_tip": null
      },
      {
        "phoneme": "p",
        "viseme": "PP",
        "fused_score_0_100": 45.0,
        "fused_score_0_10": 4.5,
        "band": "weak",
        "error_type": "Mispronunciation",
        "nbest_phonemes": [{"phoneme": "b", "score": 89.0}],
        "av_flag": "VOICING_MISMATCH",
        "av_note_ko": "입술은 정확히 닫혔지만 /b/처럼 들렸습니다. 성대 울림 여부를 교정하세요.",
        "diagnosis_tip": "입술을 완전히 닫고 성대를 울리지 마세요."
      }
    ],

    "weakest_phonemes": [ ... ],
    "strongest_phonemes": [ ... ],
    "mismatches": [ ... ],

    "llm_context": {
      "instructions": {
        "순한맛": "한국어로 2~3문장의 따뜻하고 격려하는 발음 피드백을 작성하라. ...",
        "매운맛": "한국어로 2~3문장의 욕쟁이 할머니 스타일 발음 피드백을 작성하라. ..."
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
          "av_note": "입술은 닫혔지만 /b/처럼 들렸습니다.",
          "how_to_fix": "입술을 완전히 닫고 성대를 울리지 마세요."
        }
      ],
      "praise_point": "/ae/ 발음은 잘 됐습니다.",
      "notes": [
        "av_note와 how_to_fix가 있는 음소는 반드시 해당 내용을 피드백에 포함할 것",
        "heard_as가 있으면 '~처럼 들렸다' 표현 활용 가능",
        "전체 피드백은 반드시 2~3문장 이내로 간결하게 유지"
      ]
    }
  }
}
```

**HTTP 400/500 에러 응답:**
```json
{
  "detail": "에러 설명 메시지"
}
```

---

## ④ Spring Boot → GPT

Spring Boot가 `feedback_payload.llm_context`를 사용해서 GPT 프롬프트 구성:

```
[System]
당신은 영어 발음 교정 AI입니다.

[User]
다음 발음 분석 결과를 바탕으로 피드백을 작성하세요.

단어: {word}
점수: {overall_score} / 100

잘한 부분: {praise_point}

교정이 필요한 부분:
{key_issues 내용}

작성 지침:
{instructions.순한맛 또는 instructions.매운맛}

주의사항:
{notes 내용}
```

**GPT 응답 예시 (순한맛):**
```
"apple의 /ae/ 발음은 정말 잘 하셨어요! /p/ 발음에서 입술은 잘 닫으셨는데
소리가 /b/처럼 들렸어요. /p/는 성대를 울리지 않는 무성음이니,
입술을 붙일 때 목에 힘을 빼는 연습을 해보세요. 조금만 더 하면 완벽해질 것 같아요!"
```

**GPT 응답 예시 (매운맛):**
```
"야 /p/ 발음이 왜 /b/처럼 나와? 입술은 잘 닫았는데 성대를 울리면 어떡해.
/p/는 무성음이라고, 목에 힘 빼고 다시 해봐. 이번엔 제대로 할 수 있잖아!"
```

---

## ⑤ Spring Boot → 프론트

**형식:** JSON

```json
{
  "feedback_text": "GPT가 생성한 2~3문장 피드백",
  "score": 8.2,
  "overall_band": "good",
  "word": "apple"
}
```

---

## ⑥ 프론트 → Azure TTS

프론트에서 `feedback_text`를 Azure TTS API에 전달해서 음성으로 재생.

---

## overall_band 기준

| band | 점수 범위 | 의미 |
|---|---|---|
| `excellent` | 95~100점 | 완벽 |
| `good` | 85~94점 | 잘함 |
| `needs_attention` | 70~84점 | 보통 |
| `weak` | 0~69점 | 교정 필요 |

---

## av_flag 종류 (Audio-Visual 일치 여부)

| flag | 의미 |
|---|---|
| `CONSISTENT` | 소리 + 입모양 모두 양호 |
| `AUDIO_CORRECT_VISUAL_WEAK` | 소리는 맞지만 입모양 약함 |
| `VOICING_MISMATCH` | 입술은 맞는데 유성/무성 혼동 (/p/↔/b/) |
| `FULL_MISMATCH` | 소리 + 입모양 모두 틀림 |
| `OMISSION` | 음소 생략 |
| `ARTICULATION_PARTIAL` | TH: 혀는 나왔지만 소리 부정확 |
| `VISUAL_NOT_AVAILABLE` | 카메라로 감지 불가한 음소 (skip) |
| `AUDIO_ONLY_ASSESSMENT` | 오디오만으로 평가 |
