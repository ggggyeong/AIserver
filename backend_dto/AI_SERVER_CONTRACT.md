# AI 서버 ↔ 백엔드 연동 명세

> **이 문서는 Spring Boot 백엔드 ↔ AI 서버 사이의 입/출력 규격만 정리한 문서입니다.**
> 프론트/GPT/TTS 흐름은 `API_SCHEMA.md` 참고.

---

## 0. 한눈에 보기

- **엔드포인트**: `POST http://13.209.89.223:8000/analyze`
- **Swagger**: http://13.209.89.223:8000/docs
- **요청 Content-Type**: `multipart/form-data`
- **응답 Content-Type**: `application/json`
- **타임아웃 권장**: 30초 이상 (Azure Speech 호출 + MediaPipe 분석)

```
[백엔드]  ── multipart(word, frames_json, audio_file) ──▶  [AI 서버]
[백엔드]  ◀── JSON(message, word, feedback_payload)  ──   [AI 서버]
```

---

## 1. 백엔드 → AI 서버 (요청)

### 1-1. HTTP

```
POST /analyze HTTP/1.1
Host: 13.209.89.223:8000
Content-Type: multipart/form-data; boundary=----xxxx
```

### 1-2. multipart 필드 (3개)

| 필드명 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `word` | string (text part) | ✅ | 정답 단어. 예: `"apple"` |
| `frames_json` | string (text part, JSON 직렬화된 문자열) | ✅ | MediaPipe Face Landmarker가 프레임마다 뱉은 raw 데이터 배열 |
| `audio_file` | binary (file part, WAV) | ✅ | 사용자 음성 녹음. **Content-Type은 반드시 `audio/wav` / `audio/wave` / `audio/x-wav` 중 하나** |

> ⚠️ `frames_json`은 JSON **객체**가 아니라 **JSON 문자열**로 보내야 합니다. (multipart의 text part)
> 백엔드는 프론트에서 받은 List를 `objectMapper.writeValueAsString(...)` 으로 직렬화해서 그대로 포워딩하면 됩니다.

### 1-3. `frames_json` 안쪽 구조

`List<RawFrame>` 형태이고, 각 RawFrame은 다음과 같습니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `t_ms` | number | WAV 녹음 시작 시점 기준 경과 시간(ms). 프론트가 `performance.now()` 기준으로 채움 |
| `face_landmarks` | array | 478개의 정규화 좌표 객체 (`{x, y, z}`, 각 0~1) |
| `face_blendshapes` | object | MediaPipe blendshape 52개. **key는 categoryName 문자열, value는 0~1 score** |

#### 예시 (요약본)

```json
[
  {
    "t_ms": 0,
    "face_landmarks": [
      { "x": 0.597, "y": 0.485, "z": -0.038 },
      { "x": 0.601, "y": 0.486, "z": -0.040 }
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
      "tongueOut": 0.0
    }
  },
  {
    "t_ms": 33,
    "face_landmarks": [ /* 478개 */ ],
    "face_blendshapes": { /* 52개 */ }
  }
]
```

> 권장 fps: **30fps** 정도. 너무 적으면 phoneme window 안에 프레임이 안 들어가서 시각 점수가 0이 됩니다.

### 1-4. 백엔드에서 보내는 예시 (Spring WebClient)

```java
MultipartBodyBuilder mb = new MultipartBodyBuilder();
mb.part("word", "apple");
mb.part("frames_json", objectMapper.writeValueAsString(frames));
mb.part("audio_file", new ByteArrayResource(wavBytes) {
    @Override public String getFilename() { return "recording.wav"; }
}).contentType(MediaType.parseMediaType("audio/wav"));

webClient.post()
    .uri("http://13.209.89.223:8000/analyze")
    .contentType(MediaType.MULTIPART_FORM_DATA)
    .body(BodyInserters.fromMultipartData(mb.build()))
    .retrieve()
    .bodyToMono(AnalyzeResponse.class);
```

---

## 2. AI 서버 → 백엔드 (응답)

### 2-1. HTTP 200 (정상)

```json
{
  "message": "analyzed",
  "word": "apple",
  "feedback_payload": { ... }
}
```

### 2-2. 최상위 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `message` | string | 항상 `"analyzed"` |
| `word` | string | 분석한 단어. 요청 word를 그대로 echo |
| `feedback_payload` | object | 분석 결과 본체 ([2-3 참고](#2-3-feedback_payload-구조)) |

### 2-3. `feedback_payload` 구조

| 필드 | 타입 | 설명 |
|---|---|---|
| `word` | string | 단어 |
| `overall_scores` | object | 단어 전체 점수 묶음 |
| `summary` | object | 통계 요약 |
| `phoneme_diagnostics` | array | 모든 음소 상세 진단 |
| `weakest_phonemes` | array | 점수 낮은 순 top 3 |
| `strongest_phonemes` | array | 점수 높은 순 top 3 |
| `mismatches` | array | 오디오/시각 불일치(av_consistent=false) 음소들 |
| `llm_context` | object | **GPT 프롬프트에 그대로 꽂아쓰는 컨텍스트** |

#### 2-3-1. `overall_scores`

| 필드 | 타입 | 설명 |
|---|---|---|
| `audio_score_0_100` | number | Azure 발음 점수 (0~100) |
| `visual_score_0_100` | number | MediaPipe 입모양 점수 (0~100) |
| `fused_score_0_100` | number | 융합 점수 (0~100), 현재 audio 0.75 / visual 0.25 |
| `fused_score_0_10` | number | 융합 점수의 10점 만점 환산 |
| `overall_band` | string | `"excellent"` (95+) / `"good"` (85+) / `"needs_attention"` (70+) / `"weak"` (그 이하) |

#### 2-3-2. `summary`

| 필드 | 타입 | 설명 |
|---|---|---|
| `total_phonemes` | int | 전체 phoneme 개수 |
| `weak_count` | int | band가 `weak`인 phoneme 개수 |
| `mismatch_count` | int | 오디오/시각 불일치 phoneme 개수 |

#### 2-3-3. `phoneme_diagnostics[]` (각 항목)

| 필드 | 타입 | 설명 |
|---|---|---|
| `phoneme` | string | 음소 기호 (예: `"ae"`, `"p"`, `"l"`) |
| `viseme` | string | 시각 매핑된 viseme 그룹 (예: `"PP"`, `"FF"`, `"aa"`) |
| `viseme_hint` | string | viseme에 대한 한국어 설명 |
| `duration_ms` | number | 해당 음소 발화 지속시간 |
| `audio_score_0_100` | number | 오디오 점수 |
| `visual_score_0_100` | number | 시각 점수 |
| `fused_score_0_100` | number | 융합 점수 |
| `fused_score_0_10` | number | 10점 환산 |
| `visual_reliability` | number | 0~1. **0이면 카메라로 감지 불가한 음소** (혀 안쪽 자음 등) |
| `band` | string | excellent / good / needs_attention / weak |
| `error_type` | string | Azure 에러 타입 — 아래 표 참고 |
| `nbest_phonemes` | array | Azure 후보 음소 `[{phoneme, score}, ...]`. 빈 배열 가능 |
| `av_flag` | string | 오디오/시각 일치 플래그 — 아래 표 참고 |
| `av_note_ko` | string\|null | 한국어 진단 메모 |
| `diagnosis_tip` | string\|null | 교정 팁 |
| `av_consistent` | boolean | 오디오/시각 일치 여부 |

##### `error_type` 값

| 값 | 의미 |
|---|---|
| `None` | 정상 |
| `Mispronunciation` | 오발음 |
| `Omission` | 음소 생략 |
| `Insertion` | 불필요한 삽입 |
| `UnexpectedBreak` | 의도치 않은 끊김 |
| `MissingBreak` | 끊어야 할 곳에서 안 끊음 |
| `Monotone` | 단조로움 |

##### `av_flag` 값

| 값 | 의미 |
|---|---|
| `CONSISTENT` | 소리 + 입모양 모두 양호 |
| `AUDIO_CORRECT_VISUAL_WEAK` | 소리는 맞지만 입모양 약함 |
| `VOICING_MISMATCH` | 입술은 맞는데 유성/무성 혼동 (`/p/↔/b/` 등) |
| `FULL_MISMATCH` | 소리 + 입모양 모두 틀림 |
| `OMISSION` | 음소 생략 |
| `ARTICULATION_PARTIAL` | TH 등 — 혀는 나왔지만 소리 부정확 |
| `VISUAL_NOT_AVAILABLE` | 카메라로 감지 불가 |
| `AUDIO_ONLY_ASSESSMENT` | 오디오만으로 평가 |

#### 2-3-4. `weakest_phonemes` / `strongest_phonemes` / `mismatches`

각각 `phoneme_diagnostics`와 동일 객체 구조. 정렬/필터만 다름:
- `weakest_phonemes`: fused 점수 낮은 순 top 3
- `strongest_phonemes`: fused 점수 높은 순 top 3
- `mismatches`: `av_consistent === false` 인 항목들

#### 2-3-5. `llm_context` (GPT 프롬프트 만들 때 사용)

| 필드 | 타입 | 설명 |
|---|---|---|
| `instructions` | object | `{ "순한맛": "...", "매운맛": "..." }` — 사용자가 고른 톤의 value를 GPT 지침으로 사용 |
| `word` | string | 단어 |
| `overall_score` | number | 전체 점수 (0~100) |
| `overall_band` | string | excellent/good/needs_attention/weak |
| `key_issues` | array | 약한 음소 최대 2개 (band가 weak/needs_attention인 것만) |
| `praise_point` | string\|null | 칭찬 문구. 강한 음소가 없으면 null |
| `notes` | array\<string\> | GPT가 반드시 지켜야 할 주의사항 |

##### `key_issues[]` 각 항목

| 필드 | 타입 | 설명 |
|---|---|---|
| `phoneme` | string | 음소 |
| `viseme` | string | viseme |
| `score` | number | fused 점수 |
| `error_type` | string | 위 error_type과 동일 |
| `heard_as` | string\|null | "이 소리처럼 들렸다" — n-best 1위 (예: `"/b/"`) |
| `av_note` | string\|null | `av_note_ko` 그대로 |
| `how_to_fix` | string\|null | `diagnosis_tip` 그대로 |

### 2-4. 응답 전체 예시

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
        "viseme_hint": "턱을 최대로 내려 입을 크게 여는 개모음 계열",
        "duration_ms": 120.0,
        "audio_score_0_100": 92.0,
        "visual_score_0_100": 84.0,
        "fused_score_0_100": 88.0,
        "fused_score_0_10": 8.8,
        "visual_reliability": 0.9,
        "band": "good",
        "error_type": "None",
        "nbest_phonemes": [],
        "av_flag": "CONSISTENT",
        "av_note_ko": "음성과 입모양 모두 양호합니다.",
        "diagnosis_tip": null,
        "av_consistent": true
      },
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
        "nbest_phonemes": [{ "phoneme": "b", "score": 89.0 }],
        "av_flag": "VOICING_MISMATCH",
        "av_note_ko": "입술은 정확히 닫혔지만 /b/처럼 들렸습니다. 성대 울림 여부를 교정하세요.",
        "diagnosis_tip": "입술을 완전히 닫고 성대를 울리지 마세요.",
        "av_consistent": false
      }
    ],
    "weakest_phonemes":   [ /* 위 'p' 항목과 동일 구조 */ ],
    "strongest_phonemes": [ /* 위 'ae' 항목과 동일 구조 */ ],
    "mismatches":         [ /* 위 'p' 항목과 동일 구조 */ ],
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
          "av_note": "입술은 정확히 닫혔지만 /b/처럼 들렸습니다.",
          "how_to_fix": "입술을 완전히 닫고 성대를 울리지 마세요."
        }
      ],
      "praise_point": "/ae/ 발음은 잘 됐습니다.",
      "notes": [
        "av_note와 how_to_fix가 있는 음소는 반드시 해당 내용을 피드백에 포함할 것",
        "heard_as가 있으면 '~처럼 들렸다' 표현 활용 가능",
        "visual_reliability가 0인 음소는 카메라 감지 불가이므로 입모양 피드백 생략",
        "전체 피드백은 반드시 2~3문장 이내로 간결하게 유지"
      ]
    }
  }
}
```

---

## 3. 에러 응답

FastAPI 표준 형식. HTTP 상태코드는 400 또는 500.

```json
{
  "detail": "에러 설명 메시지"
}
```

### 자주 나오는 에러

| 상황 | status | detail 예시 |
|---|---|---|
| `audio_file`이 WAV가 아닐 때 | 400 | `"WAV 형식의 오디오 파일만 지원합니다."` |
| `frames_json` JSON 파싱 실패 | 400 | `"frames_json JSON 파싱 오류: ..."` |
| `frames_json`이 빈 배열 / 배열 아님 | 400 | `"frames_json은 비어있지 않은 배열이어야 합니다."` |
| 첫 프레임 형식 오류 (필드 누락 등) | 400 | `"frames 형식 오류 (1번 프레임): ..."` |
| 얼굴이 한 프레임도 안 잡힘 | 400 | `"유효한 얼굴 프레임을 추출할 수 없습니다. 카메라와 조명을 확인하세요."` |
| Azure 발음 분석 실패 | 400 | `"Azure 발음 분석 실패: ..."` |
| 그 외 내부 예외 | 500 | `"서버 내부 오류: ..."` |

---

## 4. 백엔드가 해야 할 일 정리

1. 프론트에서 받은 `word`, `audio_file(WAV)`, `frames_json(string)` 을 그대로 `multipart/form-data` 로 AI 서버 `/analyze` 에 포워딩.
2. 응답으로 받은 `feedback_payload.llm_context` 를 사용해 GPT 프롬프트 구성:
   - 사용자가 선택한 톤(`순한맛` / `매운맛`)에 맞는 `instructions[톤]` 값을 GPT system/user 프롬프트에 포함
   - `word`, `overall_score`, `praise_point`, `key_issues`, `notes` 를 user 프롬프트에 동봉
3. GPT 응답(2~3문장)을 프론트에 내림. 점수는 `overall_scores.fused_score_0_10` 사용 권장.
