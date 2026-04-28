# Local Frontend Demo (Raw MediaPipe → FastAPI)

이 폴더는 **로컬에서 프런트도 실제로 돌려보는** 최소 기능 데모다.

목표는 이거다.

1. 브라우저에서 웹캠/마이크를 연다
2. MediaPipe Face Landmarker로 **raw** `face_landmarks` / `face_blendshapes` 를 뽑는다
3. 같은 녹음 세션 기준으로 `t_ms`를 붙인다
4. WAV 오디오 + raw frame JSON을 FastAPI `analyze-direct` 엔드포인트로 보낸다
5. FastAPI가 Azure 분석 + visual scoring + fusion + feedback payload까지 수행하고 결과를 JSON으로 돌려준다

---

## 폴더 구성

- `index.html`  
  화면과 버튼, 입력창, 결과 영역

- `app.js`  
  MediaPipe 초기화, webcam/mic 제어, raw frame 추출, FastAPI 호출

- `wav-recorder.js`  
  브라우저 마이크 입력을 WAV Blob으로 만드는 간단한 recorder

- `styles.css`  
  최소 스타일

---

## 왜 raw MediaPipe를 보내냐

운영 기준 아키텍처를 맞추기 위해서다.

- 프런트: raw MediaPipe만 추출
- AI 서버: canonical feature 계산, visual/audio/fusion scoring

즉 이 데모도 실사용 구조를 최대한 따르도록 만들었다.

---

## 준비물

### 1) Face Landmarker 모델 다운로드

공식 Web 문서는 Face Landmarker Web에서 `@mediapipe/tasks-vision` 패키지와 `Face Landmarker` 모델을 사용하라고 안내하고, WASM 루트는 jsDelivr CDN 예시를 제공한다. Face Landmarker는 웹에서 `face_landmarks`, `face_blendshapes`, `facial_transformation_matrixes`를 출력한다. citeturn849738view0

로컬에서 제일 쉽게 쓰려면 아래 모델 파일을 받아서 `models/face_landmarker.task`로 두면 된다. Arm 학습 경로 문서는 MediaPipe 모델 번들 다운로드 URL로 아래 `face_landmarker.task` 경로를 제시한다. citeturn298499search5

```text
frontend/
└── models/
    └── face_landmarker.task
```

추천 다운로드 URL:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 2) FastAPI 쪽 local demo endpoint 추가

이 프런트는 `audio_path` 문자열을 직접 줄 수 없기 때문에,
FastAPI에 **dev/demo 전용** `multipart/form-data` 엔드포인트가 있어야 한다.

아래 별도 patch 파일(`local_demo_routes.py`)을 AI 서버에 추가해서 쓴다.

### 3) CORS 허용

프런트를 `http://127.0.0.1:5500` 같은 로컬 정적 서버로 띄우면,
FastAPI는 CORS를 허용해야 한다.

---

## 실행 순서

### A. 프런트 정적 서버 실행

이 폴더에서:

```bash
python -m http.server 5500
```

그다음 브라우저에서:

```text
http://127.0.0.1:5500
```

### B. FastAPI 실행

AI 서버 프로젝트에서:

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### C. 브라우저에서 테스트

1. `모델 로드`
2. `카메라/마이크 시작`
3. 정답 단어 입력 (`apple`)
4. `녹음/추출 시작`
5. 단어를 말함
6. `중지 후 분석`

결과 JSON이 화면 아래에 뜨면 성공.

---

## 각 코드가 하는 일

### `index.html`

- AI 서버 URL 입력
- 정답 단어 입력
- 샘플 fps 입력
- 모델 경로 입력
- 버튼 5개
- video/canvas preview
- 로그 / 최종 결과 출력

### `wav-recorder.js`

- `AudioContext + ScriptProcessorNode`로 마이크 입력을 수집
- Float32 PCM 버퍼를 모음
- 마지막에 WAV Blob으로 인코딩

### `app.js`

#### MediaPipe 초기화
- `FilesetResolver.forVisionTasks(".../wasm")`
- `FaceLandmarker.createFromOptions(...)`
- `runningMode: "VIDEO"`
- `outputFaceBlendshapes: true`

#### webcam/mic 시작
- `getUserMedia({ video, audio })`
- video preview에 stream 연결

#### 캡처 시작
- `recordingStartMs = performance.now()`
- WAV 녹음 시작
- requestAnimationFrame 루프 시작

#### 프레임 추출
- `detectForVideo(video, nowMs)` 또는 fallback
- 결과에서 첫 얼굴만 사용
- `t_ms = performance.now() - recordingStartMs`
- raw DTO 생성:

```json
{
  "t_ms": 1234,
  "face_landmarks": [...478개...],
  "face_blendshapes": {
    "jawOpen": 0.3,
    "mouthClose": 0.1
  }
}
```

#### 분석 요청
- `FormData`
- `word`
- `frames_json`
- `audio_file` (WAV)

를 `POST /analyze-direct`로 전송

#### 결과 표시
- 응답 JSON을 화면에 그대로 표시

---

## 프런트가 FastAPI에 보내는 실제 형태

이 데모는 `multipart/form-data`를 쓴다.

필드:

- `word`: 문자열
- `frames_json`: raw MediaPipe frame 배열을 JSON.stringify 한 문자열
- `audio_file`: WAV 파일

이 방식은 브라우저에서 가장 단순하게 오디오와 큰 JSON을 같이 보내기 쉬워서 선택했다.

---

## 주의

- 이 데모는 **운영용 백엔드 경유 구조를 대체하는 게 아니라**, 로컬 direct demo용이다.
- 운영에서는:
  - 프런트 → Spring Boot
  - Spring Boot → AI FastAPI
- 여기서는 로컬 시연 때문에
  - 프런트 → FastAPI direct
  로 단순화했다.

---

## 코드 해석 포인트

### 왜 `t_ms`를 `performance.now() - recordingStartMs`로 찍냐
오디오와 영상이 같은 세션 기준 시간축을 공유하게 하려는 것이다.
이렇게 해야 Azure phoneme `offset_ms/duration_ms`와 frame `t_ms`를 같은 축에서 비교할 수 있다.

### 왜 frame마다 canonical feature를 계산하지 않냐
운영 기준 역할 분리를 맞추기 위해서다.
프런트는 raw MediaPipe만 보내고,
AI 서버가 내부에서 canonical feature를 계산한다.

### 왜 12fps 같은 샘플링을 쓰냐
Face Landmarker `detectForVideo()`는 동기적으로 실행되고 UI thread를 막을 수 있으므로, 모든 `requestAnimationFrame`마다 다 추출하지 않고 샘플 간격을 둔다. 공식 문서도 `detect()` / `detectForVideo()`가 synchronous 라고 설명한다. citeturn716755view0
