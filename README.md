# 🎤 Local Pronunciation Demo

웹캠 + 마이크 기반 발음 분석 로컬 데모 프로젝트입니다.

프론트에서 MediaPipe로 입 모양 데이터를 추출하고,
FastAPI 백엔드에서 Azure Pronunciation Assessment + Visual Scoring + Fusion을 수행합니다.

---

## 🧠 전체 구조

Frontend (Browser)
→ MediaPipe (face landmark)
→ raw frames + wav audio 생성
→ FastAPI 전송

Backend (FastAPI)
→ Azure Pronunciation Assessment
→ phoneme alignment
→ visual scoring
→ audio + visual fusion
→ feedback payload 생성

---

## 📦 프로젝트 구조

app/
 ├── api/
 │    └── routes.py
 ├── services/
 ├── data/
 └── main.py

frontend/
 ├── index.html
 ├── app.js
 ├── styles.css
 ├── wav-recorder.js
 └── models/
      └── face_landmarker.task

---

## ⚙️ 실행 방법

## 1. 백엔드 실행

python -m venv venv
source venv/bin/activate   # mac

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger 확인:
http://127.0.0.1:8000/docs

---

## 2. 프론트 실행

cd frontend
python -m http.server 8001

접속:
http://localhost:8001

---

## 🎬 사용 방법

1. 모델 로드
2. 카메라/마이크 시작
3. 녹음 시작
4. 중지 → 분석

테스트 단어:
apple

---

## 🔗 API

POST /analyze-direct

Request (multipart/form-data)

- word: string
- frames_json: JSON string
- audio_file: WAV file

---

## 📊 결과

- audio score (Azure)
- visual score (MediaPipe)
- fused score
- phoneme별 분석
- weakest / strongest phoneme
- mismatch 분석

---

## ⚠️ 주의사항

### 1. python-multipart 필수
pip install python-multipart

### 2. 카메라 오류
file:// 실행 금지  
반드시 localhost 서버에서 실행

### 3. Azure 설정 필요
.env 파일 생성

AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=your_region

---

## 📌 현재 상태

✔ 로컬 데모 완성  
✔ audio + visual fusion 완료  
✔ phoneme 단위 피드백 가능  

---

## 🚀 향후 개선

- UI 개선
- LLM 피드백 문장 생성
- 문장 단위 지원
- Spring Boot 연동
- 실서비스 구조 개선
