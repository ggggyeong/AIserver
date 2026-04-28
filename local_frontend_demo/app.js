import { WavRecorder } from "./wav-recorder.js";
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const els = {
  endpointInput: document.getElementById("endpointInput"),
  wordInput: document.getElementById("wordInput"),
  fpsInput: document.getElementById("fpsInput"),
  modelPathInput: document.getElementById("modelPathInput"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  startCaptureBtn: document.getElementById("startCaptureBtn"),
  stopAnalyzeBtn: document.getElementById("stopAnalyzeBtn"),
  resetBtn: document.getElementById("resetBtn"),
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  logBox: document.getElementById("logBox"),
  resultBox: document.getElementById("resultBox"),
  frameCount: document.getElementById("frameCount"),
  capturedSeconds: document.getElementById("capturedSeconds"),
  lastFaceState: document.getElementById("lastFaceState"),
  modelStatus: document.getElementById("modelStatus"),
  mediaStatus: document.getElementById("mediaStatus"),
  captureStatus: document.getElementById("captureStatus"),
};

const state = {
  faceLandmarker: null,
  mediaStream: null,
  wavRecorder: new WavRecorder(),
  isCapturing: false,
  recordingStartMs: 0,
  rawFrames: [],
  rafId: null,
  lastSampleMs: 0,
  sampleEveryMs: 1000 / 30,
};

function log(message) {
  const now = new Date().toLocaleTimeString();
  els.logBox.textContent += `[${now}] ${message}\n`;
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function setStatus() {
  els.modelStatus.textContent = `모델: ${state.faceLandmarker ? "로드됨" : "대기"}`;
  els.mediaStatus.textContent = `미디어: ${state.mediaStream ? "준비됨" : "대기"}`;
  els.captureStatus.textContent = `캡처: ${state.isCapturing ? "진행중" : "대기"}`;
  els.frameCount.textContent = String(state.rawFrames.length);
  const seconds = state.isCapturing ? ((performance.now() - state.recordingStartMs) / 1000).toFixed(2) : "0.00";
  els.capturedSeconds.textContent = seconds;
}

async function loadFaceLandmarker() {
  if (state.faceLandmarker) {
    log("Face Landmarker는 이미 로드되어 있음.");
    return;
  }

  const wasmRoot = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
  const modelPath = els.modelPathInput.value.trim();

  log(`MediaPipe wasm 로드: ${wasmRoot}`);
  const vision = await FilesetResolver.forVisionTasks(wasmRoot);

  log(`모델 로드 시도: ${modelPath}`);
  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: modelPath,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  setStatus();
  log("Face Landmarker 로드 완료.");
}

async function startCameraAndMic() {
  if (state.mediaStream) {
    log("카메라/마이크는 이미 시작되어 있음.");
    return;
  }

  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: 640,
      height: 480,
      facingMode: "user",
    },
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });

  els.video.srcObject = state.mediaStream;
  await els.video.play();

  const canvas = els.overlay;
  canvas.width = els.video.videoWidth || 640;
  canvas.height = els.video.videoHeight || 480;

  setStatus();
  log("카메라/마이크 시작 완료.");
}

function drawSimpleOverlay(result) {
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

  const faceLandmarks = result?.faceLandmarks?.[0];
  if (!faceLandmarks) return;

  ctx.fillStyle = "#22c55e";
  for (const point of faceLandmarks) {
    const x = point.x * els.overlay.width;
    const y = point.y * els.overlay.height;
    ctx.fillRect(x, y, 2, 2);
  }
}

function toBlendshapeDict(faceBlendshapesResult) {
  const dict = {};
  if (!faceBlendshapesResult || faceBlendshapesResult.length === 0) return dict;

  const categories = faceBlendshapesResult[0].categories || [];
  for (const cat of categories) {
    dict[cat.categoryName] = cat.score;
  }
  return dict;
}

function toLandmarkList(faceLandmarksResult) {
  if (!faceLandmarksResult || faceLandmarksResult.length === 0) return [];
  return faceLandmarksResult[0].map((lm) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z,
  }));
}

function detectCurrentFrame(nowMs) {
  if (!state.faceLandmarker || !els.video || els.video.readyState < 2) return null;

  // 최신 tasks-vision examples는 VIDEO 모드에서 detectForVideo()를 사용한다.
  try {
    return state.faceLandmarker.detectForVideo(els.video, nowMs);
  } catch (_e) {
    // 일부 번들/시그니처 차이를 위해 fallback
    return state.faceLandmarker.detectForVideo(els.video);
  }
}

function captureLoop() {
  if (!state.isCapturing) return;

  const nowMs = performance.now();

  if (nowMs - state.lastSampleMs >= state.sampleEveryMs) {
    const result = detectCurrentFrame(nowMs);
    drawSimpleOverlay(result);

    const faceLandmarks = toLandmarkList(result?.faceLandmarks);
    const faceBlendshapes = toBlendshapeDict(result?.faceBlendshapes);

    if (faceLandmarks.length > 0) {
      const tMs = Math.round(nowMs - state.recordingStartMs);
      state.rawFrames.push({
        t_ms: tMs,
        face_landmarks: faceLandmarks,
        face_blendshapes: faceBlendshapes,
      });
      els.lastFaceState.textContent = "detected";
    } else {
      els.lastFaceState.textContent = "not_detected";
    }

    state.lastSampleMs = nowMs;
    setStatus();
  }

  state.rafId = requestAnimationFrame(captureLoop);
}

async function startCapture() {
  if (!state.faceLandmarker) {
    throw new Error("먼저 모델을 로드해야 함.");
  }
  if (!state.mediaStream) {
    throw new Error("먼저 카메라/마이크를 시작해야 함.");
  }
  if (state.isCapturing) {
    log("이미 캡처 중.");
    return;
  }

  state.rawFrames = [];
  state.sampleEveryMs = 1000 / Number(els.fpsInput.value || 30);
  state.lastSampleMs = 0;
  els.resultBox.textContent = "";

  // WAV 녹음을 먼저 시작한 뒤 기준점 설정
  // → t_ms와 Azure offset이 동일한 기준(WAV 시작)에서 측정됨
  await state.wavRecorder.start(state.mediaStream);
  state.recordingStartMs = performance.now();
  state.isCapturing = true;

  setStatus();
  log("녹음/프레임 추출 시작.");
  captureLoop();
}

async function stopCaptureAndAnalyze() {
  if (!state.isCapturing) {
    log("캡처 중이 아니므로 분석할 수 없음.");
    return;
  }

  state.isCapturing = false;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  const wavBlob = await state.wavRecorder.stop();
  setStatus();

  if (!wavBlob) {
    throw new Error("WAV 녹음 파일 생성 실패.");
  }
  if (state.rawFrames.length === 0) {
    throw new Error("검출된 프레임이 없음. 얼굴이 화면에 잘 보이게 다시 시도.");
  }

  log(`프레임 ${state.rawFrames.length}개, WAV ${Math.round(wavBlob.size / 1024)}KB 생성 완료.`);

  const endpoint = els.endpointInput.value.trim();
  const word = els.wordInput.value.trim();
  if (!endpoint) throw new Error("AI 서버 URL이 비어 있음.");
  if (!word) throw new Error("정답 단어가 비어 있음.");

  const formData = new FormData();
  formData.append("word", word);
  formData.append("frames_json", JSON.stringify(state.rawFrames));
  formData.append("audio_file", new File([wavBlob], "pronunciation.wav", { type: "audio/wav" }));

  log(`FastAPI 전송 시작: ${endpoint}`);

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch (_e) {
    body = { rawText: text };
  }

  if (!response.ok) {
    els.resultBox.textContent = JSON.stringify(body, null, 2);
    throw new Error(`분석 실패: HTTP ${response.status}`);
  }

  els.resultBox.textContent = JSON.stringify(body, null, 2);
  log("분석 완료. 응답을 화면에 표시함.");
}

function resetAll() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = null;
  state.isCapturing = false;
  state.rawFrames = [];
  els.resultBox.textContent = "";
  els.logBox.textContent = "";
  els.lastFaceState.textContent = "none";
  setStatus();
}

els.loadModelBtn.addEventListener("click", async () => {
  try {
    await loadFaceLandmarker();
  } catch (e) {
    console.error(e);
    log(`모델 로드 실패: ${e.message}`);
  }
});

els.startCameraBtn.addEventListener("click", async () => {
  try {
    await startCameraAndMic();
  } catch (e) {
    console.error(e);
    log(`카메라/마이크 시작 실패: ${e.message}`);
  }
});

els.startCaptureBtn.addEventListener("click", async () => {
  try {
    await startCapture();
  } catch (e) {
    console.error(e);
    log(`캡처 시작 실패: ${e.message}`);
  }
});

els.stopAnalyzeBtn.addEventListener("click", async () => {
  try {
    await stopCaptureAndAnalyze();
  } catch (e) {
    console.error(e);
    log(`분석 실패: ${e.message}`);
  }
});

els.resetBtn.addEventListener("click", () => {
  resetAll();
  log("리셋 완료.");
});

setStatus();
log("페이지 로드 완료. 1) 모델 로드 -> 2) 카메라/마이크 시작 -> 3) 녹음/추출 시작 -> 4) 중지 후 분석 순서로 진행.");
