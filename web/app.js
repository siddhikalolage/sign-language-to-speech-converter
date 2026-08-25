const video = document.getElementById("video");
const canvas = document.getElementById("captureCanvas");
const imageInput = document.getElementById("imageInput");
const uploadPreview = document.getElementById("uploadPreview");
const predictButton = document.getElementById("predictButton");
const speakButton = document.getElementById("speakButton");
const undoButton = document.getElementById("undoButton");
const clearButton = document.getElementById("clearButton");
const resultWord = document.getElementById("resultWord");
const resultMeta = document.getElementById("resultMeta");
const sentenceText = document.getElementById("sentenceText");
const statusPill = document.getElementById("statusPill");
const cameraView = document.getElementById("cameraView");
const uploadView = document.getElementById("uploadView");
const handOverlay = document.getElementById("handOverlay");

let activeMode = "landmark";
let activeSource = "camera";
let uploadedImageData = "";
let autoPredictTimer = null;
let requestInFlight = false;
let stableHistory = [];
let missStreak = 0;
let committedLabel = "";
let sentenceWords = [];
let speechQueue = [];
let speechActive = false;

const CAMERA_PREDICT_INTERVAL_MS = 50;
const CAMERA_HISTORY_LIMIT = 4;
const CAMERA_MIN_VOTES = 3;
const CAMERA_CONSENSUS_RATIO = 0.7;
const CAMERA_RELEASE_MISSES = 2;
const CAMERA_FRAME_MAX_WIDTH = 320;
const LANDMARK_MIN_CONFIDENCE = 0.55;
const LANDMARK_MIN_QUALITY = 0.55;
const LANDMARK_MIN_MARGIN = 0.25;
const LANDMARK_MIN_HAND_POINTS = 12;
const YOLO_MIN_CONFIDENCE = 0.40;

function setStatus(text) {
  statusPill.textContent = text;
}

function setResult(label, meta) {
  resultWord.textContent = label;
  resultMeta.textContent = meta;
}

function backendName(mode) {
  if (mode === "landmark") {
    return "Landmark";
  }
  if (mode === "yolo-fallback") {
    return "YOLO fallback";
  }
  return "YOLO";
}

function resultMetaText(result, confidence) {
  const orientation = result.orientation ? ` ${result.orientation}` : "";
  const latencyText =
    result.latency_ms === null || result.latency_ms === undefined
      ? ""
      : ` ${Math.round(result.latency_ms)}ms.`;
  const handText =
    result.hand_points === null || result.hand_points === undefined
      ? ""
      : ` Hands ${result.left_hand_points || 0}/${result.right_hand_points || 0}.`;
  const top = Array.isArray(result.top)
    ? result.top
        .slice(0, 3)
        .map((item) => `${item.label} ${Math.round(item.confidence * 100)}%`)
        .join(", ")
    : "";
  return `${backendName(result.mode)}${orientation} confidence ${Math.round(confidence * 100)}%.${latencyText}${handText}${top ? ` Top: ${top}.` : ""}`;
}

function drawHandBoxes(boxes) {
  if (!handOverlay) {
    return;
  }
  handOverlay.replaceChildren();
  if (!Array.isArray(boxes) || activeSource !== "camera") {
    return;
  }

  boxes.forEach((box) => {
    const element = document.createElement("div");
    element.className = "hand-box";
    element.dataset.hand = box.hand || "hand";
    element.style.left = `${Math.max(0, Math.min(1, box.x || 0)) * 100}%`;
    element.style.top = `${Math.max(0, Math.min(1, box.y || 0)) * 100}%`;
    element.style.width = `${Math.max(0, Math.min(1, box.width || 0)) * 100}%`;
    element.style.height = `${Math.max(0, Math.min(1, box.height || 0)) * 100}%`;
    handOverlay.appendChild(element);
  });
}

function currentSentence() {
  return sentenceWords.join(" ").replace(/\s+/g, " ").trim();
}

function updateSentenceText() {
  sentenceText.textContent = currentSentence() || "No words yet.";
}

function pumpSpeechQueue() {
  const speech = window.speechSynthesis;
  if (!speech || speechActive || speechQueue.length === 0) {
    return;
  }

  const phrase = speechQueue.shift();
  const utterance = new SpeechSynthesisUtterance(phrase);
  utterance.lang = "en-US";
  utterance.rate = 0.95;
  utterance.onend = () => {
    speechActive = false;
    pumpSpeechQueue();
  };
  utterance.onerror = () => {
    speechActive = false;
    pumpSpeechQueue();
  };
  speechActive = true;
  speech.speak(utterance);
}

function speakText(text, { flush = false } = {}) {
  const speech = window.speechSynthesis;
  const phrase = String(text || "").trim();
  if (!speech || !phrase) {
    return;
  }

  if (flush) {
    speech.cancel();
    speechQueue = [];
    speechActive = false;
  }

  speechQueue.push(phrase);
  pumpSpeechQueue();
}

function appendWordToSentence(label, { speak = true } = {}) {
  const word = String(label || "").trim();
  if (!word) {
    return;
  }

  sentenceWords.push(word);
  updateSentenceText();
  if (speak) {
    speakText(word);
  }
}

function undoLastWord() {
  if (sentenceWords.length === 0) {
    setStatus("Sentence empty");
    return;
  }

  sentenceWords.pop();
  updateSentenceText();
  setStatus(sentenceWords.length ? "Word removed" : "Sentence empty");
}

function clearSentence() {
  sentenceWords = [];
  updateSentenceText();
  speechQueue = [];
  speechActive = false;
  window.speechSynthesis?.cancel();
}

function setActiveToggle(containerId, value, attrName) {
  const buttons = document.querySelectorAll(`#${containerId} [data-${attrName}]`);
  buttons.forEach((button) => {
    const isActive = button.dataset[attrName] === value;
    button.classList.toggle("active", isActive);
  });
}

function resetCameraConsensus({ clearCommitted = false } = {}) {
  stableHistory = [];
  missStreak = 0;
  if (clearCommitted) {
    committedLabel = "";
  }
}

function shouldKeepPrediction(result) {
  if (!result || !result.label) {
    return false;
  }

  if (result.mode === "landmark") {
    const quality = Number(result.quality ?? 0);
    const margin = Number(result.margin ?? 0);
    return (
      result.confidence >= LANDMARK_MIN_CONFIDENCE &&
      quality >= LANDMARK_MIN_QUALITY &&
      margin >= LANDMARK_MIN_MARGIN &&
      Number(result.hand_points ?? 0) >= LANDMARK_MIN_HAND_POINTS
    );
  }

  return result.confidence >= YOLO_MIN_CONFIDENCE;
}

function commitStablePrediction(result) {
  stableHistory.push(result);
  if (stableHistory.length > CAMERA_HISTORY_LIMIT) {
    stableHistory.shift();
  }
  missStreak = 0;

  const labelCounts = new Map();
  stableHistory.forEach((item) => {
    labelCounts.set(item.label, (labelCounts.get(item.label) || 0) + 1);
  });

  let bestLabel = "";
  let bestCount = 0;
  labelCounts.forEach((count, label) => {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  });

  const requiredVotes = Math.max(
    CAMERA_MIN_VOTES,
    Math.ceil(Math.min(stableHistory.length, CAMERA_HISTORY_LIMIT) * CAMERA_CONSENSUS_RATIO),
  );

  if (bestLabel && bestCount >= requiredVotes) {
    const winners = stableHistory.filter((item) => item.label === bestLabel);
    const averageConfidence =
      winners.reduce((sum, item) => sum + item.confidence, 0) / winners.length;

    if (committedLabel === bestLabel) {
      setResult(
        committedLabel,
        resultMetaText(result, averageConfidence),
      );
      setStatus("Recognized");
      return;
    }

    committedLabel = bestLabel;
    appendWordToSentence(bestLabel);
    setResult(
      committedLabel,
      resultMetaText(result, averageConfidence),
    );
    setStatus("Word added");
  } else {
    setStatus("Hold the phrase steady");
  }
}

function handleCameraMiss(message, { clearBoxes = true } = {}) {
  if (clearBoxes) {
    drawHandBoxes([]);
  }
  missStreak += 1;
  if (!committedLabel) {
    setResult("Waiting", message);
  }
  if (missStreak >= CAMERA_RELEASE_MISSES) {
    stableHistory = [];
    if (committedLabel) {
      committedLabel = "";
      setStatus("Ready for next phrase");
    } else {
      setStatus("Show a phrase");
    }
  }
}

function handleCameraPrediction(result) {
  drawHandBoxes(result.hand_boxes);
  if (!shouldKeepPrediction(result)) {
    handleCameraMiss("Move closer and hold the phrase still.", { clearBoxes: false });
    return;
  }

  commitStablePrediction(result);
}

function stopAutoPredictLoop() {
  if (autoPredictTimer !== null) {
    clearTimeout(autoPredictTimer);
    autoPredictTimer = null;
  }
}

function startAutoPredictLoop() {
  stopAutoPredictLoop();
  if (activeSource !== "camera") {
    return;
  }

  const tick = async () => {
    if (activeSource !== "camera") {
      autoPredictTimer = null;
      return;
    }

    if (requestInFlight || document.hidden) {
      autoPredictTimer = window.setTimeout(tick, CAMERA_PREDICT_INTERVAL_MS);
      return;
    }

    try {
      requestInFlight = true;
      const imageData = captureCameraFrame();
      const result = await sendPrediction(imageData);
      handleCameraPrediction(result);
    } catch (error) {
      console.error(error);
      handleCameraMiss(error.message || "Prediction failed.");
    } finally {
      requestInFlight = false;
      autoPredictTimer = window.setTimeout(tick, CAMERA_PREDICT_INTERVAL_MS);
    }
  };

  autoPredictTimer = window.setTimeout(tick, 0);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera unavailable");
    setResult("Unavailable", "Your browser does not support webcam access.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    setStatus("Camera ready");
    video.onloadedmetadata = () => {
      setStatus("Show a phrase");
      startAutoPredictLoop();
    };
  } catch (error) {
    console.error(error);
    setStatus("Camera blocked");
    setResult("Permission needed", "Allow webcam access or switch to upload mode.");
  }
}

function captureCameraFrame() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    throw new Error("Camera frame is not ready yet.");
  }

  const scale = Math.min(1, CAMERA_FRAME_MAX_WIDTH / width);
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.72);
}

async function sendPrediction(imageData) {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_data: imageData,
      mode: activeMode,
    }),
  });

  const rawText = await response.text();
  let payload = {};
  try {
    payload = rawText ? JSON.parse(rawText) : {};
  } catch (error) {
    payload = { detail: rawText || "Prediction failed." };
  }

  if (!response.ok) {
    throw new Error(payload.detail || "Prediction failed.");
  }
  return payload;
}

async function runPrediction() {
  try {
    setStatus("Predicting");
    let imageData = "";

    if (activeSource === "camera") {
      imageData = captureCameraFrame();
    } else {
      if (!uploadedImageData) {
        throw new Error("Choose an image before predicting.");
      }
      imageData = uploadedImageData;
    }

    const result = await sendPrediction(imageData);
    if (activeSource === "camera") {
      handleCameraPrediction(result);
    } else {
      const qualityText =
        result.quality === null || result.quality === undefined
          ? ""
          : ` Landmarks ${Math.round(result.quality * 100)}%.`;

      setResult(
        result.label,
        `${resultMetaText(result, result.confidence)}${qualityText}`
      );
      appendWordToSentence(result.label);
      setStatus("Word added");
    }
  } catch (error) {
    console.error(error);
    if (activeSource === "camera") {
      handleCameraMiss(error.message || "Prediction failed.");
    } else {
      setResult("No prediction", error.message);
      setStatus("Try again");
    }
  }
}

document.getElementById("modeToggle").addEventListener("click", (event) => {
  const button = event.target.closest("[data-mode]");
  if (!button) {
    return;
  }
  activeMode = button.dataset.mode;
  resetCameraConsensus({ clearCommitted: true });
  setActiveToggle("modeToggle", activeMode, "mode");
});

document.getElementById("sourceToggle").addEventListener("click", (event) => {
  const button = event.target.closest("[data-source]");
  if (!button) {
    return;
  }
  activeSource = button.dataset.source;
  setActiveToggle("sourceToggle", activeSource, "source");
  cameraView.classList.toggle("hidden", activeSource !== "camera");
  uploadView.classList.toggle("hidden", activeSource !== "upload");
  drawHandBoxes([]);
  resetCameraConsensus({ clearCommitted: true });
  setStatus(activeSource === "camera" ? "Show a phrase" : "Upload ready");
  if (activeSource === "camera") {
    startAutoPredictLoop();
  } else {
    stopAutoPredictLoop();
  }
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files || [];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImageData = String(reader.result || "");
    uploadPreview.src = uploadedImageData;
    uploadPreview.classList.remove("hidden");
    setStatus("Image loaded");
  };
  reader.readAsDataURL(file);
});

predictButton.addEventListener("click", () => {
  runPrediction();
});

speakButton.addEventListener("click", () => {
  const sentence = currentSentence();
  if (!sentence) {
    setStatus("Sentence empty");
    return;
  }
  speakText(sentence, { flush: true });
  setStatus("Speaking");
});

undoButton.addEventListener("click", () => {
  undoLastWord();
});

clearButton.addEventListener("click", () => {
  uploadedImageData = "";
  uploadPreview.removeAttribute("src");
  resetCameraConsensus({ clearCommitted: true });
  clearSentence();
  drawHandBoxes([]);
  resultWord.textContent = "Waiting";
  resultMeta.textContent = "Point the camera at a phrase or upload an image.";
  setStatus("Idle");
});

setActiveToggle("modeToggle", activeMode, "mode");
setActiveToggle("sourceToggle", activeSource, "source");
updateSentenceText();
startCamera();
