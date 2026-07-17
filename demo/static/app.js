const state = {
  file: null,
  objectUrl: null,
  config: null,
};

const elements = {
  systemStatus: document.querySelector("#system-status"),
  modelName: document.querySelector("#model-name"),
  deviceName: document.querySelector("#device-name"),
  inputSize: document.querySelector("#input-size"),
  threshold: document.querySelector("#threshold"),
  uploadLimits: document.querySelector("#upload-limits"),
  dropzone: document.querySelector("#dropzone"),
  fileInput: document.querySelector("#file-input"),
  fileTicket: document.querySelector("#file-ticket"),
  fileName: document.querySelector("#file-name"),
  fileMeta: document.querySelector("#file-meta"),
  clearFile: document.querySelector("#clear-file"),
  runButton: document.querySelector("#run-button"),
  error: document.querySelector("#error-message"),
  viewerEmpty: document.querySelector("#viewer-empty"),
  imagePreview: document.querySelector("#image-preview"),
  videoPreview: document.querySelector("#video-preview"),
  processing: document.querySelector("#processing"),
  results: document.querySelector("#results"),
  decisionStrip: document.querySelector("#decision-strip"),
  decisionText: document.querySelector("#decision-text"),
  detections: document.querySelector("#metric-detections"),
  violations: document.querySelector("#metric-violations"),
  frames: document.querySelector("#metric-frames"),
  flagged: document.querySelector("#metric-flagged"),
  classCounts: document.querySelector("#class-counts"),
  jsonOutput: document.querySelector("#json-output"),
  jsonDownload: document.querySelector("#json-download"),
};

function humanBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function resetMedia() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  elements.imagePreview.removeAttribute("src");
  elements.videoPreview.pause();
  elements.videoPreview.removeAttribute("src");
  elements.imagePreview.hidden = true;
  elements.videoPreview.hidden = true;
}

function showMedia(url, type) {
  elements.viewerEmpty.hidden = true;
  elements.processing.hidden = true;
  elements.imagePreview.hidden = true;
  elements.videoPreview.hidden = true;
  if (type === "image") {
    elements.imagePreview.src = url;
    elements.imagePreview.hidden = false;
  } else {
    elements.videoPreview.src = url;
    elements.videoPreview.hidden = false;
    elements.videoPreview.load();
  }
}

function selectFile(file) {
  if (!file) return;
  state.file = file;
  resetMedia();
  state.objectUrl = URL.createObjectURL(file);
  const mediaType = file.type.startsWith("video/") ? "video" : "image";
  showMedia(state.objectUrl, mediaType);
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${mediaType.toUpperCase()} · ${humanBytes(file.size)}`;
  elements.fileTicket.hidden = false;
  elements.runButton.disabled = false;
  elements.results.hidden = true;
  elements.error.hidden = true;
}

function clearFile() {
  state.file = null;
  elements.fileInput.value = "";
  elements.fileTicket.hidden = true;
  elements.runButton.disabled = true;
  elements.results.hidden = true;
  resetMedia();
  elements.viewerEmpty.hidden = false;
}

function renderReport(report) {
  const summary = report.summary;
  const isViolation = summary.status === "violation_detected";
  elements.decisionStrip.classList.toggle("is-danger", isViolation);
  elements.decisionStrip.classList.toggle("is-safe", !isViolation);
  elements.decisionText.textContent = isViolation ? "REVIEW REQUIRED" : "NO FLAGGED VIOLATION";
  elements.detections.textContent = summary.total_detection_events;
  elements.violations.textContent = summary.violation_detection_events;
  elements.frames.textContent = summary.frames_processed;
  elements.flagged.textContent = summary.frames_with_violations;
  elements.classCounts.replaceChildren();

  const entries = Object.entries(summary.class_counts);
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "class-count";
    empty.textContent = "No detections above threshold";
    elements.classCounts.append(empty);
  } else {
    entries.forEach(([name, count]) => {
      const row = document.createElement("div");
      row.className = "class-count";
      const label = document.createElement("span");
      label.textContent = name;
      const value = document.createElement("strong");
      value.textContent = count;
      row.append(label, value);
      elements.classCounts.append(row);
    });
  }

  elements.jsonOutput.textContent = JSON.stringify(report, null, 2);
  elements.jsonDownload.href = report.report_url;
  showMedia(`${report.result_url}?v=${Date.now()}`, report.media_type);
  elements.results.hidden = false;
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runInference() {
  if (!state.file) return;
  elements.runButton.disabled = true;
  elements.runButton.querySelector("span").textContent = "SCAN IN PROGRESS";
  elements.error.hidden = true;
  elements.imagePreview.hidden = true;
  elements.videoPreview.hidden = true;
  elements.processing.hidden = false;
  elements.viewerEmpty.hidden = true;
  elements.systemStatus.textContent = "MODEL BUSY · LOCAL INFERENCE";

  const form = new FormData();
  form.append("file", state.file, state.file.name);
  try {
    const response = await fetch("/api/infer", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    renderReport(payload);
    elements.systemStatus.textContent = "MODEL READY · SCAN COMPLETE";
  } catch (error) {
    elements.processing.hidden = true;
    elements.viewerEmpty.hidden = false;
    elements.error.textContent = error.message;
    elements.error.hidden = false;
    elements.systemStatus.textContent = "SCAN FAILED · CHECK INPUT";
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.querySelector("span").textContent = "RUN SAFETY SCAN";
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Could not load demo configuration");
    const config = await response.json();
    state.config = config;
    elements.modelName.textContent = config.model;
    elements.deviceName.textContent = config.device.toUpperCase();
    elements.inputSize.textContent = `${config.imgsz} × ${config.imgsz}`;
    elements.threshold.textContent = config.confidence.toFixed(2);
    elements.uploadLimits.textContent =
      `LOCAL ONLY · MAX ${config.max_upload_mb} MB · VIDEO ≤ ${config.max_video_seconds}s`;
    elements.systemStatus.textContent = "MODEL READY · AWAITING INPUT";
  } catch (error) {
    elements.systemStatus.textContent = "MODEL LINK UNAVAILABLE";
    elements.error.textContent = error.message;
    elements.error.hidden = false;
  }
}

elements.fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
elements.clearFile.addEventListener("click", clearFile);
elements.runButton.addEventListener("click", runInference);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragging");
  });
});

elements.dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
loadConfig();
