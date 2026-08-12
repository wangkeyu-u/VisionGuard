const state = {
  file: null,
  objectUrl: null,
  config: null,
  reviews: [],
  activeReview: null,
  corrections: [],
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
  reviewFilter: document.querySelector("#review-filter"),
  queueCount: document.querySelector("#queue-count"),
  reviewQueue: document.querySelector("#review-queue"),
  reviewEmpty: document.querySelector("#review-empty"),
  reviewForm: document.querySelector("#review-form"),
  reviewImage: document.querySelector("#review-image"),
  boxOverlay: document.querySelector("#box-overlay"),
  reviewState: document.querySelector("#review-state"),
  reviewSource: document.querySelector("#review-source"),
  reviewTimestamp: document.querySelector("#review-timestamp"),
  reviewer: document.querySelector("#reviewer"),
  reviewNotes: document.querySelector("#review-notes"),
  bboxList: document.querySelector("#bbox-list"),
  addBox: document.querySelector("#add-box"),
  reviewMessage: document.querySelector("#review-message"),
};

const reviewClasses = ["person", "helmet", "vest", "gloves", "boots", "no_helmet", "no_vest"];

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
  loadReviews(report.review_id);
}

function correctionFromPrediction(prediction, index) {
  return {
    action: "update",
    original_index: index,
    class_name: prediction.class_name,
    xyxy: [...prediction.xyxy],
  };
}

function renderBoxes() {
  elements.boxOverlay.replaceChildren();
  if (!state.activeReview || !elements.reviewImage.naturalWidth) return;
  elements.boxOverlay.style.left = `${elements.reviewImage.offsetLeft}px`;
  elements.boxOverlay.style.top = `${elements.reviewImage.offsetTop}px`;
  elements.boxOverlay.style.width = `${elements.reviewImage.clientWidth}px`;
  elements.boxOverlay.style.height = `${elements.reviewImage.clientHeight}px`;
  const width = elements.reviewImage.naturalWidth;
  const height = elements.reviewImage.naturalHeight;
  state.corrections.forEach((correction, index) => {
    if (correction.action === "delete") return;
    const [x1, y1, x2, y2] = correction.xyxy;
    const box = document.createElement("div");
    box.className = "review-box";
    box.style.left = `${(x1 / width) * 100}%`;
    box.style.top = `${(y1 / height) * 100}%`;
    box.style.width = `${((x2 - x1) / width) * 100}%`;
    box.style.height = `${((y2 - y1) / height) * 100}%`;
    box.dataset.label = `${index + 1} · ${correction.class_name}`;
    elements.boxOverlay.append(box);
  });
}

function renderCorrectionRows() {
  elements.bboxList.replaceChildren();
  state.corrections.forEach((correction, index) => {
    const row = document.createElement("div");
    row.className = `bbox-row${correction.action === "delete" ? " is-deleted" : ""}`;
    const indexLabel = document.createElement("span");
    indexLabel.className = "bbox-index";
    indexLabel.textContent = String(index + 1).padStart(2, "0");
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Box ${index + 1} label`);
    reviewClasses.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      option.selected = correction.class_name === name;
      select.append(option);
    });
    select.disabled = correction.action === "delete";
    select.addEventListener("change", () => {
      correction.class_name = select.value;
      renderBoxes();
    });
    const coordinates = document.createElement("div");
    coordinates.className = "coordinate-grid";
    (correction.xyxy || [0, 0, 1, 1]).forEach((value, coordinateIndex) => {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.value = value;
      input.disabled = correction.action === "delete";
      input.setAttribute("aria-label", `Box ${index + 1} coordinate ${coordinateIndex + 1}`);
      input.addEventListener("change", () => {
        correction.xyxy[coordinateIndex] = Number(input.value);
        renderBoxes();
      });
      coordinates.append(input);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = correction.action === "delete" ? "UNDO" : "DELETE";
    remove.addEventListener("click", () => {
      if (correction.action === "add") state.corrections.splice(index, 1);
      else correction.action = correction.action === "delete" ? "update" : "delete";
      renderCorrectionRows();
      renderBoxes();
    });
    row.append(indexLabel, select, coordinates, remove);
    elements.bboxList.append(row);
  });
  renderBoxes();
}

function openReview(review) {
  state.activeReview = review;
  state.corrections = review.corrections.length
    ? JSON.parse(JSON.stringify(review.corrections))
    : review.predictions.map(correctionFromPrediction);
  elements.reviewEmpty.hidden = true;
  elements.reviewForm.hidden = false;
  elements.reviewImage.src = review.source_url;
  elements.reviewState.textContent = review.status.toUpperCase();
  elements.reviewState.dataset.state = review.status;
  elements.reviewSource.textContent = review.source_name;
  elements.reviewTimestamp.textContent = `QUEUED ${new Date(review.created_at).toLocaleString()}`;
  elements.reviewer.value = review.reviewer;
  elements.reviewNotes.value = review.notes;
  elements.reviewMessage.textContent = "";
  renderCorrectionRows();
  elements.reviewImage.onload = renderBoxes;
}

function renderQueue(focusId) {
  elements.queueCount.textContent = state.reviews.length;
  elements.reviewQueue.replaceChildren();
  if (!state.reviews.length) {
    const empty = document.createElement("p");
    empty.className = "queue-empty";
    empty.textContent = "NO ITEMS IN THIS STATE";
    elements.reviewQueue.append(empty);
    return;
  }
  state.reviews.forEach((review) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "queue-item";
    button.dataset.state = review.status;
    button.innerHTML = `<span>${review.status.toUpperCase()}</span><strong></strong><small></small>`;
    button.querySelector("strong").textContent = review.source_name;
    button.querySelector("small").textContent = `${review.predictions.length} predictions · #${review.id}`;
    button.addEventListener("click", () => openReview(review));
    elements.reviewQueue.append(button);
  });
  const focused = state.reviews.find((review) => review.id === focusId);
  if (focused) openReview(focused);
}

async function loadReviews(focusId) {
  try {
    const suffix = elements.reviewFilter.value ? `?status=${elements.reviewFilter.value}` : "";
    const response = await fetch(`/api/reviews${suffix}`);
    if (!response.ok) throw new Error("Could not load review queue");
    state.reviews = (await response.json()).reviews;
    renderQueue(focusId);
  } catch (error) {
    elements.reviewQueue.textContent = error.message;
  }
}

async function submitReview(status) {
  if (!state.activeReview) return;
  const payload = {
    status,
    reviewer: elements.reviewer.value,
    notes: elements.reviewNotes.value,
    corrections: status === "corrected" ? state.corrections : [],
  };
  try {
    const response = await fetch(`/api/reviews/${state.activeReview.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Review update failed");
    elements.reviewMessage.textContent = `AUDIT SAVED · ${result.status.toUpperCase()}`;
    await loadReviews(result.id);
  } catch (error) {
    elements.reviewMessage.textContent = error.message;
  }
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
elements.reviewFilter.addEventListener("change", () => loadReviews());
elements.addBox.addEventListener("click", () => {
  const width = elements.reviewImage.naturalWidth || 100;
  const height = elements.reviewImage.naturalHeight || 100;
  state.corrections.push({action: "add", class_name: "person", xyxy: [0, 0, width, height]});
  renderCorrectionRows();
});
document.querySelectorAll("[data-status]").forEach((button) => {
  button.addEventListener("click", () => submitReview(button.dataset.status));
});
window.addEventListener("resize", renderBoxes);

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
loadReviews();
