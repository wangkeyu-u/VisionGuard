const VIOLATIONS = {
  no_helmet: {
    label: "未戴安全帽",
    evidence: "The visible head is not protected by a safety helmet.",
  },
  no_vest: {
    label: "未穿反光衣",
    evidence: "The visible torso has no high-visibility safety vest.",
  },
};

const TAG_LABELS = {
  blur: "模糊",
  occlusion: "遮挡",
  low_light: "低光照",
  crowded: "多人密集",
  small_person: "人物过小",
};

const state = {
  snapshot: null,
  split: "train",
  status: "pending",
  category: "all",
  filtered: [],
  currentId: null,
  people: [],
  selectedUid: null,
  uncertainties: [],
  qualityTags: new Set(),
  contractConfirmed: false,
  dirty: false,
  busy: false,
  addMode: false,
  pointer: null,
  uidCounter: 0,
  toastTimer: null,
};

const elements = {
  reviewer: document.querySelector("#reviewer-input"),
  systemState: document.querySelector("#system-state"),
  refresh: document.querySelector("#refresh-button"),
  splitProgress: document.querySelector("#split-progress"),
  splitFilter: document.querySelector("#split-filter"),
  statusFilter: document.querySelector("#status-filter"),
  categoryFilter: document.querySelector("#category-filter"),
  queueCount: document.querySelector("#queue-count"),
  queuePosition: document.querySelector("#queue-position"),
  candidateList: document.querySelector("#candidate-list"),
  composition: document.querySelector("#composition-stats"),
  candidateContext: document.querySelector("#candidate-context"),
  candidateTitle: document.querySelector("#candidate-title"),
  candidateMeta: document.querySelector("#candidate-meta"),
  previous: document.querySelector("#previous-button"),
  next: document.querySelector("#next-button"),
  addPerson: document.querySelector("#add-person-button"),
  deletePerson: document.querySelector("#delete-person-button"),
  sourceToggle: document.querySelector("#source-overlay-toggle"),
  sourceToggleLabel: document.querySelector(".overlay-toggle"),
  blindBanner: document.querySelector("#blind-banner"),
  stageEmpty: document.querySelector("#stage-empty"),
  mediaLayer: document.querySelector("#media-layer"),
  imageLoading: document.querySelector("#image-loading"),
  image: document.querySelector("#candidate-image"),
  canvas: document.querySelector("#annotation-canvas"),
  canvasHint: document.querySelector("#canvas-hint"),
  emptyInspector: document.querySelector("#empty-inspector"),
  reviewForm: document.querySelector("#review-form"),
  saveState: document.querySelector("#save-state"),
  markCompliant: document.querySelector("#mark-compliant-button"),
  personCount: document.querySelector("#person-count"),
  personList: document.querySelector("#person-list"),
  uncertainties: document.querySelector("#uncertainties-input"),
  qualityTags: document.querySelector("#quality-tags"),
  contractConfirmation: document.querySelector("#person-box-contract"),
  outputPath: document.querySelector("#output-path"),
  recordSummary: document.querySelector("#record-summary"),
  exclude: document.querySelector("#exclude-button"),
  approve: document.querySelector("#approve-button"),
  excludeDialog: document.querySelector("#exclude-dialog"),
  excludeForm: document.querySelector("#exclude-form"),
  excludeReason: document.querySelector("#exclude-reason"),
  cancelExclude: document.querySelector("#cancel-exclude"),
  toast: document.querySelector("#toast"),
};

function currentCandidate() {
  return state.snapshot?.candidates.find((item) => item.id === state.currentId) || null;
}

function nextUid() {
  state.uidCounter += 1;
  return `person-${state.uidCounter}`;
}

function makePerson(box, findings = {}) {
  return {
    uid: nextUid(),
    box: box.map((value) => Math.max(0, Math.min(1000, Math.round(value)))),
    findings: Object.fromEntries(
      Object.entries(VIOLATIONS).map(([name, config]) => [
        name,
        {
          enabled: Boolean(findings[name]),
          evidence: findings[name]?.evidence || config.evidence,
          confidence: findings[name]?.confidence || "medium",
        },
      ]),
    ),
  };
}

function orderedPeople() {
  return [...state.people].sort((a, b) => a.box[0] - b.box[0] || a.box[1] - b.box[1]);
}

function setDirty(dirty = true) {
  state.dirty = dirty;
  elements.saveState.classList.toggle("is-dirty", dirty);
  elements.saveState.classList.toggle("is-saved", !dirty && Boolean(state.currentId));
  elements.saveState.textContent = dirty ? "有未保存修改" : state.currentId ? "已同步" : "尚未修改";
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3200);
}

function setBusy(busy, message = "") {
  state.busy = busy;
  syncApprovalState();
  elements.exclude.disabled = busy;
  elements.systemState.textContent = message || (busy ? "正在写入金标" : "本地审核服务已连接");
}

function syncApprovalState() {
  elements.approve.disabled = state.busy || !state.contractConfirmed;
}

function invalidateGeometryContract() {
  state.contractConfirmed = false;
  elements.contractConfirmation.checked = false;
  syncApprovalState();
}

function availableSplits() {
  return new Set(state.snapshot.candidates.map((candidate) => candidate.source_split));
}

function syncAvailableSplits() {
  const available = availableSplits();
  elements.splitFilter.querySelectorAll("button[data-value]").forEach((button) => {
    button.disabled = !available.has(button.dataset.value);
  });
  if (!available.has(state.split)) {
    state.split = ["train", "dev", "test"].find((split) => available.has(split)) || "train";
  }
  syncFilterButtons(elements.splitFilter, state.split);
}

function renderProgress() {
  elements.splitProgress.replaceChildren();
  const available = availableSplits();
  ["train", "dev", "test"].filter((split) => available.has(split)).forEach((split) => {
    const count = state.snapshot.counts[split] || 0;
    const target = state.snapshot.targets[split];
    const row = document.createElement("div");
    row.className = `progress-row${count >= target ? " is-complete" : ""}`;
    const label = document.createElement("span");
    label.textContent = split.toUpperCase();
    const track = document.createElement("div");
    track.className = "progress-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.min(100, (count / target) * 100)}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = `${count} / ${target}`;
    row.append(label, track, value);
    elements.splitProgress.append(row);
  });
}

function renderComposition() {
  const stats = state.snapshot.composition || {};
  elements.composition.replaceChildren();
  [
    ["violation", "违规"],
    ["compliant", "合规"],
    ["uncertain", "含不确定性"],
  ].forEach(([key, label]) => {
    const item = document.createElement("div");
    item.className = "mix-stat";
    const value = document.createElement("strong");
    value.textContent = stats[key] || 0;
    const caption = document.createElement("span");
    caption.textContent = label;
    item.append(value, caption);
    elements.composition.append(item);
  });
}

function filterCandidates() {
  state.filtered = state.snapshot.candidates.filter((candidate) => {
    const splitMatches = candidate.source_split === state.split;
    const statusMatches = candidate.status === state.status;
    const categoryMatches = state.category === "all" || candidate.category === state.category;
    return splitMatches && statusMatches && categoryMatches;
  });
}

function renderQueue() {
  filterCandidates();
  elements.candidateList.replaceChildren();
  elements.queueCount.textContent = `${state.filtered.length} 个样本`;
  const currentIndex = state.filtered.findIndex((item) => item.id === state.currentId);
  elements.queuePosition.textContent = currentIndex < 0 ? "— / —" : `${currentIndex + 1} / ${state.filtered.length}`;

  if (!state.filtered.length) {
    const empty = document.createElement("div");
    empty.className = "queue-empty";
    empty.textContent = "这个筛选条件下没有样本。可以切换状态或数据切分。";
    elements.candidateList.append(empty);
  }

  state.filtered.forEach((candidate, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `candidate-item${candidate.id === state.currentId ? " is-active" : ""}`;
    button.dataset.id = candidate.id;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(candidate.id === state.currentId));
    const number = document.createElement("span");
    number.className = "candidate-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const copy = document.createElement("span");
    copy.className = "candidate-name";
    const name = document.createElement("strong");
    name.textContent = candidate.id;
    const detail = document.createElement("small");
    detail.textContent = candidate.blind ? "BLIND REVIEW" : String(candidate.category || "candidate").replaceAll("_", " ");
    copy.append(name, detail);
    const pip = document.createElement("i");
    pip.className = `status-pip ${candidate.status}`;
    button.append(number, copy, pip);
    button.addEventListener("click", () => selectCandidate(candidate.id));
    elements.candidateList.append(button);
  });

  elements.previous.disabled = currentIndex <= 0;
  elements.next.disabled = currentIndex < 0 || currentIndex >= state.filtered.length - 1;
}

function renderSnapshot({ preferredId = null, preferredIndex = 0 } = {}) {
  renderProgress();
  renderComposition();
  filterCandidates();
  let candidateId = preferredId;
  if (!state.filtered.some((candidate) => candidate.id === candidateId)) {
    candidateId = state.filtered[Math.min(preferredIndex, Math.max(0, state.filtered.length - 1))]?.id || null;
  }
  renderQueue();
  if (candidateId) {
    selectCandidate(candidateId, true);
  } else {
    clearWorkspace();
  }
}

function syncFilterButtons(container, value) {
  container.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.value === value);
  });
}

function hydrateReview(candidate) {
  state.people = [];
  state.selectedUid = null;
  state.uncertainties = [];
  state.qualityTags = new Set();
  state.contractConfirmed = false;

  if (candidate.saved_target) {
    const findingsByPerson = new Map();
    (candidate.saved_target.findings || []).forEach((finding) => {
      if (!findingsByPerson.has(finding.person_id)) findingsByPerson.set(finding.person_id, {});
      findingsByPerson.get(finding.person_id)[finding.violation] = finding;
    });
    const savedBoxes = candidate.saved_review?.person_boxes;
    if (Array.isArray(savedBoxes) && savedBoxes.length) {
      savedBoxes.forEach((box, index) => {
        state.people.push(makePerson(box, findingsByPerson.get(`p${index + 1}`) || {}));
      });
    } else {
      const grouped = new Map();
      (candidate.saved_target.findings || []).forEach((finding) => {
        if (!grouped.has(finding.person_id)) grouped.set(finding.person_id, finding.person_box);
      });
      grouped.forEach((box, personId) => {
        state.people.push(makePerson(box, findingsByPerson.get(personId) || {}));
      });
    }
    state.uncertainties = [...(candidate.saved_target.uncertainties || [])];
    state.qualityTags = new Set(candidate.saved_review?.quality_tags || []);
    state.contractConfirmed = candidate.saved_review?.person_box_contract_confirmed === true;
  } else {
    candidate.source_annotations
      .filter((annotation) => annotation.class_name === "person")
      .forEach((annotation) => state.people.push(makePerson(annotation.box)));
  }
  state.selectedUid = orderedPeople()[0]?.uid || null;
  elements.uncertainties.value = state.uncertainties.join("\n");
  elements.contractConfirmation.checked = state.contractConfirmed;
  syncApprovalState();
}

function selectCandidate(candidateId, force = false) {
  if (candidateId === state.currentId && !force) return;
  if (state.dirty && !force && !window.confirm("当前样本有未保存修改，确定离开吗？")) return;
  const candidate = state.snapshot.candidates.find((item) => item.id === candidateId);
  if (!candidate) return;
  state.currentId = candidateId;
  state.addMode = false;
  state.pointer = null;
  hydrateReview(candidate);
  setDirty(false);
  elements.candidateContext.textContent = `02 / ${candidate.source_split.toUpperCase()} / ${candidate.status.toUpperCase()}`;
  elements.candidateTitle.textContent = candidate.id;
  const sourceSummary = candidate.blind
    ? "来源信号已隐藏"
    : `${candidate.source_classes.length} 个来源标注 · ${candidate.source_classes.join(" / ") || "无类别"}`;
  elements.candidateMeta.textContent = `${candidate.source_group} · ${sourceSummary}`;
  elements.outputPath.textContent = `${state.snapshot.output_dir}/${candidate.source_split}.jsonl`;
  elements.blindBanner.hidden = !candidate.blind;
  elements.sourceToggle.disabled = candidate.blind;
  elements.sourceToggle.checked = !candidate.blind;
  elements.sourceToggleLabel.classList.toggle("is-disabled", candidate.blind);
  elements.emptyInspector.hidden = true;
  elements.reviewForm.hidden = false;
  elements.stageEmpty.hidden = true;
  elements.mediaLayer.hidden = true;
  elements.imageLoading.hidden = false;
  elements.addPerson.disabled = false;
  elements.deletePerson.disabled = !state.selectedUid;
  elements.canvasHint.textContent = candidate.blind
    ? "Test 盲审：从原图手工绘制人物框"
    : "蓝色框为 YOLO 来源标注，黄色框为人工人物框";
  renderQualityTags();
  renderPersonList();
  updateRecordSummary();
  renderQueue();

  const loadingId = candidate.id;
  elements.image.onload = () => {
    if (state.currentId !== loadingId) return;
    elements.imageLoading.hidden = true;
    elements.mediaLayer.hidden = false;
    resizeCanvas();
  };
  elements.image.onerror = () => {
    if (state.currentId !== loadingId) return;
    elements.imageLoading.hidden = true;
    elements.stageEmpty.hidden = false;
    showToast("候选图片读取失败", true);
  };
  elements.image.src = `${candidate.image_url}?v=${Date.now()}`;
}

function clearWorkspace() {
  state.currentId = null;
  state.people = [];
  state.selectedUid = null;
  state.dirty = false;
  state.contractConfirmed = false;
  elements.contractConfirmation.checked = false;
  syncApprovalState();
  elements.candidateTitle.textContent = "当前筛选下没有待处理样本";
  elements.candidateMeta.textContent = "切换左侧筛选条件继续审核";
  elements.stageEmpty.hidden = false;
  elements.mediaLayer.hidden = true;
  elements.imageLoading.hidden = true;
  elements.emptyInspector.hidden = false;
  elements.reviewForm.hidden = true;
  elements.blindBanner.hidden = true;
  elements.addPerson.disabled = true;
  elements.deletePerson.disabled = true;
  setDirty(false);
  renderQueue();
}

function renderQualityTags() {
  elements.qualityTags.replaceChildren();
  state.snapshot.quality_tags.forEach((tag) => {
    const label = document.createElement("label");
    label.className = "quality-tag";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = tag;
    input.checked = state.qualityTags.has(tag);
    const caption = document.createElement("span");
    caption.textContent = TAG_LABELS[tag] || tag;
    input.addEventListener("change", () => {
      if (input.checked) state.qualityTags.add(tag);
      else state.qualityTags.delete(tag);
      setDirty();
    });
    label.append(input, caption);
    elements.qualityTags.append(label);
  });
}

function formatBox(box) {
  return box.map(Math.round).join(" · ");
}

function renderPersonList() {
  const people = orderedPeople();
  elements.personCount.textContent = `${people.length} 人`;
  elements.personList.replaceChildren();
  elements.deletePerson.disabled = !state.selectedUid;
  if (!people.length) {
    const empty = document.createElement("div");
    empty.className = "person-empty";
    empty.textContent = "尚未有人物框。点击图像上方“绘制人物框”，然后在人物可见区域拖拽。";
    elements.personList.append(empty);
    drawCanvas();
    updateRecordSummary();
    return;
  }

  people.forEach((person, index) => {
    const card = document.createElement("article");
    card.className = `person-card${person.uid === state.selectedUid ? " is-selected" : ""}`;
    const head = document.createElement("button");
    head.type = "button";
    head.className = "person-card-head";
    const title = document.createElement("strong");
    title.textContent = `P${index + 1}`;
    const coordinates = document.createElement("span");
    coordinates.textContent = formatBox(person.box);
    head.append(title, coordinates);
    head.addEventListener("click", () => {
      state.selectedUid = person.uid;
      renderPersonList();
    });
    const body = document.createElement("div");
    body.className = "person-card-body";

    Object.entries(VIOLATIONS).forEach(([name, config]) => {
      const finding = person.findings[name];
      const editor = document.createElement("div");
      editor.className = "violation-editor";
      const toggle = document.createElement("label");
      toggle.className = "violation-toggle";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = finding.enabled;
      const copy = document.createElement("strong");
      copy.textContent = config.label;
      const code = document.createElement("span");
      code.textContent = name;
      toggle.append(checkbox, copy, code);
      const fields = document.createElement("div");
      fields.className = "finding-fields";
      fields.hidden = !finding.enabled;
      const evidence = document.createElement("input");
      evidence.type = "text";
      evidence.value = finding.evidence;
      evidence.setAttribute("aria-label", `${config.label}证据描述`);
      const confidence = document.createElement("select");
      confidence.setAttribute("aria-label", `${config.label}置信度`);
      ["low", "medium", "high"].forEach((level) => {
        const option = document.createElement("option");
        option.value = level;
        option.textContent = { low: "低", medium: "中", high: "高" }[level];
        option.selected = finding.confidence === level;
        confidence.append(option);
      });
      checkbox.addEventListener("change", () => {
        finding.enabled = checkbox.checked;
        fields.hidden = !finding.enabled;
        setDirty();
        drawCanvas();
        updateRecordSummary();
      });
      evidence.addEventListener("input", () => {
        finding.evidence = evidence.value;
        setDirty();
      });
      confidence.addEventListener("change", () => {
        finding.confidence = confidence.value;
        setDirty();
      });
      fields.append(evidence, confidence);
      editor.append(toggle, fields);
      body.append(editor);
    });
    card.append(head, body);
    elements.personList.append(card);
  });
  drawCanvas();
  updateRecordSummary();
}

function buildTarget() {
  const findings = [];
  orderedPeople().forEach((person, index) => {
    Object.keys(VIOLATIONS).forEach((violation) => {
      const item = person.findings[violation];
      if (!item.enabled) return;
      findings.push({
        person_id: `p${index + 1}`,
        person_box: person.box.map((value) => Math.round(value)),
        violation,
        evidence: item.evidence.trim(),
        confidence: item.confidence,
      });
    });
  });
  const uncertainties = elements.uncertainties.value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
  return {
    findings,
    uncertainties,
    recommended_action: findings.length || uncertainties.length ? "human_review" : "no_action",
  };
}

function updateRecordSummary() {
  if (!state.currentId) return;
  const target = buildTarget();
  const peopleWithFindings = new Set(target.findings.map((finding) => finding.person_id)).size;
  elements.recordSummary.textContent = target.findings.length
    ? `${state.people.length} 人 · ${peopleWithFindings} 人存在违规 · ${target.findings.length} 条 finding · ${target.uncertainties.length} 条不确定性`
    : `合规样本 · ${state.people.length} 个人物框 · ${target.uncertainties.length} 条不确定性`;
}

function validateBeforeSave(target) {
  if (!elements.reviewer.value.trim()) return "请先填写审核人姓名。";
  if (!state.people.length) return "至少绘制一个完整可见人物框；没有可审核人物时请排除脏数据。";
  if (!state.contractConfirmed) return "请确认所有人物框都覆盖完整可见人体，而不是头部或反光衣。";
  for (const finding of target.findings) {
    if (!finding.evidence) return `${finding.person_id} / ${finding.violation} 缺少证据描述。`;
  }
  return null;
}

async function submitDecision(payload) {
  setBusy(true);
  try {
    const response = await fetch("/api/review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `写入失败 (${response.status})`);
    return result;
  } finally {
    setBusy(false);
  }
}

async function approveCurrent(event) {
  event?.preventDefault();
  if (!state.currentId || state.busy) return;
  const target = buildTarget();
  const error = validateBeforeSave(target);
  if (error) {
    showToast(error, true);
    return;
  }
  const previousIndex = Math.max(0, state.filtered.findIndex((item) => item.id === state.currentId));
  try {
    const snapshot = await submitDecision({
      candidate_id: state.currentId,
      decision: "approve",
      reviewer: elements.reviewer.value.trim(),
      quality_tags: [...state.qualityTags],
      person_boxes: orderedPeople().map((person) => person.box.map((value) => Math.round(value))),
      person_box_contract_confirmed: state.contractConfirmed,
      target,
    });
    localStorage.setItem("visionguard-reviewer", elements.reviewer.value.trim());
    state.snapshot = snapshot;
    state.dirty = false;
    showToast(`已写入 ${currentCandidate()?.source_split || state.split}.jsonl`);
    renderSnapshot({ preferredIndex: previousIndex });
  } catch (error) {
    showToast(error.message, true);
  }
}

async function excludeCurrent() {
  if (!state.currentId || state.busy) return;
  const reason = elements.excludeReason.value.trim();
  if (!reason) {
    showToast("请填写排除原因。", true);
    return;
  }
  if (!elements.reviewer.value.trim()) {
    showToast("请先填写审核人姓名。", true);
    return;
  }
  const previousIndex = Math.max(0, state.filtered.findIndex((item) => item.id === state.currentId));
  try {
    const snapshot = await submitDecision({
      candidate_id: state.currentId,
      decision: "exclude",
      reviewer: elements.reviewer.value.trim(),
      quality_tags: [...state.qualityTags],
      reason,
    });
    localStorage.setItem("visionguard-reviewer", elements.reviewer.value.trim());
    state.snapshot = snapshot;
    state.dirty = false;
    elements.excludeDialog.close();
    elements.excludeReason.value = "";
    showToast("已排除样本并写入 review_log.jsonl");
    renderSnapshot({ preferredIndex: previousIndex });
  } catch (error) {
    showToast(error.message, true);
  }
}

function navigate(offset) {
  const index = state.filtered.findIndex((item) => item.id === state.currentId);
  const next = state.filtered[index + offset];
  if (next) selectCandidate(next.id);
}

function resizeCanvas() {
  if (elements.mediaLayer.hidden || !elements.image.clientWidth) return;
  const ratio = window.devicePixelRatio || 1;
  const width = elements.image.clientWidth;
  const height = elements.image.clientHeight;
  elements.canvas.width = Math.round(width * ratio);
  elements.canvas.height = Math.round(height * ratio);
  elements.canvas.style.width = `${width}px`;
  elements.canvas.style.height = `${height}px`;
  drawCanvas();
}

function canvasMetrics() {
  const rect = elements.canvas.getBoundingClientRect();
  return { rect, width: rect.width, height: rect.height, ratio: window.devicePixelRatio || 1 };
}

function normalizedPoint(event) {
  const { rect } = canvasMetrics();
  return {
    x: Math.max(0, Math.min(1000, ((event.clientX - rect.left) / rect.width) * 1000)),
    y: Math.max(0, Math.min(1000, ((event.clientY - rect.top) / rect.height) * 1000)),
  };
}

function boxToPixels(box, width, height) {
  return [
    (box[0] / 1000) * width,
    (box[1] / 1000) * height,
    (box[2] / 1000) * width,
    (box[3] / 1000) * height,
  ];
}

function drawLabel(context, text, x, y, color) {
  context.font = "700 10px SFMono-Regular, monospace";
  const width = context.measureText(text).width + 10;
  const top = Math.max(0, y - 20);
  context.fillStyle = color;
  context.fillRect(x, top, width, 18);
  context.fillStyle = "#090b0b";
  context.fillText(text, x + 5, top + 12);
}

function drawBox(context, box, color, width, height, options = {}) {
  const [x1, y1, x2, y2] = boxToPixels(box, width, height);
  context.save();
  context.strokeStyle = color;
  context.lineWidth = options.lineWidth || 2;
  if (options.dashed) context.setLineDash([6, 4]);
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  context.restore();
  if (options.label) drawLabel(context, options.label, x1, y1, color);
  if (options.handles) {
    [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([x, y]) => {
      context.fillStyle = "#090b0b";
      context.fillRect(x - 5, y - 5, 10, 10);
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.strokeRect(x - 5, y - 5, 10, 10);
    });
  }
}

function drawCanvas() {
  if (!elements.canvas.width) return;
  const candidate = currentCandidate();
  if (!candidate) return;
  const { width, height, ratio } = canvasMetrics();
  const context = elements.canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  if (elements.sourceToggle.checked && !candidate.blind) {
    candidate.source_annotations.forEach((annotation) => {
      const color = annotation.class_name === "person"
        ? "#7fa9ff"
        : annotation.class_name.startsWith("no_")
          ? "#ff7b6b"
          : "#75d7a1";
      drawBox(context, annotation.box, color, width, height, {
        dashed: true,
        lineWidth: 1.5,
        label: `YOLO · ${annotation.class_name}`,
      });
    });
  }

  orderedPeople().forEach((person, index) => {
    const active = person.uid === state.selectedUid;
    const violations = Object.entries(person.findings)
      .filter(([, finding]) => finding.enabled)
      .map(([name]) => name.replace("no_", "−"));
    drawBox(context, person.box, active ? "#ffe170" : "#f1c84b", width, height, {
      lineWidth: active ? 3 : 2,
      handles: active,
      label: `P${index + 1}${violations.length ? ` · ${violations.join(" · ")}` : ""}`,
    });
  });

  if (state.pointer?.mode === "draw") {
    drawBox(context, state.pointer.box, "#f1c84b", width, height, { dashed: true, lineWidth: 2 });
  }
}

function hitTest(point) {
  const { width, height } = canvasMetrics();
  const toleranceX = (10 / width) * 1000;
  const toleranceY = (10 / height) * 1000;
  const selected = state.people.find((person) => person.uid === state.selectedUid);
  if (selected) {
    const [x1, y1, x2, y2] = selected.box;
    const corners = { nw: [x1, y1], ne: [x2, y1], sw: [x1, y2], se: [x2, y2] };
    for (const [corner, [x, y]] of Object.entries(corners)) {
      if (Math.abs(point.x - x) <= toleranceX && Math.abs(point.y - y) <= toleranceY) {
        return { person: selected, mode: "resize", corner };
      }
    }
  }
  const people = [...state.people].reverse();
  const person = people.find(({ box }) => point.x >= box[0] && point.x <= box[2] && point.y >= box[1] && point.y <= box[3]);
  return person ? { person, mode: "move" } : null;
}

function startPointer(event) {
  if (!state.currentId) return;
  const point = normalizedPoint(event);
  elements.canvas.setPointerCapture(event.pointerId);
  if (state.addMode) {
    state.pointer = { mode: "draw", start: point, box: [point.x, point.y, point.x, point.y] };
    drawCanvas();
    return;
  }
  const hit = hitTest(point);
  if (!hit) {
    state.selectedUid = null;
    renderPersonList();
    return;
  }
  state.selectedUid = hit.person.uid;
  state.pointer = {
    mode: hit.mode,
    corner: hit.corner,
    start: point,
    person: hit.person,
    original: [...hit.person.box],
  };
  renderPersonList();
}

function movePointer(event) {
  if (!state.pointer) return;
  const point = normalizedPoint(event);
  if (state.pointer.mode === "draw") {
    state.pointer.box = [
      Math.min(state.pointer.start.x, point.x),
      Math.min(state.pointer.start.y, point.y),
      Math.max(state.pointer.start.x, point.x),
      Math.max(state.pointer.start.y, point.y),
    ];
  } else if (state.pointer.mode === "move") {
    const deltaX = point.x - state.pointer.start.x;
    const deltaY = point.y - state.pointer.start.y;
    const [x1, y1, x2, y2] = state.pointer.original;
    const width = x2 - x1;
    const height = y2 - y1;
    const nextX = Math.max(0, Math.min(1000 - width, x1 + deltaX));
    const nextY = Math.max(0, Math.min(1000 - height, y1 + deltaY));
    state.pointer.person.box = [nextX, nextY, nextX + width, nextY + height];
  } else if (state.pointer.mode === "resize") {
    const [x1, y1, x2, y2] = state.pointer.original;
    const box = [x1, y1, x2, y2];
    if (state.pointer.corner.includes("n")) box[1] = Math.min(point.y, y2 - 20);
    if (state.pointer.corner.includes("s")) box[3] = Math.max(point.y, y1 + 20);
    if (state.pointer.corner.includes("w")) box[0] = Math.min(point.x, x2 - 20);
    if (state.pointer.corner.includes("e")) box[2] = Math.max(point.x, x1 + 20);
    state.pointer.person.box = box.map((value) => Math.max(0, Math.min(1000, value)));
  }
  drawCanvas();
}

function endPointer(event) {
  if (!state.pointer) return;
  const pointer = state.pointer;
  state.pointer = null;
  if (elements.canvas.hasPointerCapture(event.pointerId)) elements.canvas.releasePointerCapture(event.pointerId);
  if (pointer.mode === "draw") {
    if (pointer.box[2] - pointer.box[0] >= 20 && pointer.box[3] - pointer.box[1] >= 20) {
      const person = makePerson(pointer.box);
      state.people.push(person);
      state.selectedUid = person.uid;
      invalidateGeometryContract();
      setDirty();
    }
    state.addMode = false;
    elements.addPerson.classList.remove("is-active");
  } else {
    invalidateGeometryContract();
    setDirty();
  }
  renderPersonList();
}

function deleteSelectedPerson() {
  if (!state.selectedUid) return;
  state.people = state.people.filter((person) => person.uid !== state.selectedUid);
  state.selectedUid = orderedPeople()[0]?.uid || null;
  invalidateGeometryContract();
  setDirty();
  renderPersonList();
}

async function loadSnapshot() {
  elements.systemState.textContent = "正在读取审核队列";
  try {
    const response = await fetch("/api/review/snapshot");
    const snapshot = await response.json();
    if (!response.ok) throw new Error(snapshot.error || "无法读取审核队列");
    state.snapshot = snapshot;
    elements.reviewer.value = localStorage.getItem("visionguard-reviewer") || snapshot.reviewer || "";
    syncAvailableSplits();
    renderSnapshot();
    elements.systemState.textContent = "本地审核服务已连接";
  } catch (error) {
    elements.systemState.textContent = "审核服务不可用";
    showToast(error.message, true);
  }
}

elements.splitFilter.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-value]");
  if (!button) return;
  state.split = button.dataset.value;
  state.category = "all";
  elements.categoryFilter.value = "all";
  syncFilterButtons(elements.splitFilter, state.split);
  renderSnapshot();
});

elements.statusFilter.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-value]");
  if (!button) return;
  state.status = button.dataset.value;
  syncFilterButtons(elements.statusFilter, state.status);
  renderSnapshot();
});

elements.categoryFilter.addEventListener("change", () => {
  state.category = elements.categoryFilter.value;
  renderSnapshot();
});

elements.refresh.addEventListener("click", async () => {
  if (state.dirty && !window.confirm("刷新会丢弃当前未保存修改，确定继续吗？")) return;
  await loadSnapshot();
});
elements.previous.addEventListener("click", () => navigate(-1));
elements.next.addEventListener("click", () => navigate(1));
elements.addPerson.addEventListener("click", () => {
  state.addMode = !state.addMode;
  elements.addPerson.classList.toggle("is-active", state.addMode);
  elements.canvas.style.cursor = state.addMode ? "crosshair" : "default";
});
elements.deletePerson.addEventListener("click", deleteSelectedPerson);
elements.sourceToggle.addEventListener("change", drawCanvas);
elements.canvas.addEventListener("pointerdown", startPointer);
elements.canvas.addEventListener("pointermove", movePointer);
elements.canvas.addEventListener("pointerup", endPointer);
elements.canvas.addEventListener("pointercancel", endPointer);
window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe(elements.mediaLayer);

elements.markCompliant.addEventListener("click", () => {
  state.people.forEach((person) => {
    Object.values(person.findings).forEach((finding) => { finding.enabled = false; });
  });
  elements.uncertainties.value = "";
  setDirty();
  renderPersonList();
  showToast("已清除违规发现；提交后作为合规样本入库");
});
elements.uncertainties.addEventListener("input", () => {
  setDirty();
  updateRecordSummary();
});
elements.contractConfirmation.addEventListener("change", () => {
  state.contractConfirmed = elements.contractConfirmation.checked;
  setDirty();
  syncApprovalState();
});
elements.reviewForm.addEventListener("submit", approveCurrent);
elements.exclude.addEventListener("click", () => {
  if (!elements.reviewer.value.trim()) {
    showToast("请先填写审核人姓名。", true);
    return;
  }
  elements.excludeDialog.showModal();
  elements.excludeReason.focus();
});
elements.cancelExclude.addEventListener("click", () => elements.excludeDialog.close());
elements.excludeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  excludeCurrent();
});
elements.reviewer.addEventListener("change", () => {
  localStorage.setItem("visionguard-reviewer", elements.reviewer.value.trim());
});

document.addEventListener("keydown", (event) => {
  const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    approveCurrent();
    return;
  }
  if (editing || elements.excludeDialog.open) return;
  if (event.key.toLowerCase() === "n") navigate(1);
  if (event.key.toLowerCase() === "p") navigate(-1);
  if (event.key.toLowerCase() === "b" && state.currentId) elements.addPerson.click();
  if ((event.key === "Backspace" || event.key === "Delete") && state.selectedUid) deleteSelectedPerson();
});

loadSnapshot();
