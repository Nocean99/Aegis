const reportList = document.querySelector("#reportList");
const metrics = document.querySelector("#metrics");
const activeReport = document.querySelector("#activeReport");
const missionText = document.querySelector("#missionText");
const scorerMeta = document.querySelector("#scorerMeta");
const visionPlan = document.querySelector("#visionPlan");
const candidateGrid = document.querySelector("#candidateGrid");
const candidateFilter = document.querySelector("#candidateFilter");
const refreshReports = document.querySelector("#refreshReports");
const refreshMemory = document.querySelector("#refreshMemory");
const missionMemory = document.querySelector("#missionMemory");
const missionPlanForm = document.querySelector("#missionPlanForm");
const missionRequest = document.querySelector("#missionRequest");
const operatingMode = document.querySelector("#operatingMode");
const missionPlanResult = document.querySelector("#missionPlanResult");
const reviewPanelTitle = document.querySelector("#reviewPanelTitle");

let selectedReportPath = null;
let selectedPayload = null;

refreshReports.addEventListener("click", loadReports);
refreshMemory.addEventListener("click", loadMissionMemory);
candidateFilter.addEventListener("change", renderCandidates);
missionPlanForm.addEventListener("submit", planMission);

async function planMission(event) {
  event.preventDefault();
  const request = missionRequest.value.trim();
  if (!request) {
    missionPlanResult.innerHTML = `<p class="empty">Enter a mission request first.</p>`;
    return;
  }
  missionPlanResult.innerHTML = `<p class="empty">Planning mission...</p>`;
  const response = await fetch("/api/mission-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mission_request: request,
      operating_mode: operatingMode.value,
    }),
  });
  const data = await response.json();
  if (!data.ok) {
    missionPlanResult.innerHTML = `<p class="empty">${escapeHtml(data.error || "Could not create mission plan")}</p>`;
    return;
  }
  renderMissionPlan(data);
}

async function loadReports() {
  const response = await fetch("/api/reports");
  const data = await response.json();
  reportList.innerHTML = "";
  data.reports.forEach((report, index) => {
    const button = document.createElement("button");
    button.className = `report-button${index === 0 ? " newest-report" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${index === 0 ? '<span class="latest-badge">Newest</span> ' : ""}${escapeHtml(report.mission_request || "Untitled mission")}</strong>
      <span>${escapeHtml(report.type || "vision")} · updated ${escapeHtml(formatDateTime(report.updated_at || report.timestamp || ""))}</span>
      <span class="report-path">${escapeHtml(report.path || "")}</span>
      <span>Confirmed ${display(report.precision)} · Capture ${display(report.capture_recall)} · ${display(report.detections)} detections</span>
    `;
    button.addEventListener("click", () => loadReport(report.path));
    reportList.appendChild(button);
  });
  if (!data.reports.length) {
    reportList.innerHTML = `<p class="empty">No vision reports found yet.</p>`;
  }
}

async function loadReport(path) {
  selectedReportPath = path;
  const response = await fetch(`/api/report?path=${encodeURIComponent(path)}`);
  selectedPayload = await response.json();
  if (!selectedPayload.ok) {
    activeReport.textContent = selectedPayload.error || "Could not load report";
    return;
  }
  renderReport();
}

function renderReport() {
  const report = selectedPayload.report;
  const isAcoustic = report.sensor_modality === "acoustic";
  const summary = report.summary || {};
  const evaluation = report.evaluation || {};
  activeReport.textContent = selectedPayload.path;
  reviewPanelTitle.textContent = isAcoustic ? "Acoustic Review" : "Candidate Review";
  missionText.textContent = report.mission_request || "No mission request";
  scorerMeta.textContent = isAcoustic ? "acoustic · local-acoustic-proposal-v1" : `${report.proposal_mode || "unknown"} · ${report.scorer || "unknown"}`;
  const capture = evaluation.analyst_capture || {};
  metrics.innerHTML = [
    metric("Processed", summary.processed),
    metric(isAcoustic ? "Proposals" : "Detections", isAcoustic ? summary.candidate_count : summary.detections),
    metric("Shortlist", isAcoustic ? summary.candidate_count : summary.shortlist_count),
    metric(isAcoustic ? "Capture Precision" : "Confirmed Precision", isAcoustic ? evaluation.capture_precision : evaluation.precision),
    metric(isAcoustic ? "Capture Recall" : "Confirmed Recall", isAcoustic ? evaluation.capture_recall : evaluation.recall),
    metric(isAcoustic ? "Capture F1" : "Confirmed F1", isAcoustic ? evaluation.capture_f1 : evaluation.f1),
    metric("Capture Precision", isAcoustic ? evaluation.capture_precision : capture.precision),
    metric("Capture Recall", isAcoustic ? evaluation.capture_recall : capture.recall),
    metric("Capture F1", isAcoustic ? evaluation.capture_f1 : capture.f1),
    metric("False Neg", isAcoustic ? evaluation.false_negative : evaluation.false_negative),
  ].join("");
  if (isAcoustic) renderAcousticPlan(report);
  else renderVisionPlan(report.vision_plan || {});
  renderCandidates();
}

function renderAcousticPlan(report) {
  const metadata = report.metadata || [];
  visionPlan.innerHTML = [
    `Modality: acoustic`,
    `Audio files: ${metadata.length}`,
    `Spectrograms: ${report.summary?.spectrogram_count ?? 0}`,
    `Candidates: ${report.summary?.candidate_count ?? 0}`,
  ]
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("");
}

function renderVisionPlan(plan) {
  const values = [
    ["Colors", plan.important_colors],
    ["Categories", plan.possible_categories],
    ["Context", plan.context_hints],
    ["Modes", plan.proposal_modes],
  ];
  visionPlan.innerHTML = values
    .flatMap(([label, items]) => (items && items.length ? items.map((item) => `${label}: ${item}`) : [`${label}: none`]))
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("");
}

function renderMissionPlan(data) {
  const command = data.command || {};
  const objective = command.objective || {};
  const vision = data.vision_plan || {};
  const context = data.contextual_search_plan || {};
  const linkLoss = command.link_loss_policy || {};
  missionPlanResult.innerHTML = `
    <div class="plan-grid">
      ${planBlock("Objective", [
        ["Target", objective.target_description || "unspecified"],
        ["Area", objective.search_area_description || "unspecified"],
        ["Urgency", objective.urgency || "normal"],
      ])}
      ${planBlock("Operating Policy", [
        ["Mode", command.operating_mode || "n/a"],
        ["Confirmation", command.confirmation_mode || "n/a"],
        ["Link loss", linkLoss.action || "n/a"],
      ])}
      ${planBlock("Sensor Strategy", [
        ["Proposal modes", joinList(vision.proposal_modes)],
        ["Colors", joinList(vision.important_colors)],
        ["Categories", joinList(vision.possible_categories)],
      ])}
      ${priorityBlock("Search Priorities", context.likely_locations || [])}
      ${planBlock("Routing Guidance", (context.routing_guidance || []).map((item, index) => [String(index + 1), item]))}
      ${planBlock("Next Actions", (data.next_actions || []).map((item, index) => [String(index + 1), item]))}
    </div>
  `;
}

function planBlock(title, rows) {
  return `
    <section class="plan-block">
      <h3>${escapeHtml(title)}</h3>
      ${rows
        .map(([label, value]) => `
          <div class="plan-row">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
          </div>
        `)
        .join("")}
    </section>
  `;
}

function priorityBlock(title, items) {
  const rows = items.length
    ? items.slice(0, 5).map((item) => [
        `${item.priority ?? "n/a"} · ${item.name || "Priority zone"}`,
        `${item.rationale || ""} Cues: ${joinList(item.cues)}`,
      ])
    : [["None", "No contextual priorities inferred yet."]];
  return planBlock(title, rows);
}

function renderCandidates() {
  if (!selectedPayload) {
    candidateGrid.innerHTML = `<p class="empty">Select a report to review candidates.</p>`;
    return;
  }
  const report = selectedPayload.report;
  const isAcoustic = report.sensor_modality === "acoustic";
  const evaluation = report.evaluation || {};
  const summary = report.summary || {};
  let items = [];
  if (isAcoustic) {
    items = report.candidates || [];
  } else if (candidateFilter.value === "false_positive") items = evaluation.false_positives || [];
  else if (candidateFilter.value === "false_negative") items = evaluation.false_negatives || [];
  else if (candidateFilter.value === "all") items = (report.results || []).filter((item) => item.detected || item.full_frame_semantic);
  else items = summary.shortlist || [];
  items = sortByReviewPriority(items);

  if (!items.length) {
    candidateGrid.innerHTML = `<p class="empty">No candidates in this view.</p>`;
    return;
  }
  candidateGrid.innerHTML = "";
  for (const item of items) {
    candidateGrid.appendChild(candidateCard(item));
  }
}

function candidateCard(item) {
  const card = document.createElement("article");
  card.className = "candidate-card";
  const key = candidateKey(item);
  const review = (selectedPayload.reviews || {})[key] || {};
  const isAcoustic = selectedPayload.report?.sensor_modality === "acoustic" || item.sensor_modality === "acoustic";
  const imagePath = isAcoustic ? item.spectrogram_path : item.debug_path || item.crop_path || item.image_path;
  const image = imagePath
    ? `<img src="/api/file?path=${encodeURIComponent(imagePath)}" alt="${escapeHtml(fileName(imagePath))}" loading="lazy" />`
    : `<div class="image-missing">${isAcoustic ? "No spectrogram" : "No image"}</div>`;
  card.innerHTML = `
    ${image}
    <div class="candidate-body">
      <h3>${escapeHtml(isAcoustic ? item.candidate_id : fileName(item.image_path || ""))}</h3>
      <p class="review-note">${escapeHtml(isAcoustic ? acousticTimeRange(item) : item.label?.label || "unlabeled")} ${review.decision || review.status ? `· reviewed: ${escapeHtml(review.decision || review.status)}` : ""}</p>
      <p class="review-note">${escapeHtml(isAcoustic ? item.proposal_reason || "Acoustic segment proposal" : decisionMeaning(item.decision || item.final_decision || item.semantic?.decision))}</p>
      <p class="review-note">${escapeHtml(reviewReasons(item))}</p>
      <div class="candidate-meta">
        <span>Proposal <strong>${display(item.proposal_score ?? item.candidate_rank?.proposal_score)}</strong></span>
        <span>${isAcoustic ? "RMS" : "Semantic"} <strong>${display(isAcoustic ? item.rms_amplitude : item.semantic_score ?? item.candidate_rank?.semantic_score ?? item.score ?? item.final_score ?? item.semantic?.score)}</strong></span>
        <span>Uncertainty <strong>${display(item.uncertainty_score ?? item.candidate_rank?.uncertainty_score)}</strong></span>
        <span>${isAcoustic ? "Peak" : "Mission relevance"} <strong>${display(isAcoustic ? item.peak_amplitude : item.mission_relevance_score ?? item.candidate_rank?.mission_relevance_score)}</strong></span>
        <span>Review priority <strong>${display(item.review_priority ?? item.candidate_rank?.review_priority)}</strong></span>
        <span>Decision <strong>${escapeHtml(item.decision || item.final_decision || item.semantic?.decision || "n/a")}</strong></span>
        <span>${isAcoustic ? "Duration" : "Detector"} <strong>${display(isAcoustic ? item.duration_s : item.detector_confidence)}</strong></span>
        <span>${isAcoustic ? "Time" : "BBox"} <strong>${escapeHtml(isAcoustic ? acousticTimeRange(item) : JSON.stringify(item.bbox || []))}</strong></span>
      </div>
      <select class="reason-tag" aria-label="Reason tag">
        ${reasonTagOptions(review.reason_tag || "")}
      </select>
      <input class="reason-input" placeholder="Decision reason, e.g. shoreline debris" value="${escapeHtml(review.reason || "")}" />
      <textarea placeholder="Review notes">${escapeHtml(review.notes || "")}</textarea>
      <div class="review-actions">
        <button type="button" data-decision="approve">Approve</button>
        <button type="button" data-decision="reject">Reject</button>
        <button type="button" data-decision="investigate">Investigate</button>
      </div>
    </div>
  `;
  card.querySelectorAll("button[data-decision]").forEach((button) => {
    button.addEventListener("click", async () => {
      const notes = card.querySelector("textarea").value;
      const reason = card.querySelector(".reason-input").value;
      const reasonTag = card.querySelector(".reason-tag").value;
      await saveReview(key, item.candidate_id || key, button.dataset.decision, reasonTag, reason, notes);
    });
  });
  return card;
}

async function saveReview(candidateKeyValue, candidateId, decision, reasonTag, reason, notes) {
  const response = await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      report_path: selectedReportPath,
      candidate_key: candidateKeyValue,
      candidate_id: candidateId,
      decision,
      reason_tag: reasonTag,
      reason,
      notes,
    }),
  });
  const data = await response.json();
  if (data.ok) {
    selectedPayload.reviews = data.reviews;
    renderCandidates();
  }
}

async function loadMissionMemory() {
  missionMemory.innerHTML = `<p class="empty">Loading mission memory...</p>`;
  const response = await fetch("/api/mission-memory");
  const data = await response.json();
  if (!data.ok) {
    missionMemory.innerHTML = `<p class="empty">Could not load mission memory.</p>`;
    return;
  }
  renderMissionMemory(data.memory || {});
}

function renderMissionMemory(memory) {
  const categories = Object.entries(memory.category_metrics || {});
  const recommendations = memory.recommendations || [];
  const memoryV2 = memory.mission_memory_v2 || {};
  const acoustic = memory.acoustic_memory || {};
  missionMemory.innerHTML = `
    ${memoryBlock("Coverage", [
      ["Reports", memory.report_count ?? 0],
      ["Mission types", Object.keys(memory.mission_types || {}).join(", ") || "none"],
      ["Review statuses", formatCounts(memory.review_statuses)],
    ])}
    ${memoryBlock("Repeated Patterns", [
      ["False-positive cues", formatCounts(memory.common_false_positive_terms)],
      ["Miss cues", formatCounts(memory.common_false_negative_terms)],
    ])}
    ${memoryBlock("Analyst Lessons", [
      ["False-positive causes", formatCounts(memoryV2.common_false_positive_causes)],
      ["Confirmed indicators", formatCounts(memoryV2.confirmed_positive_indicators)],
      ["Uncertainty causes", formatCounts(memoryV2.common_uncertainty_causes)],
    ])}
    ${memoryBlock("V2 Lessons", (memoryV2.lessons || []).map((item, index) => [String(index + 1), item]))}
    ${memoryBlock("Acoustic Memory", [
      ["Reports", acoustic.report_count ?? 0],
      ["Candidates", acoustic.candidate_count ?? 0],
      ["False positives", formatCounts(acoustic.recurring_acoustic_false_positives)],
      ["Uncertainty", formatCounts(acoustic.recurring_acoustic_uncertainty)],
      ["Recommended data", joinList(acoustic.recommended_acoustic_data || [])],
    ])}
    ${memoryBlock("Recommendations", recommendations.map((item, index) => [String(index + 1), item]))}
    ${memoryBlock("Category Performance", categories.length ? categories.map(([name, metric]) => [
      name,
      `runs ${metric.runs}, F1 ${display(metric.avg_f1)}, capture ${display(metric.avg_capture_recall)}`,
    ]) : [["None", "No labeled category history yet."]])}
  `;
}

function memoryBlock(title, rows) {
  return `
    <section class="memory-block">
      <h3>${escapeHtml(title)}</h3>
      ${rows.map(([label, value]) => `
        <div class="plan-row">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `).join("")}
    </section>
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${display(value)}</strong></div>`;
}

function candidateKey(item) {
  return item.candidate_id || `${item.image_path || item.audio_path || ""}::${item.frame_index ?? item.start_s ?? ""}`;
}

function acousticTimeRange(item) {
  return `${display(item.start_s)}s-${display(item.end_s)}s`;
}

function fileName(path) {
  return String(path).split("/").pop();
}

function joinList(items) {
  return items && items.length ? items.join(", ") : "none";
}

function reviewReasons(item) {
  const reasons = item.review_reasons || [];
  return reasons.length ? `Priority: ${reasons.join(", ")}` : "Priority: no explanation available";
}

function sortByReviewPriority(items) {
  return [...items].sort((a, b) => Number(b.review_priority ?? b.candidate_rank?.review_priority ?? 0) - Number(a.review_priority ?? a.candidate_rank?.review_priority ?? 0));
}

function formatCounts(counts) {
  const entries = Object.entries(counts || {});
  return entries.length ? entries.slice(0, 5).map(([key, value]) => `${key} ${value}`).join(", ") : "none";
}

function reasonTagOptions(selected) {
  const tags = [
    ["", "Reason tag"],
    ["person_visible", "Person visible"],
    ["vehicle_visible", "Vehicle visible"],
    ["too_small", "Too small"],
    ["vegetation", "Vegetation"],
    ["shadow", "Shadow"],
    ["debris", "Debris"],
    ["rooftop", "Rooftop"],
    ["road_marking", "Road marking"],
    ["building", "Building"],
    ["hot_object", "Hot object"],
    ["thermal_clutter", "Thermal clutter"],
    ["wave_noise", "Wave noise"],
    ["dock_machinery", "Dock machinery"],
    ["wind_noise", "Wind noise"],
    ["low_snr", "Low SNR"],
    ["overlapping_signals", "Overlapping signals"],
    ["vessel_sound", "Vessel sound"],
    ["acoustic_clutter", "Acoustic clutter"],
    ["false_alarm", "False alarm"],
    ["uncertain_vehicle", "Uncertain vehicle"],
  ];
  return tags
    .map(([value, label]) => `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(label)}</option>`)
    .join("");
}

function display(value) {
  if (value === undefined || value === null) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return escapeHtml(value);
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function decisionMeaning(decision) {
  if (decision === "LIKELY_MATCH") return "System says this is a likely mission match.";
  if (decision === "POSSIBLE_MATCH") return "System says this could be a mission match.";
  if (decision === "NEEDS_REVIEW") return "Uncertain. This needs analyst review, not a confirmed match.";
  if (decision === "REJECT") return "System does not consider this a match.";
  return "No decision available.";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadReports();
loadMissionMemory();
