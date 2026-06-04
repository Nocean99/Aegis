const reportList = document.querySelector("#reportList");
const metrics = document.querySelector("#metrics");
const activeReport = document.querySelector("#activeReport");
const missionText = document.querySelector("#missionText");
const scorerMeta = document.querySelector("#scorerMeta");
const visionPlan = document.querySelector("#visionPlan");
const candidateGrid = document.querySelector("#candidateGrid");
const candidateFilter = document.querySelector("#candidateFilter");
const refreshReports = document.querySelector("#refreshReports");

let selectedReportPath = null;
let selectedPayload = null;

refreshReports.addEventListener("click", loadReports);
candidateFilter.addEventListener("change", renderCandidates);

async function loadReports() {
  const response = await fetch("/api/reports");
  const data = await response.json();
  reportList.innerHTML = "";
  for (const report of data.reports) {
    const button = document.createElement("button");
    button.className = "report-button";
    button.type = "button";
    button.innerHTML = `
      <strong>${escapeHtml(report.mission_request || "Untitled mission")}</strong>
      <span>${escapeHtml(report.timestamp || "")}</span>
      <span>Precision ${display(report.precision)} · Recall ${display(report.recall)} · ${display(report.detections)} detections</span>
    `;
    button.addEventListener("click", () => loadReport(report.path));
    reportList.appendChild(button);
  }
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
  const summary = report.summary || {};
  const evaluation = report.evaluation || {};
  activeReport.textContent = selectedPayload.path;
  missionText.textContent = report.mission_request || "No mission request";
  scorerMeta.textContent = `${report.proposal_mode || "unknown"} · ${report.scorer || "unknown"}`;
  metrics.innerHTML = [
    metric("Processed", summary.processed),
    metric("Detections", summary.detections),
    metric("Shortlist", summary.shortlist_count),
    metric("Precision", evaluation.precision),
    metric("Recall", evaluation.recall),
    metric("F1", evaluation.f1),
    metric("False Neg", evaluation.false_negative),
  ].join("");
  renderVisionPlan(report.vision_plan || {});
  renderCandidates();
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

function renderCandidates() {
  if (!selectedPayload) {
    candidateGrid.innerHTML = `<p class="empty">Select a report to review candidates.</p>`;
    return;
  }
  const report = selectedPayload.report;
  const evaluation = report.evaluation || {};
  const summary = report.summary || {};
  let items = [];
  if (candidateFilter.value === "false_positive") items = evaluation.false_positives || [];
  else if (candidateFilter.value === "false_negative") items = evaluation.false_negatives || [];
  else if (candidateFilter.value === "all") items = (report.results || []).filter((item) => item.detected || item.full_frame_semantic);
  else items = summary.shortlist || [];

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
  const imagePath = item.debug_path || item.crop_path || item.image_path;
  const image = imagePath
    ? `<img src="/api/file?path=${encodeURIComponent(imagePath)}" alt="${escapeHtml(fileName(imagePath))}" loading="lazy" />`
    : `<div class="image-missing">No image</div>`;
  card.innerHTML = `
    ${image}
    <div class="candidate-body">
      <h3>${escapeHtml(fileName(item.image_path || ""))}</h3>
      <p class="review-note">${escapeHtml(item.label?.label || "unlabeled")} ${review.status ? `· reviewed: ${escapeHtml(review.status)}` : ""}</p>
      <p class="review-note">${decisionMeaning(item.decision || item.final_decision || item.semantic?.decision)}</p>
      <div class="candidate-meta">
        <span>Score <strong>${display(item.score ?? item.final_score ?? item.semantic?.score)}</strong></span>
        <span>Review priority <strong>${display(item.review_priority)}</strong></span>
        <span>Decision <strong>${escapeHtml(item.decision || item.final_decision || item.semantic?.decision || "n/a")}</strong></span>
        <span>Detector <strong>${display(item.detector_confidence)}</strong></span>
        <span>BBox <strong>${escapeHtml(JSON.stringify(item.bbox || []))}</strong></span>
      </div>
      <textarea placeholder="Review notes">${escapeHtml(review.notes || "")}</textarea>
      <div class="review-actions">
        <button type="button" data-status="approved">Approve</button>
        <button type="button" data-status="rejected">Reject</button>
        <button type="button" data-status="investigate">Investigate</button>
      </div>
    </div>
  `;
  card.querySelectorAll("button[data-status]").forEach((button) => {
    button.addEventListener("click", async () => {
      const notes = card.querySelector("textarea").value;
      await saveReview(key, button.dataset.status, notes);
    });
  });
  return card;
}

async function saveReview(candidateKeyValue, status, notes) {
  const response = await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      report_path: selectedReportPath,
      candidate_key: candidateKeyValue,
      status,
      notes,
    }),
  });
  const data = await response.json();
  if (data.ok) {
    selectedPayload.reviews = data.reviews;
    renderCandidates();
  }
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${display(value)}</strong></div>`;
}

function candidateKey(item) {
  return `${item.image_path || ""}::${item.frame_index ?? ""}`;
}

function fileName(path) {
  return String(path).split("/").pop();
}

function display(value) {
  if (value === undefined || value === null) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return escapeHtml(value);
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
