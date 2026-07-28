import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import "./styles.css";
import {
  InputValidationError,
  buildSampleFromCsv,
  predictEnsemble,
} from "./inference.js";

const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const REQUIRED_FILES = ["node_features.csv", "adjacency_area.csv"];
const assetUrl = (path) => `${import.meta.env.BASE_URL}${path}`;

const elements = {
  modelStatus: document.querySelector("#model-status"),
  dropZone: document.querySelector("#drop-zone"),
  chooseFiles: document.querySelector("#choose-files"),
  chooseFolder: document.querySelector("#choose-folder"),
  fileInput: document.querySelector("#file-input"),
  folderInput: document.querySelector("#folder-input"),
  inputSummary: document.querySelector("#input-summary"),
  uploadError: document.querySelector("#upload-error"),
  loadExample: document.querySelector("#load-example"),
  downloadExample: document.querySelector("#download-example"),
  runPrediction: document.querySelector("#run-prediction"),
  resultState: document.querySelector("#result-state"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  predictionValue: document.querySelector("#prediction-value"),
  domainBadge: document.querySelector("#domain-badge"),
  intervalValue: document.querySelector("#interval-value"),
  intervalRange: document.querySelector("#interval-range"),
  intervalCenter: document.querySelector("#interval-center"),
  graphReadout: document.querySelector("#graph-readout"),
  memberSpread: document.querySelector("#member-spread"),
  memberList: document.querySelector("#member-list"),
  domainChecks: document.querySelector("#domain-checks"),
  copyResult: document.querySelector("#copy-result"),
  downloadResult: document.querySelector("#download-result"),
  resultCanvas: document.querySelector("#result-canvas"),
  heroCanvas: document.querySelector("#hero-canvas"),
  heroFrame: document.querySelector("#hero-frame"),
  toast: document.querySelector("#toast"),
};

const state = {
  bundle: null,
  sample: null,
  result: null,
  sourceLabel: null,
  exampleTexts: null,
};

function basename(path) {
  return String(path).split(/[\\/]/).pop().toLowerCase();
}

function parentPath(path) {
  const parts = String(path).split(/[\\/]/);
  parts.pop();
  return parts.join("/");
}

function setModelStatus(mode, text) {
  elements.modelStatus.classList.toggle("ready", mode === "ready");
  elements.modelStatus.classList.toggle("error", mode === "error");
  elements.modelStatus.querySelector("span:last-child").textContent = text;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 2_800);
}

function showUploadError(error) {
  const message =
    error instanceof InputValidationError
      ? error.message
      : "The sample could not be read. Check the two required files and try again.";
  elements.uploadError.textContent = message;
  elements.uploadError.hidden = false;
}

function clearUploadError() {
  elements.uploadError.hidden = true;
  elements.uploadError.textContent = "";
}

function updateContract(loadedFiles = []) {
  for (const fileName of REQUIRED_FILES) {
    const row = document.querySelector(`[data-file="${fileName}"]`);
    const loaded = loadedFiles.includes(fileName);
    row.classList.toggle("loaded", loaded);
    row.querySelector(".contract-state").textContent = loaded ? "Loaded" : "Required";
  }
}

function formatNumber(value, digits = 4) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(digits);
}

function renderInputSummary(sample, sourceLabel) {
  elements.inputSummary.innerHTML = `
    <div><span>Sample</span><strong>${escapeHtml(sourceLabel)}</strong></div>
    <div><span>Nodes</span><strong>${sample.stats.nodeCount}</strong></div>
    <div><span>Edges</span><strong>${sample.stats.edgeCount}</strong></div>
  `;
  elements.inputSummary.hidden = false;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resetResult() {
  state.result = null;
  elements.resultEmpty.hidden = false;
  elements.resultContent.hidden = true;
  elements.resultState.textContent = state.sample ? "READY TO RUN" : "AWAITING INPUT";
  elements.resultState.className = "result-state";
}

function registerSample(nodeText, adjacencyText, sourceLabel) {
  const sample = buildSampleFromCsv(nodeText, adjacencyText);
  state.sample = sample;
  state.sourceLabel = sourceLabel;
  clearUploadError();
  updateContract(REQUIRED_FILES);
  renderInputSummary(sample, sourceLabel);
  elements.runPrediction.disabled = !state.bundle;
  resetResult();
  showToast(`Validated ${sample.stats.nodeCount} nodes and ${sample.stats.edgeCount} edges.`);
  return sample;
}

function matchRequiredEntries(entries) {
  const matches = Object.fromEntries(REQUIRED_FILES.map((name) => [name, []]));
  for (const [path, text] of entries) {
    const name = basename(path);
    if (matches[name]) matches[name].push({ path, text });
  }

  for (const required of REQUIRED_FILES) {
    if (matches[required].length === 0) {
      throw new InputValidationError(`Missing ${required}.`);
    }
    if (matches[required].length > 1) {
      const parents = new Set(matches[required].map(({ path }) => parentPath(path)));
      if (parents.size > 1) {
        throw new InputValidationError(
          "Multiple lattice samples were detected. Upload one sample at a time.",
        );
      }
    }
  }

  return {
    nodeText: matches["node_features.csv"][0].text,
    adjacencyText: matches["adjacency_area.csv"][0].text,
  };
}

async function entriesFromFiles(fileList) {
  const files = Array.from(fileList);
  if (files.length === 0) {
    throw new InputValidationError("Choose a ZIP, folder, or the two required CSV files.");
  }
  if (files.some((file) => file.size > MAX_UPLOAD_BYTES)) {
    throw new InputValidationError("Each uploaded file must be 15 MB or smaller.");
  }

  const zipFiles = files.filter((file) => file.name.toLowerCase().endsWith(".zip"));
  if (zipFiles.length > 1 || (zipFiles.length === 1 && files.length > 1)) {
    throw new InputValidationError("Upload one ZIP at a time.");
  }

  if (zipFiles.length === 1) {
    const archive = unzipSync(new Uint8Array(await zipFiles[0].arrayBuffer()));
    const entries = [];
    let expandedBytes = 0;
    for (const [path, bytes] of Object.entries(archive)) {
      expandedBytes += bytes.length;
      if (expandedBytes > MAX_UPLOAD_BYTES) {
        throw new InputValidationError("The expanded ZIP is larger than 15 MB.");
      }
      if (REQUIRED_FILES.includes(basename(path))) {
        entries.push([path, strFromU8(bytes)]);
      }
    }
    return { entries, sourceLabel: zipFiles[0].name };
  }

  const entries = await Promise.all(
    files
      .filter((file) => REQUIRED_FILES.includes(basename(file.name)))
      .map(async (file) => [file.webkitRelativePath || file.name, await file.text()]),
  );
  const relativePath = files.find((file) => file.webkitRelativePath)?.webkitRelativePath;
  return {
    entries,
    sourceLabel: relativePath ? relativePath.split("/")[0] : "Uploaded sample",
  };
}

async function handleFiles(fileList) {
  clearUploadError();
  try {
    const { entries, sourceLabel } = await entriesFromFiles(fileList);
    const { nodeText, adjacencyText } = matchRequiredEntries(entries);
    registerSample(nodeText, adjacencyText, sourceLabel);
  } catch (error) {
    state.sample = null;
    elements.runPrediction.disabled = true;
    elements.inputSummary.hidden = true;
    updateContract();
    resetResult();
    showUploadError(error);
  }
}

async function loadExampleTexts() {
  if (state.exampleTexts) return state.exampleTexts;
  const [nodeResponse, adjacencyResponse] = await Promise.all([
    fetch(assetUrl("examples/node_features.csv")),
    fetch(assetUrl("examples/adjacency_area.csv")),
  ]);
  if (!nodeResponse.ok || !adjacencyResponse.ok) {
    throw new Error("The reference sample is unavailable.");
  }
  state.exampleTexts = {
    nodeText: await nodeResponse.text(),
    adjacencyText: await adjacencyResponse.text(),
  };
  return state.exampleTexts;
}

async function loadReferenceSample() {
  clearUploadError();
  try {
    const { nodeText, adjacencyText } = await loadExampleTexts();
    registerSample(nodeText, adjacencyText, "Reference / randomness 0.1000");
  } catch (error) {
    showUploadError(error);
  }
}

async function downloadReferenceSample() {
  try {
    const { nodeText, adjacencyText } = await loadExampleTexts();
    const archive = zipSync(
      {
        "lattice_sample/node_features.csv": strToU8(nodeText),
        "lattice_sample/adjacency_area.csv": strToU8(adjacencyText),
      },
      { level: 6 },
    );
    downloadBlob(
      new Blob([archive], { type: "application/zip" }),
      "lattice_gcn_reference_sample.zip",
    );
    showToast("Reference sample downloaded.");
  } catch {
    showToast("The reference sample could not be downloaded.");
  }
}

function domainLabel(status) {
  if (status === "in") return "Within training range";
  if (status === "near") return "Near training boundary";
  return "Outside training range";
}

function renderPrediction(result) {
  const interval = result.intervals[95];
  elements.predictionValue.textContent = result.prediction.toFixed(6);
  elements.intervalValue.textContent =
    `${interval.lower.toFixed(6)} — ${interval.upper.toFixed(6)} N/mm`;

  elements.domainBadge.textContent = domainLabel(result.distribution.overall);
  elements.domainBadge.className = `domain-badge ${result.distribution.overall}`;
  elements.memberSpread.textContent =
    `σ ${result.memberStd.toFixed(6)} · range ${result.memberRange.toFixed(6)}`;

  const visualMin = result.memberMin - Math.max(result.memberRange * 0.3, 0.0001);
  const visualMax = result.memberMax + Math.max(result.memberRange * 0.3, 0.0001);
  const visualSpan = visualMax - visualMin;
  elements.memberList.innerHTML = result.memberPredictions
    .map(({ seed, value }) => {
      const position = ((value - visualMin) / visualSpan) * 100;
      return `
        <div class="member-row">
          <span>S${seed}</span>
          <div class="member-track" aria-hidden="true"><i style="left:${position}%"></i></div>
          <strong>${value.toFixed(6)}</strong>
        </div>
      `;
    })
    .join("");

  elements.domainChecks.innerHTML = result.distribution.checks
    .map(
      (check) => `
        <div class="check-row">
          <span>${escapeHtml(check.label)}</span>
          <span>${formatNumber(check.value)} / ${formatNumber(check.min)}–${formatNumber(check.max)}</span>
          <strong class="check-state ${check.status}">${check.status === "in" ? "Pass" : check.status}</strong>
        </div>
      `,
    )
    .join("");

  elements.graphReadout.innerHTML = `
    <span>Nodes</span><strong>${result.graph.nodeCount}</strong>
    <span>Edges</span><strong>${result.graph.edgeCount}</strong>
    <span>Density</span><strong>${result.graph.density.toFixed(4)}</strong>
    <span>Mean degree</span><strong>${result.graph.meanDegree.toFixed(3)}</strong>
  `;

  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.resultState.textContent = "PREDICTION COMPLETE";
  elements.resultState.className = "result-state ready";
  drawUploadedGraph();
}

async function runPrediction() {
  if (!state.bundle || !state.sample) return;
  elements.runPrediction.disabled = true;
  elements.resultState.textContent = "RUNNING 5 MEMBERS";
  elements.resultState.className = "result-state running";
  await new Promise((resolve) => requestAnimationFrame(resolve));

  try {
    state.result = predictEnsemble(state.sample, state.bundle);
    renderPrediction(state.result);
  } catch (error) {
    elements.resultState.textContent = "PREDICTION ERROR";
    elements.resultState.className = "result-state";
    showUploadError(error);
  } finally {
    elements.runPrediction.disabled = false;
  }
}

function resultPayload() {
  if (!state.result) return null;
  return {
    generatedAt: new Date().toISOString(),
    source: state.sourceLabel,
    target: "Lattice stiffness",
    units: state.result.units,
    prediction: state.result.prediction,
    ensembleMeanIntervals: state.result.intervals,
    members: state.result.memberPredictions,
    memberStandardDeviation: state.result.memberStd,
    memberRange: state.result.memberRange,
    distribution: state.result.distribution,
    graph: state.result.graph,
    model: state.result.model,
    limitation:
      "Ensemble intervals measure model-initialization disagreement and are not calibrated predictive intervals. Research use only.",
  };
}

async function copyResult() {
  const payload = resultPayload();
  if (!payload) return;
  const text =
    `Predicted lattice stiffness: ${payload.prediction.toFixed(6)} ${payload.units}\n` +
    `95% ensemble-mean interval: ${payload.ensembleMeanIntervals[95].lower.toFixed(6)}–` +
    `${payload.ensembleMeanIntervals[95].upper.toFixed(6)} ${payload.units}\n` +
    `Distribution check: ${domainLabel(payload.distribution.overall)}`;
  try {
    await navigator.clipboard.writeText(text);
    showToast("Prediction copied.");
  } catch {
    showToast("Clipboard access is unavailable in this browser.");
  }
}

function downloadResult() {
  const payload = resultPayload();
  if (!payload) return;
  downloadBlob(
    new Blob([`${JSON.stringify(payload, null, 2)}\n`], {
      type: "application/json",
    }),
    "lattice_gcn_prediction.json",
  );
  showToast("Prediction JSON downloaded.");
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function prepareCanvas(canvas) {
  const rectangle = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rectangle.width * ratio));
  const height = Math.max(1, Math.round(rectangle.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rectangle.width, height: rectangle.height };
}

function drawUploadedGraph() {
  if (!state.sample || elements.resultContent.hidden) return;
  const { context, width, height } = prepareCanvas(elements.resultCanvas);
  const coordinates = state.sample.nodeCoordinates;
  const xValues = coordinates.map(([x]) => x);
  const yValues = coordinates.map(([, y]) => y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const padding = 22;
  const scale = Math.min(
    (width - 2 * padding) / Math.max(1, maxX - minX),
    (height - 2 * padding) / Math.max(1, maxY - minY),
  );
  const offsetX = (width - (maxX - minX) * scale) / 2;
  const offsetY = (height - (maxY - minY) * scale) / 2;
  const project = ([x, y]) => [
    offsetX + (x - minX) * scale,
    height - offsetY - (y - minY) * scale,
  ];

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#101010";
  context.fillRect(0, 0, width, height);
  context.lineWidth = 0.8;
  context.strokeStyle = "rgba(216, 255, 62, 0.45)";
  context.beginPath();
  for (const [source, target] of state.sample.edges) {
    const [x1, y1] = project(coordinates[source]);
    const [x2, y2] = project(coordinates[target]);
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
  }
  context.stroke();

  context.fillStyle = "#f2f0e8";
  for (const coordinate of coordinates) {
    const [x, y] = project(coordinate);
    context.beginPath();
    context.arc(x, y, 1.35, 0, Math.PI * 2);
    context.fill();
  }
}

function createHeroGraph() {
  let seed = 4_292;
  const random = () => {
    seed = (seed * 1_664_525 + 1_013_904_223) >>> 0;
    return seed / 2 ** 32;
  };
  const columns = 9;
  const rows = 11;
  const nodes = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      nodes.push({
        x: (column + 0.5 + (random() - 0.5) * 0.48) / columns,
        y: (row + 0.5 + (random() - 0.5) * 0.48) / rows,
        phase: random() * Math.PI * 2,
      });
    }
  }
  const edges = [];
  const indexOf = (column, row) => row * columns + column;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      if (column + 1 < columns) edges.push([indexOf(column, row), indexOf(column + 1, row)]);
      if (row + 1 < rows) edges.push([indexOf(column, row), indexOf(column, row + 1)]);
      if (column + 1 < columns && row + 1 < rows && random() > 0.57) {
        edges.push([indexOf(column, row), indexOf(column + 1, row + 1)]);
      }
      if (column > 0 && row + 1 < rows && random() > 0.76) {
        edges.push([indexOf(column, row), indexOf(column - 1, row + 1)]);
      }
    }
  }
  return { nodes, edges };
}

const heroGraph = createHeroGraph();
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function drawHeroGraph(timestamp = 0) {
  const { context, width, height } = prepareCanvas(elements.heroCanvas);
  const time = timestamp / 1_000;
  const padding = 28;
  const project = (node) => [
    padding + node.x * (width - 2 * padding),
    padding + node.y * (height - 2 * padding),
  ];

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#101010";
  context.fillRect(0, 0, width, height);

  const gradient = context.createRadialGradient(
    width * 0.58,
    height * 0.42,
    0,
    width * 0.58,
    height * 0.42,
    Math.max(width, height) * 0.65,
  );
  gradient.addColorStop(0, "rgba(40, 87, 255, 0.16)");
  gradient.addColorStop(1, "rgba(40, 87, 255, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  context.lineWidth = 0.7;
  context.strokeStyle = "rgba(242, 240, 232, 0.30)";
  context.beginPath();
  for (const [source, target] of heroGraph.edges) {
    const [x1, y1] = project(heroGraph.nodes[source]);
    const [x2, y2] = project(heroGraph.nodes[target]);
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
  }
  context.stroke();

  for (const node of heroGraph.nodes) {
    const [x, y] = project(node);
    const pulse = reducedMotion ? 0.5 : (Math.sin(time * 1.5 + node.phase) + 1) / 2;
    context.fillStyle = pulse > 0.91 ? "#d8ff3e" : `rgba(242, 240, 232, ${0.58 + pulse * 0.3})`;
    context.beginPath();
    context.arc(x, y, pulse > 0.91 ? 2.6 : 1.45, 0, Math.PI * 2);
    context.fill();
  }

  const sweepY = reducedMotion ? height * 0.55 : ((time * 36) % (height + 80)) - 40;
  const sweep = context.createLinearGradient(0, sweepY - 35, 0, sweepY + 35);
  sweep.addColorStop(0, "rgba(216, 255, 62, 0)");
  sweep.addColorStop(0.5, "rgba(216, 255, 62, 0.13)");
  sweep.addColorStop(1, "rgba(216, 255, 62, 0)");
  context.fillStyle = sweep;
  context.fillRect(0, sweepY - 35, width, 70);

  elements.heroFrame.textContent = `FRAME ${String(Math.floor(timestamp / 33) % 1_000).padStart(3, "0")}`;
  if (!reducedMotion) requestAnimationFrame(drawHeroGraph);
}

function bindEvents() {
  elements.chooseFiles.addEventListener("click", (event) => {
    event.stopPropagation();
    elements.fileInput.click();
  });
  elements.chooseFolder.addEventListener("click", (event) => {
    event.stopPropagation();
    elements.folderInput.click();
  });
  elements.dropZone.addEventListener("click", () => elements.fileInput.click());
  elements.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });
  elements.fileInput.addEventListener("change", () => handleFiles(elements.fileInput.files));
  elements.folderInput.addEventListener("change", () => handleFiles(elements.folderInput.files));

  for (const eventName of ["dragenter", "dragover"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("dragging");
    });
  }
  elements.dropZone.addEventListener("drop", (event) => handleFiles(event.dataTransfer.files));

  elements.loadExample.addEventListener("click", loadReferenceSample);
  elements.downloadExample.addEventListener("click", downloadReferenceSample);
  elements.runPrediction.addEventListener("click", runPrediction);
  elements.copyResult.addEventListener("click", copyResult);
  elements.downloadResult.addEventListener("click", downloadResult);

  const resultObserver = new ResizeObserver(() => drawUploadedGraph());
  resultObserver.observe(elements.resultCanvas);
}

async function loadModel() {
  try {
    const response = await fetch(assetUrl("model_bundle.json"));
    if (!response.ok) throw new Error(`Model bundle request failed (${response.status}).`);
    const bundle = await response.json();
    if (bundle.schemaVersion !== 1 || bundle.members?.length !== 5) {
      throw new Error("The model bundle has an unsupported schema.");
    }
    state.bundle = bundle;
    setModelStatus("ready", "5 models ready");
    elements.runPrediction.disabled = !state.sample;
  } catch (error) {
    console.error(error);
    setModelStatus("error", "Model unavailable");
    showUploadError(new Error("The GCN model bundle could not be loaded."));
  }
}

bindEvents();
loadModel();
requestAnimationFrame(drawHeroGraph);
