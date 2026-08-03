const fixedMondFormula = {
  name: "Fixed simple-MOND comparator",
  family: "algebraic_acceleration",
  outputUnit: "m/s^2",
  parameters: { a0: { value: 1.2e-10, unit: "m/s^2" } },
  parameterPolicy: { universal: true, perObjectParameters: 0 },
  expression: {
    op: "mul",
    args: [
      { const: 0.5 },
      {
        op: "add",
        args: [
          { input: "g_bar" },
          {
            op: "sqrt",
            args: [
              {
                op: "add",
                args: [
                  { op: "pow", args: [{ input: "g_bar" }, { const: 2 }] },
                  { op: "mul", args: [{ const: 4 }, { op: "mul", args: [{ input: "g_bar" }, { parameter: "a0" }] }] },
                ],
              },
            ],
          },
        ],
      },
    ],
  },
};

const elements = Object.fromEntries(
  [
    "service-dot", "system-count", "dataset-line", "galaxy-select", "load-system",
    "system-summary", "formula-editor", "validate-formula", "run-formula", "action-status",
    "result-title", "metrics", "curve-chart", "manifest-output", "download-result",
    "run-twin", "twin-proof", "result-legend",
    "create-synthetic", "synthetic-summary", "syn-seed", "syn-mass", "syn-gas", "syn-bulge",
    "syn-scale", "syn-noise",
    "model-example", "load-model-example", "model-editor", "validate-model", "model-status",
    "model-audit-title", "model-audit",
    "resolved-evidence-status", "resolved-galaxy-select", "resolved-model-select",
    "resolved-system-summary", "resolved-fidelity-value", "resolved-fidelity-detail",
    "resolved-transport-value", "resolved-transport-detail", "resolved-accuracy-card",
    "resolved-accuracy-value", "resolved-accuracy-detail", "resolved-chart-title",
    "resolved-curve-chart", "resolved-verdict", "resolved-geometry-note", "resolved-manifest",
  ].map((id) => [id, document.getElementById(id)]),
);

let currentResult = null;
let currentSynthetic = null;
let resolvedEvidence = null;

function setStatus(message, type = "") {
  elements["action-status"].textContent = message;
  elements["action-status"].className = `notice ${type}`.trim();
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
  if (!response.ok) {
    const error = new Error(body.message ?? body.error ?? `HTTP ${response.status}`);
    error.details = body.details;
    throw error;
  }
  return body;
}

function parseFormula() {
  try {
    return JSON.parse(elements["formula-editor"].value);
  } catch (error) {
    throw new Error(`Formula is not valid JSON: ${error.message}`);
  }
}

function parseModel() {
  try {
    return JSON.parse(elements["model-editor"].value);
  } catch (error) {
    throw new Error(`Field model is not valid JSON: ${error.message}`);
  }
}

function setModelStatus(message, type = "") {
  elements["model-status"].textContent = message;
  elements["model-status"].className = `notice ${type}`.trim();
}

async function loadModelExample() {
  const name = elements["model-example"].value;
  const model = await api(`/examples/models/${encodeURIComponent(name)}.json`);
  elements["model-editor"].value = JSON.stringify(model, null, 2);
  setModelStatus(`${model.name} loaded. Validate to confirm the canonical equation tree.`);
}

async function validateModel() {
  setModelStatus("Checking field ranks, units, boundaries, data keys, and parameter policy…");
  const model = parseModel();
  const result = await api("/api/v1/models/validate", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(model),
  });
  elements["model-audit-title"].textContent = model.name;
  elements["model-audit"].textContent = JSON.stringify({
    modelSha256: result.modelSha256,
    documentSha256: result.documentSha256,
    parameterAccounting: result.parameterAccounting,
    requiredCapabilities: result.requiredCapabilities,
    typeAudit: result.typeAudit,
    executionReadiness: result.executionReadiness,
    warnings: result.warnings,
  }, null, 2);
  setModelStatus(`Valid field model · ${result.typeAudit.expressionNodes} expression nodes · ${result.modelSha256.slice(0, 14)}… · worker connection pending`, "success");
}

async function loadCatalog() {
  const [health, datasets, systems] = await Promise.all([
    api("/api/v1/health"), api("/api/v1/datasets"), api("/api/v1/systems"),
  ]);
  elements["service-dot"].classList.toggle("online", health.status === "ok");
  elements["system-count"].textContent = systems.page.total;
  const dataset = datasets.items[0];
  elements["dataset-line"].textContent = `${dataset.id} · ${dataset.systemCount} galaxies · ${dataset.sha256.slice(0, 10)}…`;
  elements["galaxy-select"].innerHTML = systems.items
    .map((item) => `<option value="${item.id}" ${item.id === "DDO154" ? "selected" : ""}>${item.id} · ${item.morphology.label} · Q${item.quality}</option>`)
    .join("");
  await inspectSelectedSystem();
}

async function loadResolvedEvidence() {
  const result = await api("/api/v1/resolved-twin-evidence");
  if (result.schemaVersion !== "sigma-hosted-resolved-twin-evidence/1" || !result.systems?.length) {
    throw new Error("resolved evidence document has an unsupported schema");
  }
  resolvedEvidence = result;
  elements["resolved-galaxy-select"].innerHTML = result.systems
    .map((system) => `<option value="${system.id}">${system.id} · ${system.split}</option>`)
    .join("");
  elements["resolved-evidence-status"].textContent = `${result.stage} · ${result.sample.developmentSystems.length} development + ${result.sample.validationSystems.length} validation + ${result.sample.holdoutSystems.length} final holdout galaxies · ${result.sample.scoredVelocityPixels.toLocaleString("en-US")} scored velocity pixels`;
  renderResolvedEvidence();
}

function selectedResolvedSystem() {
  if (!resolvedEvidence) return null;
  return resolvedEvidence.systems.find(
    (system) => system.id === elements["resolved-galaxy-select"].value,
  ) ?? resolvedEvidence.systems[0];
}

function renderResolvedEvidence() {
  const system = selectedResolvedSystem();
  if (!system) return;
  const modelId = elements["resolved-model-select"].value;
  const model = system.models[modelId];
  const definition = resolvedEvidence.models[modelId];
  const fidelity = system.twinFidelity;
  const transport = model.sourceToTwinTransport.lineOfSightRmseKmS;
  const accuracy = model.twinVersusObserved;
  const massError = fidelity.massRelativeError === 0
    ? "numerically zero"
    : `${(100 * fidelity.massRelativeError).toExponential(1)}%`;

  elements["resolved-system-summary"].textContent = `${system.id} (${system.split}): ${system.observation.scoredPixels.toLocaleString("en-US")} H I velocity pixels; inclination ${system.observation.inclinationDeg.toFixed(1)}°; median gas dispersion ${system.observation.medianDispersionKmS.toFixed(1)} km/s. The displayed comparator has ${definition.perGalaxyGravityParameters} per-galaxy gravity parameters.`;
  elements["resolved-fidelity-value"].textContent = `${(100 * fidelity.totalMapNormalizedL2).toFixed(1)}% map error`;
  elements["resolved-fidelity-detail"].textContent = `Pixel correlation ${fidelity.totalMapPixelCorrelation.toFixed(3)}; mass error ${massError}.`;
  elements["resolved-transport-value"].textContent = `${transport.toFixed(2)} km/s`;
  elements["resolved-transport-detail"].textContent = "2D line-of-sight RMSE between the same formula on registered baryons and the fake twin.";
  elements["resolved-accuracy-value"].textContent = `${accuracy.rmseKmS.toFixed(2)} km/s`;
  elements["resolved-accuracy-detail"].textContent = `${accuracy.errorRatio.toFixed(2)}× the declared uncertainty RMS · ${accuracy.classification}.`;
  elements["resolved-accuracy-card"].className = `score-card ${accuracy.classification}`;
  elements["resolved-chart-title"].textContent = `${system.id} · ${definition.label}`;

  const twinFaithful = fidelity.totalMapPixelCorrelation >= 0.95 && transport <= system.simulatorFidelityLimitKmS;
  const formulaVerdict = accuracy.classification === "consistent"
    ? "matches this raw velocity field within its declared uncertainty"
    : accuracy.classification === "close"
      ? "is close, but remains outside the declared uncertainty"
      : "misses the raw velocity field by more than twice the declared uncertainty";
  elements["resolved-verdict"].innerHTML = `<strong>${twinFaithful ? "The twin is faithful enough for this test." : "The twin itself is not faithful enough."}</strong> ${definition.label} ${formulaVerdict}. A good transport score therefore cannot be read as evidence that the formula is correct.`;
  const diagnostic = system.geometryDiagnostic;
  if (diagnostic) {
    const diagnosticModel = diagnostic.models[modelId];
    if (diagnostic.status === "preregistered_final_holdout_observation_policy") {
      const raw = diagnosticModel.imageAxisSourceVersusObserved;
      const adjusted = diagnosticModel.sourceVersusObserved;
      elements["resolved-geometry-note"].innerHTML = `<strong>Preregistered holdout viewing policy:</strong> the image-only axis differed from the H I kinematic axis by ${diagnostic.axisOffsetDeg.toFixed(1)}°. The policy set one shared kinematic phase before scoring, changing the registered-baryon RMSE from ${raw.rmseKmS.toFixed(1)} to ${adjusted.rmseKmS.toFixed(1)} km/s (${adjusted.classification}). Its fitted speed amplitude was discarded, and no gravity parameter was fitted.`;
    } else {
      const adjusted = diagnosticModel.sourceVersusObserved;
      elements["resolved-geometry-note"].innerHTML = `<strong>Post-reveal viewing-angle method development:</strong> the image-only axis differed from the H I kinematic axis by ${diagnostic.axisOffsetDeg.toFixed(1)}°. Using that one shared observation nuisance changes the registered-baryon RMSE from ${model.sourceVersusObserved.rmseKmS.toFixed(1)} to ${adjusted.rmseKmS.toFixed(1)} km/s (${adjusted.classification}). No speed amplitude or gravity parameter was fitted; this is not blind validation evidence.`;
    }
    elements["resolved-geometry-note"].hidden = false;
  } else {
    elements["resolved-geometry-note"].textContent = "No post-reveal viewing-angle adjustment is included for this development system.";
    elements["resolved-geometry-note"].hidden = false;
  }
  elements["resolved-manifest"].textContent = JSON.stringify({
    evidenceSha256: resolvedEvidence.evidenceSha256,
    parents: resolvedEvidence.parents,
    evidenceClass: resolvedEvidence.evidenceClass,
    selectedSystem: system.id,
    selectedModel: { id: modelId, ...definition },
    generator: resolvedEvidence.generator,
    twinFidelity: fidelity,
    modelScores: {
      sourceVersusObserved: model.sourceVersusObserved,
      twinVersusObserved: model.twinVersusObserved,
      sourceToTwinTransport: model.sourceToTwinTransport,
    },
    geometryDiagnostic: system.geometryDiagnostic ?? null,
    executionBoundary: resolvedEvidence.executionBoundary,
    claimBoundary: resolvedEvidence.claimBoundary,
  }, null, 2);
  drawResolvedCurve(model.radialCurve);
}

function drawResolvedCurve(points) {
  const width = 840, height = 520, left = 66, right = 24;
  const top = 25, curveBottom = 330, residualTop = 386, residualBottom = 480;
  const maxX = Math.max(...points.map((point) => point.radiusKpc));
  const maxY = Math.max(...points.flatMap((point) => [
    point.observedKmS,
    point.sourcePredictionKmS,
    point.twinPredictionKmS,
  ])) * 1.12;
  const residualLimit = Math.max(5, ...points.map(
    (point) => Math.abs(point.twinPredictionKmS - point.observedKmS),
  )) * 1.12;
  const x = (value) => left + (value / maxX) * (width - left - right);
  const y = (value) => curveBottom - (value / maxY) * (curveBottom - top);
  const yr = (value) => residualTop + ((residualLimit - value) / (2 * residualLimit)) * (residualBottom - residualTop);
  const line = (field, vertical = y) => points.map(
    (point, index) => `${index ? "L" : "M"}${x(point.radiusKpc).toFixed(1)},${vertical(point[field]).toFixed(1)}`,
  ).join(" ");
  const residualPath = points.map((point, index) => {
    const value = point.twinPredictionKmS - point.observedKmS;
    return `${index ? "L" : "M"}${x(point.radiusKpc).toFixed(1)},${yr(value).toFixed(1)}`;
  }).join(" ");
  const grids = [];
  for (let index = 0; index <= 4; index += 1) {
    const gy = top + ((curveBottom - top) * index) / 4;
    const value = maxY * (1 - index / 4);
    grids.push(`<line x1="${left}" y1="${gy}" x2="${width-right}" y2="${gy}" stroke="#20272f"/><text x="${left-10}" y="${gy+4}" fill="#79818a" text-anchor="end" font-size="11">${value.toFixed(0)}</text>`);
  }
  const observations = points.map((point) => `<circle cx="${x(point.radiusKpc).toFixed(1)}" cy="${y(point.observedKmS).toFixed(1)}" r="3.2" fill="#f4f2ec"/>`).join("");
  elements["resolved-curve-chart"].innerHTML = `
    <rect width="${width}" height="${height}" fill="#0b0f13"/>
    ${grids.join("")}
    <line x1="${left}" y1="${curveBottom}" x2="${width-right}" y2="${curveBottom}" stroke="#4b5560"/>
    <line x1="${left}" y1="${top}" x2="${left}" y2="${curveBottom}" stroke="#4b5560"/>
    <path d="${line("sourcePredictionKmS")}" fill="none" stroke="#a7ffe0" stroke-width="2.5" stroke-dasharray="4 4"/>
    <path d="${line("twinPredictionKmS")}" fill="none" stroke="#ffba70" stroke-width="3"/>
    ${observations}
    <text transform="translate(17 ${(top+curveBottom)/2}) rotate(-90)" fill="#9299a1" text-anchor="middle" font-size="12">Circular speed (km/s)</text>
    <line x1="${left}" y1="${yr(0)}" x2="${width-right}" y2="${yr(0)}" stroke="#6d747d"/>
    <path d="${residualPath}" fill="none" stroke="#ffba70" stroke-width="2.5"/>
    <line x1="${left}" y1="${residualTop}" x2="${left}" y2="${residualBottom}" stroke="#4b5560"/>
    <text x="${left-10}" y="${residualTop+4}" fill="#79818a" text-anchor="end" font-size="10">+${residualLimit.toFixed(0)}</text>
    <text x="${left-10}" y="${yr(0)+4}" fill="#79818a" text-anchor="end" font-size="10">0</text>
    <text x="${left-10}" y="${residualBottom+4}" fill="#79818a" text-anchor="end" font-size="10">-${residualLimit.toFixed(0)}</text>
    <text transform="translate(17 ${(residualTop+residualBottom)/2}) rotate(-90)" fill="#9299a1" text-anchor="middle" font-size="11">Twin - observed</text>
    <text x="${(left+width-right)/2}" y="${height-10}" fill="#9299a1" text-anchor="middle" font-size="12">Radius (kpc)</text>
    <text x="${left}" y="${height-27}" fill="#79818a" text-anchor="middle" font-size="11">0</text>
    <text x="${width-right}" y="${height-27}" fill="#79818a" text-anchor="middle" font-size="11">${maxX.toFixed(1)}</text>`;
}

async function inspectSelectedSystem() {
  currentSynthetic = null;
  elements["synthetic-summary"].textContent = "";
  const system = await api(`/api/v1/systems/${encodeURIComponent(elements["galaxy-select"].value)}`);
  elements["system-summary"].textContent = `${system.morphology.label}; ${system.pointCount} points to ${system.outerRadiusKpc.toFixed(2)} kpc; ${system.distanceMpc.toFixed(2)} Mpc away; quality ${system.quality}.`;
}

async function validate() {
  setStatus("Validating dimensions and parameter policy…");
  const result = await api("/api/v1/formulas/validate", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(parseFormula()),
  });
  setStatus(`Valid · ${result.safetyAudit.nodeCount} AST nodes · ${result.parameterAccounting.universal} universal parameter(s) · ${result.formulaSha256.slice(0, 14)}…`, "success");
  return result;
}

async function createSynthetic() {
  setStatus("Creating deterministic synthetic radial system…");
  const request = {
    seed: Number(elements["syn-seed"].value),
    physical: {
      baryonicMassMsolar: Number(elements["syn-mass"].value),
      gasFraction: Number(elements["syn-gas"].value),
      bulgeFraction: Number(elements["syn-bulge"].value),
      diskScaleKpc: Number(elements["syn-scale"].value),
    },
    observation: { pointCount: 32, noiseKmS: Number(elements["syn-noise"].value) },
  };
  currentSynthetic = await api("/api/v1/synthetic-galaxies", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(request),
  });
  elements["synthetic-summary"].textContent = `${currentSynthetic.id}; ${currentSynthetic.points.length} points. The next run will use this synthetic system.`;
  setStatus("Synthetic system created. Formula parameters remain universal.", "success");
}

async function run() {
  elements["run-formula"].disabled = true;
  setStatus("Running submitted, fixed-MOND, and Newtonian curves…");
  try {
    await validate();
    const body = {
      systemIds: currentSynthetic ? [] : [elements["galaxy-select"].value],
      syntheticSystem: currentSynthetic ?? undefined,
      tests: ["rotation_curve"],
      formula: parseFormula(),
    };
    currentResult = await api("/api/v1/runs", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
    });
    renderResult(currentResult);
    setStatus(`Run complete · ${currentResult.id} · no fitted nuisance parameters`, "success");
  } finally {
    elements["run-formula"].disabled = false;
  }
}

async function runTwin() {
  elements["run-twin"].disabled = true;
  elements["run-formula"].disabled = true;
  currentSynthetic = null;
  elements["synthetic-summary"].textContent = "";
  setStatus("Building a baryonic twin without reading measured speeds…");
  try {
    await validate();
    currentResult = await api("/api/v1/twin-runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        systemId: elements["galaxy-select"].value,
        formula: parseFormula(),
        twinOptions: { controlPointCount: 6 },
      }),
    });
    renderTwinResult(currentResult);
    const sourceError = 100 * currentResult.metrics.sourceReconstruction.gBarNormalizedRmse;
    const formulaError = currentResult.metrics.formulaOnGeneratedTwin.rmseKmS;
    setStatus(`Twin test complete · source error ${sourceError.toFixed(2)}% · formula error ${formulaError.toFixed(2)} km/s`, "success");
  } finally {
    elements["run-twin"].disabled = false;
    elements["run-formula"].disabled = false;
  }
}

function metric(label, value, suffix = "") {
  return `<div class="metric"><strong>${Number(value).toFixed(2)}${suffix}</strong><span>${label}</span></div>`;
}

function renderResult(result) {
  const system = result.results[0];
  const mond = result.comparators.fixedMond[0];
  const newtonian = result.comparators.newtonian[0];
  elements["result-title"].textContent = `${system.systemId} rotation curve`;
  elements.metrics.className = "metrics";
  elements.metrics.innerHTML = [
    metric("Submitted RMSE", system.metrics.rmseKmS, " km/s"),
    metric("Fixed MOND RMSE", mond.metrics.rmseKmS, " km/s"),
    metric("Newtonian RMSE", newtonian.metrics.rmseKmS, " km/s"),
  ].join("");
  elements["twin-proof"].hidden = true;
  elements["result-legend"].innerHTML = `
    <span><i class="observed"></i>Observed</span>
    <span><i class="submitted"></i>Submitted</span>
    <span><i class="mond"></i>Fixed MOND</span>
    <span><i class="newtonian"></i>Newtonian baryons</span>`;
  drawChart(system.predictions, mond.predictions, newtonian.predictions);
  elements["manifest-output"].textContent = JSON.stringify({ manifest: result.manifest, scores: result.scores, caveats: result.caveats }, null, 2);
  elements["download-result"].disabled = false;
}

function renderTwinResult(result) {
  const metrics = result.metrics;
  elements["result-title"].textContent = `${result.system.id} held-out radial twin`;
  elements.metrics.className = "metrics";
  elements.metrics.innerHTML = [
    metric("Twin source error", 100 * metrics.sourceReconstruction.gBarNormalizedRmse, "%"),
    metric("Formula · twin", metrics.formulaOnGeneratedTwin.rmseKmS, " km/s"),
    metric("Formula · measured", metrics.formulaOnMeasuredBaryons.rmseKmS, " km/s"),
    metric("Twin transport", metrics.transport.predictionRmseKmS, " km/s"),
  ].join("");
  elements["twin-proof"].hidden = false;
  elements["twin-proof"].textContent = `Leakage guard: measured speeds and uncertainties were withheld during twin extraction. Gravity parameters used to build the twin: ${result.manifest.twinProtocol.gravityParametersInExtraction}.`;
  elements["result-legend"].innerHTML = `
    <span><i class="observed"></i>Observed</span>
    <span><i class="submitted"></i>Formula on twin</span>
    <span><i class="measured-source"></i>Same formula on measured baryons</span>
    <span><i class="mond"></i>Fixed MOND on twin</span>
    <span><i class="newtonian"></i>Newtonian twin</span>
    <span><i class="uncertainty"></i>1σ band in residual panel</span>`;
  drawTwinChart(result.predictions);
  elements["manifest-output"].textContent = JSON.stringify({
    manifest: result.manifest,
    metrics: result.metrics,
    twinParameterPackage: result.twin.parameterPackage,
    caveats: result.caveats,
  }, null, 2);
  elements["download-result"].disabled = false;
}

function drawChart(submitted, mond, newtonian) {
  const width = 800, height = 440, left = 62, right = 24, top = 24, bottom = 50;
  elements["curve-chart"].setAttribute("viewBox", `0 0 ${width} ${height}`);
  const maxX = Math.max(...submitted.map((point) => point.radiusKpc));
  const maxY = Math.max(...submitted.flatMap((point, index) => [point.observedKmS + point.uncertaintyKmS, point.predictedKmS, mond[index].predictedKmS, newtonian[index].predictedKmS])) * 1.12;
  const x = (value) => left + (value / maxX) * (width - left - right);
  const y = (value) => height - bottom - (value / maxY) * (height - top - bottom);
  const line = (points) => points.map((point, index) => `${index ? "L" : "M"}${x(point.radiusKpc).toFixed(1)},${y(point.predictedKmS).toFixed(1)}`).join(" ");
  const grids = [];
  for (let index = 0; index <= 5; index += 1) {
    const gy = top + ((height - top - bottom) * index) / 5;
    const value = maxY * (1 - index / 5);
    grids.push(`<line x1="${left}" y1="${gy}" x2="${width-right}" y2="${gy}" stroke="#20272f"/><text x="${left-10}" y="${gy+4}" fill="#79818a" text-anchor="end" font-size="11">${value.toFixed(0)}</text>`);
  }
  const observations = submitted.map((point) => {
    const cx = x(point.radiusKpc), cy = y(point.observedKmS), y1 = y(point.observedKmS - point.uncertaintyKmS), y2 = y(point.observedKmS + point.uncertaintyKmS);
    return `<line x1="${cx}" y1="${y1}" x2="${cx}" y2="${y2}" stroke="#d8d8d2" opacity=".6"/><circle cx="${cx}" cy="${cy}" r="3" fill="#f4f2ec"/>`;
  }).join("");
  elements["curve-chart"].innerHTML = `
    <rect width="${width}" height="${height}" fill="#0b0f13"/>
    ${grids.join("")}
    <line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" stroke="#4b5560"/>
    <line x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}" stroke="#4b5560"/>
    <path d="${line(newtonian)}" fill="none" stroke="#77a8ff" stroke-width="2" opacity=".8"/>
    <path d="${line(mond)}" fill="none" stroke="#ffba70" stroke-width="2" stroke-dasharray="6 5"/>
    <path d="${line(submitted)}" fill="none" stroke="#6cf0bd" stroke-width="3"/>
    ${observations}
    <text x="${(left+width-right)/2}" y="${height-12}" fill="#9299a1" text-anchor="middle" font-size="12">Radius (kpc)</text>
    <text transform="translate(16 ${(top+height-bottom)/2}) rotate(-90)" fill="#9299a1" text-anchor="middle" font-size="12">Circular speed (km/s)</text>
    <text x="${left}" y="${height-29}" fill="#79818a" text-anchor="middle" font-size="11">0</text>
    <text x="${width-right}" y="${height-29}" fill="#79818a" text-anchor="middle" font-size="11">${maxX.toFixed(1)}</text>`;
}

function drawTwinChart(points) {
  const width = 800, height = 560, left = 62, right = 24;
  const top = 24, curveBottom = 350, residualTop = 406, residualBottom = 520;
  const maxX = Math.max(...points.map((point) => point.radiusKpc));
  const maxY = Math.max(...points.flatMap((point) => [
    point.observedKmS + point.uncertaintyKmS,
    point.submittedMeasuredBaryonsKmS,
    point.submittedTwinKmS,
    point.fixedMondTwinKmS,
    point.newtonianTwinKmS,
  ])) * 1.1;
  const residualLimit = Math.max(5, ...points.map(
    (point) => Math.abs(point.submittedTwinResidualKmS) + point.uncertaintyKmS,
  )) * 1.12;
  const x = (value) => left + (value / maxX) * (width - left - right);
  const y = (value) => curveBottom - (value / maxY) * (curveBottom - top);
  const yr = (value) => residualTop + ((residualLimit - value) / (2 * residualLimit)) * (residualBottom - residualTop);
  const line = (field, vertical = y) => points.map(
    (point, index) => `${index ? "L" : "M"}${x(point.radiusKpc).toFixed(1)},${vertical(point[field]).toFixed(1)}`,
  ).join(" ");
  const grids = [];
  for (let index = 0; index <= 4; index += 1) {
    const gy = top + ((curveBottom - top) * index) / 4;
    const value = maxY * (1 - index / 4);
    grids.push(`<line x1="${left}" y1="${gy}" x2="${width-right}" y2="${gy}" stroke="#20272f"/><text x="${left-10}" y="${gy+4}" fill="#79818a" text-anchor="end" font-size="11">${value.toFixed(0)}</text>`);
  }
  const observations = points.map((point) => {
    const cx = x(point.radiusKpc), cy = y(point.observedKmS);
    const y1 = y(point.observedKmS - point.uncertaintyKmS), y2 = y(point.observedKmS + point.uncertaintyKmS);
    return `<line x1="${cx}" y1="${y1}" x2="${cx}" y2="${y2}" stroke="#d8d8d2" opacity=".6"/><circle cx="${cx}" cy="${cy}" r="3" fill="#f4f2ec"/>`;
  }).join("");
  const upperBand = points.map((point) => `${x(point.radiusKpc).toFixed(1)},${yr(point.uncertaintyKmS).toFixed(1)}`).join(" ");
  const lowerBand = [...points].reverse().map((point) => `${x(point.radiusKpc).toFixed(1)},${yr(-point.uncertaintyKmS).toFixed(1)}`).join(" ");
  elements["curve-chart"].setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements["curve-chart"].innerHTML = `
    <rect width="${width}" height="${height}" fill="#0b0f13"/>
    ${grids.join("")}
    <line x1="${left}" y1="${curveBottom}" x2="${width-right}" y2="${curveBottom}" stroke="#4b5560"/>
    <line x1="${left}" y1="${top}" x2="${left}" y2="${curveBottom}" stroke="#4b5560"/>
    <path d="${line("newtonianTwinKmS")}" fill="none" stroke="#77a8ff" stroke-width="2" opacity=".8"/>
    <path d="${line("fixedMondTwinKmS")}" fill="none" stroke="#ffba70" stroke-width="2" stroke-dasharray="6 5"/>
    <path d="${line("submittedMeasuredBaryonsKmS")}" fill="none" stroke="#a7ffe0" stroke-width="2" stroke-dasharray="3 4" opacity=".85"/>
    <path d="${line("submittedTwinKmS")}" fill="none" stroke="#6cf0bd" stroke-width="3"/>
    ${observations}
    <text transform="translate(16 ${(top+curveBottom)/2}) rotate(-90)" fill="#9299a1" text-anchor="middle" font-size="12">Circular speed (km/s)</text>
    <polygon points="${upperBand} ${lowerBand}" fill="rgba(244,242,236,.10)"/>
    <line x1="${left}" y1="${yr(0)}" x2="${width-right}" y2="${yr(0)}" stroke="#6d747d"/>
    <path d="${line("submittedTwinResidualKmS", yr)}" fill="none" stroke="#6cf0bd" stroke-width="2.5"/>
    <line x1="${left}" y1="${residualTop}" x2="${left}" y2="${residualBottom}" stroke="#4b5560"/>
    <text x="${left-10}" y="${residualTop+4}" fill="#79818a" text-anchor="end" font-size="10">+${residualLimit.toFixed(0)}</text>
    <text x="${left-10}" y="${yr(0)+4}" fill="#79818a" text-anchor="end" font-size="10">0</text>
    <text x="${left-10}" y="${residualBottom+4}" fill="#79818a" text-anchor="end" font-size="10">−${residualLimit.toFixed(0)}</text>
    <text transform="translate(16 ${(residualTop+residualBottom)/2}) rotate(-90)" fill="#9299a1" text-anchor="middle" font-size="11">Twin − observed</text>
    <text x="${(left+width-right)/2}" y="${height-10}" fill="#9299a1" text-anchor="middle" font-size="12">Radius (kpc)</text>
    <text x="${left}" y="${height-27}" fill="#79818a" text-anchor="middle" font-size="11">0</text>
    <text x="${width-right}" y="${height-27}" fill="#79818a" text-anchor="middle" font-size="11">${maxX.toFixed(1)}</text>`;
}

function downloadResult() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${currentResult.id}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

function handle(action) {
  return async () => {
    try { await action(); }
    catch (error) { setStatus(`${error.message}${error.details?.length ? ` — ${error.details.join("; ")}` : ""}`, "error"); }
  };
}

elements["formula-editor"].value = JSON.stringify(fixedMondFormula, null, 2);
elements["load-system"].addEventListener("click", handle(inspectSelectedSystem));
elements["galaxy-select"].addEventListener("change", handle(inspectSelectedSystem));
elements["validate-formula"].addEventListener("click", handle(validate));
elements["run-formula"].addEventListener("click", handle(run));
elements["run-twin"].addEventListener("click", handle(runTwin));
elements["create-synthetic"].addEventListener("click", handle(createSynthetic));
elements["download-result"].addEventListener("click", downloadResult);
elements["load-model-example"].addEventListener("click", handle(loadModelExample));
elements["validate-model"].addEventListener("click", handle(validateModel));
elements["resolved-galaxy-select"].addEventListener("change", renderResolvedEvidence);
elements["resolved-model-select"].addEventListener("change", renderResolvedEvidence);
handle(loadModelExample)();
handle(loadCatalog)();
loadResolvedEvidence().catch((error) => {
  elements["resolved-evidence-status"].textContent = `Resolved evidence unavailable: ${error.message}`;
  elements["resolved-evidence-status"].classList.add("error-text");
});
