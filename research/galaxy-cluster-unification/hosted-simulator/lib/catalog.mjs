import { readFileSync } from "node:fs";

const catalog = JSON.parse(
  readFileSync(new URL("../data/sparc-v1.json", import.meta.url), "utf8"),
);

export function datasetSummary() {
  return {
    id: catalog.dataset.id,
    title: catalog.dataset.title,
    version: catalog.dataset.version,
    sha256: catalog.dataset.sha256,
    sourceUrl: catalog.dataset.sourceUrl,
    citation: catalog.dataset.citation,
    licenseNote: catalog.dataset.licenseNote,
    evidenceClass: "public_spent_validation",
    systemCount: catalog.systems.length,
  };
}

export function listSystems(filters = {}) {
  const query = String(filters.query ?? "").trim().toLowerCase();
  const morphology = String(filters.morphology ?? "").trim().toLowerCase();
  const quality = Number(filters.quality ?? 0);
  return catalog.systems
    .filter((system) => !query || system.id.toLowerCase().includes(query))
    .filter((system) => !morphology || system.morphology.label.toLowerCase().includes(morphology))
    .filter((system) => !quality || system.quality === quality)
    .map(summarizeSystem);
}

export function getSystem(id) {
  return catalog.systems.find((system) => system.id.toLowerCase() === String(id).toLowerCase());
}

export function summarizeSystem(system) {
  return {
    id: system.id,
    type: "galaxy",
    datasetRelease: catalog.dataset.id,
    sampleState: "spent",
    morphology: system.morphology,
    distanceMpc: system.distanceMpc,
    inclinationDeg: system.inclinationDeg,
    quality: system.quality,
    pointCount: system.points.length,
    outerRadiusKpc: system.points.at(-1)?.radiusKpc ?? null,
    vFlatKmS: system.vFlatKmS,
    supportedTests: ["rotation_curve"],
  };
}

export { catalog };
