import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { sha256 } from "../lib/canonical.mjs";

const root = resolve(import.meta.dirname, "..");
const sparc = resolve(root, "..", "data", "raw", "sparc");
const morphologyLabels = ["S0", "Sa", "Sab", "Sb", "Sbc", "Sc", "Scd", "Sd", "Sdm", "Sm", "Im", "BCD"];

function parseMetadata(line) {
  const fields = line.trim().split(/\s+/);
  return {
    id: fields[0],
    morphology: { code: Number(fields[1]), label: morphologyLabels[Number(fields[1])] ?? "unknown" },
    distanceMpc: Number(fields[2]),
    distanceErrorMpc: Number(fields[3]),
    distanceMethod: Number(fields[4]),
    inclinationDeg: Number(fields[5]),
    inclinationErrorDeg: Number(fields[6]),
    luminosity36BillionLsolar: Number(fields[7]),
    effectiveRadiusKpc: Number(fields[9]),
    effectiveSurfaceBrightnessLsolarPc2: Number(fields[10]),
    diskScaleKpc: Number(fields[11]),
    diskCentralSurfaceBrightnessLsolarPc2: Number(fields[12]),
    hiMassBillionMsolar: Number(fields[13]),
    hiRadiusKpc: Number(fields[14]),
    vFlatKmS: Number(fields[15]),
    vFlatErrorKmS: Number(fields[16]),
    quality: Number(fields[17]),
    references: fields.slice(18).join(" "),
  };
}

function parseCurve(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .map((line) => line.split(/\s+/).map(Number))
    .map(([radiusKpc, vObsKmS, eVObsKmS, vGasKmS, vDiskKmS, vBulgeKmS, sbDisk, sbBulge]) => ({
      radiusKpc, vObsKmS, eVObsKmS, vGasKmS, vDiskKmS, vBulgeKmS, sbDisk, sbBulge,
    }));
}

const metadataText = await readFile(resolve(sparc, "table1.dat"), "utf8");
const metadata = new Map(metadataText.split(/\r?\n/).filter((line) => line.trim()).map(parseMetadata).map((item) => [item.id, item]));
const curveDirectory = resolve(sparc, "rotmod");
const files = (await readdir(curveDirectory)).filter((name) => name.endsWith("_rotmod.dat")).sort();
if (metadata.size !== 175 || files.length !== 175) throw new Error(`expected 175 metadata rows and curves, received ${metadata.size} and ${files.length}`);

const systems = [];
for (const file of files) {
  const id = file.replace(/_rotmod\.dat$/, "");
  const item = metadata.get(id);
  if (!item) throw new Error(`metadata missing for ${id}`);
  systems.push({ ...item, points: parseCurve(await readFile(resolve(curveDirectory, file), "utf8")) });
}
const dataset = {
  id: "sparc-2016-v1",
  title: "SPARC mass models for 175 disk galaxies",
  version: "2016-cds-snapshot",
  sourceUrl: "https://astroweb.case.edu/SPARC/",
  citation: "Lelli, McGaugh & Schombert, AJ 152, 157 (2016)",
  licenseNote: "Public research data; users must cite the source publication and inspect source terms before redistribution.",
};
dataset.sha256 = sha256({ dataset, systems });
const output = `${JSON.stringify({ dataset, systems })}\n`;
await mkdir(resolve(root, "data"), { recursive: true });
await writeFile(resolve(root, "data", "sparc-v1.json"), output, "utf8");
console.log(`wrote ${systems.length} galaxies, ${systems.reduce((sum, item) => sum + item.points.length, 0)} points, sha256 ${dataset.sha256}`);
