import { readFileSync } from "node:fs";


const evidence = Object.freeze(JSON.parse(readFileSync(
  new URL("../data/resolved-twin-development-v1.json", import.meta.url),
  "utf8",
)));

export function getResolvedTwinEvidence(galaxy) {
  if (!galaxy) return evidence;
  const key = String(galaxy).trim().toUpperCase();
  const system = evidence.systems.find((item) => item.id.toUpperCase() === key);
  if (!system) {
    const error = new Error(`unknown resolved development system: ${galaxy}`);
    error.code = "unknown_system";
    throw error;
  }
  return {
    ...evidence,
    sample: { ...evidence.sample, returnedSystems: 1 },
    systems: [system],
  };
}

