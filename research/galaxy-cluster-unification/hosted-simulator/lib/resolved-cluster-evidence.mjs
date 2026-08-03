import { readFileSync } from "node:fs";


const evidence = Object.freeze(JSON.parse(readFileSync(
  new URL("../data/resolved-cluster-evidence-v1.json", import.meta.url),
  "utf8",
)));

export function getResolvedClusterEvidence(systemId) {
  if (!systemId) return evidence;
  const key = String(systemId).trim().toUpperCase();
  const system = evidence.systems.find((item) => item.id.toUpperCase() === key);
  if (!system) {
    const error = new Error(`unknown resolved cluster evidence system: ${systemId}`);
    error.code = "unknown_system";
    throw error;
  }
  return {
    ...evidence,
    sample: { ...evidence.sample, returnedSystems: 1 },
    systems: [system],
  };
}
