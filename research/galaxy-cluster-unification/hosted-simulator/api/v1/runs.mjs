import { getSystem } from "../../lib/catalog.mjs";
import { runRotationCurveBenchmark } from "../../lib/simulator.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const body = parseBody(request);
    const tests = body.tests ?? ["rotation_curve"];
    const unsupported = tests.filter((test) => test !== "rotation_curve");
    if (unsupported.length) {
      send(response, 503, {
        error: "worker_not_connected",
        message: `The hosted preview cannot yet execute: ${unsupported.join(", ")}`,
        availableTests: ["rotation_curve"],
        plannedTests: ["velocity_field", "raw_lensing_roots", "critical_curves", "solar_ppn", "full_benchmark"],
      });
      return;
    }
    const ids = body.systemIds ?? body.system_ids ?? [];
    const systems = ids.map((id) => {
      const system = getSystem(id);
      if (!system) throw new Error(`unknown system: ${id}`);
      return system;
    });
    if (body.syntheticSystem) systems.push(body.syntheticSystem);
    const result = runRotationCurveBenchmark({ systems, formula: body.formula ?? body.formulaManifest });
    send(response, 200, result);
  } catch (error) {
    fail(response, error);
  }
}
