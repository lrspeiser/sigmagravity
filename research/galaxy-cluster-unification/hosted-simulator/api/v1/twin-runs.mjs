import { getSystem } from "../../lib/catalog.mjs";
import { runHeldoutTwinBenchmark } from "../../lib/simulator.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const body = parseBody(request);
    const system = getSystem(body.systemId ?? body.system_id);
    if (!system) throw new Error(`unknown system: ${body.systemId ?? body.system_id}`);
    send(response, 200, runHeldoutTwinBenchmark({
      system,
      formula: body.formula ?? body.formulaManifest,
      twinOptions: body.twinOptions,
    }));
  } catch (error) {
    fail(response, error);
  }
}
