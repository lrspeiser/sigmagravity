import { getSystem, summarizeSystem } from "../../lib/catalog.mjs";
import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  const system = getSystem(request.query.id);
  if (!system) {
    send(response, 404, { error: "not_found", message: `unknown system: ${request.query.id ?? ""}` });
    return;
  }
  send(response, 200, { ...summarizeSystem(system), metadata: system, points: system.points });
}
