import { fail, options, requireMethod, send } from "../../lib/http.mjs";
import { getResolvedTwinEvidence } from "../../lib/resolved-twin-evidence.mjs";


export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  try {
    send(response, 200, getResolvedTwinEvidence(request.query?.galaxy));
  } catch (error) {
    fail(response, error, error.code === "unknown_system" ? 404 : 422);
  }
}
