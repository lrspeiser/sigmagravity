import { fail, options, requireMethod, send } from "../../lib/http.mjs";
import { getResolvedClusterEvidence } from "../../lib/resolved-cluster-evidence.mjs";


export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  try {
    send(response, 200, getResolvedClusterEvidence(request.query?.system));
  } catch (error) {
    fail(response, error, error.code === "unknown_system" ? 404 : 422);
  }
}
