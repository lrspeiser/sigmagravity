import { handleProductionJob } from "../../lib/production-api-handler.mjs";
import { send } from "../../lib/http.mjs";

const IDENTIFIER = /^job_[0-9a-f]{24}$/;
const ARTIFACT = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/;

export default function handler(request, response) {
  const { id, resource, name } = request.query ?? {};
  if (
    typeof id !== "string"
    || !IDENTIFIER.test(id)
    || ![undefined, "events", "artifacts", "artifact", "cancel"].includes(resource)
    || (resource === "artifact" && (typeof name !== "string" || !ARTIFACT.test(name)))
    || (resource !== "artifact" && name !== undefined)
  ) {
    send(response, 404, { error: "not_found" });
    return;
  }
  return handleProductionJob(request, response);
}
