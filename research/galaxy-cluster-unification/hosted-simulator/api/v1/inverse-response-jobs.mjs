import { options, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "no-store");
  send(response, 503, {
    error: "production_worker_not_connected",
    message: "The inverse baryon-to-response discovery contract is implemented by the local reference backend; durable storage and isolated workers are not connected to this deployment.",
    classification: "hypothesis_generator_not_forward_theory_test",
    localReference: "npm run dev",
    requestSchema: "/schemas/inverse-response-job-submit-v1.schema.json",
  });
}
