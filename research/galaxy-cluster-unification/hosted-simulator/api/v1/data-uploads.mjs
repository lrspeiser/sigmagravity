import { options, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response)) return;
  if (request.method !== "POST") {
    response.setHeader("Allow", "OPTIONS, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["POST"] });
    return;
  }
  response.setHeader("Cache-Control", "no-store");
  send(response, 503, {
    error: "production_storage_not_connected",
    message: "The upload contract is implemented by the local reference backend; durable object storage is not connected to this gateway deployment.",
    localReference: "npm run dev",
    requestSchema: "/schemas/data-upload-request-v1.schema.json",
  });
}
