import { handleProductionUpload } from "../../lib/production-api-handler.mjs";
import { send } from "../../lib/http.mjs";

const IDENTIFIER = /^upload_[0-9a-f]{24}$/;

export default function handler(request, response) {
  const id = request.query?.id;
  const resource = request.query?.resource;
  if (typeof id !== "string" || !IDENTIFIER.test(id) || ![undefined, "content"].includes(resource)) {
    send(response, 404, { error: "not_found" });
    return;
  }
  return handleProductionUpload(request, response);
}
