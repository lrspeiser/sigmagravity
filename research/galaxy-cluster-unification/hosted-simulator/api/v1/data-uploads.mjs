import { handleProductionUploads } from "../../lib/production-api-handler.mjs";

export default function handler(request, response) {
  return handleProductionUploads(request, response);
}
