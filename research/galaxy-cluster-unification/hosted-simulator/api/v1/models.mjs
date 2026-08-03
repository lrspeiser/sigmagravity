import { handleProductionModels } from "../../lib/production-api-handler.mjs";

export default function handler(request, response) {
  return handleProductionModels(request, response);
}
