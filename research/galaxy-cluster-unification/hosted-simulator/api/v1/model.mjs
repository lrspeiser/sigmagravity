import { handleProductionModel } from "../../lib/production-api-handler.mjs";

export default function handler(request, response) {
  return handleProductionModel(request, response);
}
