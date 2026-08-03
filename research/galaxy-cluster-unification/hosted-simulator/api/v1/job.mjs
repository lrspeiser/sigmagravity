import { handleProductionJob } from "../../lib/production-api-handler.mjs";

export default function handler(request, response) {
  return handleProductionJob(request, response);
}
