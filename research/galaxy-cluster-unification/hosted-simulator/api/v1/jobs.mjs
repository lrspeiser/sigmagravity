import { handleProductionJobs } from "../../lib/production-api-handler.mjs";

export default function handler(request, response) {
  return handleProductionJobs(request, response);
}
