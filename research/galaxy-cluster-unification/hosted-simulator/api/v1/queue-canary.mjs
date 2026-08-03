import { options, requireMethod, send } from "../../lib/http.mjs";
import {
  productionQueueState,
  publishQueueCanary,
  readQueueCanary,
} from "../../lib/production-queue.mjs";

export default async function handler(request, response) {
  if (options(request, response)) return;
  response.setHeader("Cache-Control", "no-store");
  if (!requireMethod(request, response, request.method === "POST" ? "POST" : "GET")) return;
  const queue = productionQueueState();
  if (queue !== "configured") {
    send(response, 503, {
      error: queue === "misconfigured" ? "production_queue_misconfigured" : "production_queue_not_configured",
    });
    return;
  }
  try {
    const result = request.method === "POST" ? await publishQueueCanary() : await readQueueCanary();
    send(response, request.method === "POST" ? 202 : 200, result);
  } catch (error) {
    send(response, 503, {
      error: "production_queue_canary_unavailable",
      message: "The deployment-bound queue canary could not be published or verified.",
      reason: error.message,
    });
  }
}
