import { timingSafeEqual } from "node:crypto";
import { requestHeader } from "../../lib/production-auth.mjs";
import { getProductionRuntime } from "../../lib/production-runtime.mjs";
import { options, send } from "../../lib/http.mjs";

function authorized(request, secret) {
  const supplied = requestHeader(request, "authorization");
  const expected = `Bearer ${secret}`;
  if (typeof supplied !== "string") return false;
  const left = Buffer.from(supplied);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

export default async function handler(request, response) {
  if (options(request, response)) return;
  if (!["GET", "POST"].includes(request.method)) {
    response.setHeader("Allow", "OPTIONS, GET, POST");
    send(response, 405, { error: "method_not_allowed", allowed: ["GET", "POST"] });
    return;
  }
  response.setHeader("Cache-Control", "private, no-store");
  const secret = process.env.CRON_SECRET;
  if (typeof secret !== "string" || Buffer.byteLength(secret) < 32) {
    send(response, 503, {
      error: "outbox_scheduler_not_configured",
      message: "The transactional outbox dispatcher requires a long CRON_SECRET",
    });
    return;
  }
  if (!authorized(request, secret)) {
    send(response, 401, { error: "invalid_scheduler_credential" });
    return;
  }
  try {
    const runtime = getProductionRuntime();
    const results = await runtime.controlPlane.dispatchAvailable(runtime.publisher, { limit: 64 });
    send(response, 200, {
      schemaVersion: "sigma-production-outbox-dispatch/1",
      processed: results.length,
      results,
    });
  } catch (error) {
    send(response, error.statusCode ?? 503, {
      error: error.code ?? "outbox_dispatch_unavailable",
      message: error.statusCode ? error.message : "The transactional outbox dispatcher is unavailable",
      ...(error.details ? { details: error.details } : {}),
    });
  }
}
