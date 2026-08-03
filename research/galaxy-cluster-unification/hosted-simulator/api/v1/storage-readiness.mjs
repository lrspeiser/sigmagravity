import { options, requireMethod, send } from "../../lib/http.mjs";
import { privateBlobStorageState } from "../../lib/private-blob-store.mjs";
import { productionDatabaseState } from "../../lib/production-database.mjs";
import { productionQueueState, readQueueCanary } from "../../lib/production-queue.mjs";
import { statelessWorkerState } from "../../lib/stateless-worker-client.mjs";

export default async function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  response.setHeader("Cache-Control", "no-store");
  const objectStorage = privateBlobStorageState();
  const queueConfiguration = productionQueueState();
  const database = productionDatabaseState();
  const worker = statelessWorkerState();
  let queueCanary = "not_run";
  if (objectStorage === "configured" && queueConfiguration === "configured") {
    try {
      queueCanary = (await readQueueCanary()).state;
    } catch {
      queueCanary = "verification_failed";
    }
  }
  const queue = queueCanary === "verified_consumed" ? "verified_consumed" : queueConfiguration;
  const productionReady = objectStorage === "configured"
    && queue === "verified_consumed"
    && database === "configured"
    && worker === "configured";
  send(response, 200, {
    schemaVersion: "sigma-production-storage-readiness/1",
    status: productionReady
      ? "production_control_plane_ready"
      : queue === "verified_consumed"
        ? "durable_storage_and_queue_connected"
        : objectStorage === "configured"
          ? "durable_object_storage_connected"
          : "incomplete",
    objectStorage: {
      provider: "vercel_blob",
      access: "private",
      state: objectStorage,
      guarantees: [
        "content_addressed_path",
        "bounded_object_size",
        "verified_sha256_on_read",
        "idempotent_immutable_write",
      ],
    },
    queue: {
      provider: "vercel_queues",
      state: queue,
      canary: queueCanary,
      delivery: "at_least_once",
      consumer: "private_deployment_bound_trigger",
    },
    jobMetadataDatabase: {
      provider: "postgresql",
      state: database,
      migration: "production-control-plane-v1",
    },
    statelessScientificContainer: { state: worker },
    productionExecution: productionReady ? "ready" : "not_ready",
    boundary: productionReady
      ? "All control-plane layers are configured; scientific validity still depends on the confirmed model, data, solver diagnostics, and report."
      : "Production execution remains fail-closed until private objects, a verified queue, Postgres metadata, and a stateless scientific worker are all connected.",
  });
}
