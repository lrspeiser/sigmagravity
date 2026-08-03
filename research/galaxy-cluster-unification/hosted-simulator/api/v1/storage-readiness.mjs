import { options, requireMethod, send } from "../../lib/http.mjs";
import { privateBlobStorageState } from "../../lib/private-blob-store.mjs";
import { readProductionDatabaseReadiness } from "../../lib/production-database.mjs";
import { productionQueueState, readQueueCanary } from "../../lib/production-queue.mjs";
import { productionOutboxSchedulerState } from "../../lib/production-runtime.mjs";
import { statelessWorkerState } from "../../lib/stateless-worker-client.mjs";

export default async function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  response.setHeader("Cache-Control", "no-store");
  const objectStorage = privateBlobStorageState();
  const queueConfiguration = productionQueueState();
  const database = await readProductionDatabaseReadiness();
  const worker = statelessWorkerState();
  const scheduler = productionOutboxSchedulerState();
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
    && database.state === "verified_migrated"
    && scheduler === "verified"
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
      state: database.state,
      migrations: ["production-control-plane-v1", "production-research-api-v2"],
      verifiedMigrations: database.migrations,
      projectIsolation: "hashed_bearer_credentials_and_database_scope_checks_built",
      quotas: ["active_jobs", "upload_bytes", "attempts_per_job"],
      auditEvents: "built",
    },
    outboxScheduler: {
      endpoint: "/api/v1/outbox-dispatch",
      state: scheduler,
    },
    statelessScientificContainer: { state: worker },
    productionExecution: productionReady ? "ready" : "not_ready",
    boundary: productionReady
      ? "All control-plane layers are configured; scientific validity still depends on the confirmed model, data, solver diagnostics, and report."
      : "Production execution remains fail-closed until private objects, a verified queue, migrated Postgres metadata, a verified outbox scheduler, and a stateless scientific worker are all connected.",
  });
}
