import { PrivateBlobStore, privateBlobStorageState } from "./private-blob-store.mjs";
import { ProductionControlPlane } from "./production-control-plane.mjs";
import { NeonProductionDatabase, productionDatabaseState } from "./production-database.mjs";
import { productionQueueState, VercelQueuePublisher } from "./production-queue.mjs";
import { ProductionResearchService } from "./production-research-service.mjs";
import { statelessWorkerState } from "./stateless-worker-client.mjs";

export class ProductionRuntimeError extends Error {
  constructor(components) {
    super("The durable production research API is not fully configured");
    this.name = "ProductionRuntimeError";
    this.code = "production_control_plane_not_connected";
    this.statusCode = 503;
    this.details = components;
  }
}

export function productionRuntimeState(environment = process.env) {
  const components = {
    database: productionDatabaseState(environment),
    objectStorage: privateBlobStorageState(environment),
    queue: productionQueueState(environment),
  };
  return {
    state: Object.values(components).every((value) => value === "configured") ? "configured" : "not_ready",
    components,
  };
}

export function productionOutboxSchedulerState(environment = process.env) {
  const secretConfigured = typeof environment.CRON_SECRET === "string"
    && Buffer.byteLength(environment.CRON_SECRET) >= 32;
  if (!secretConfigured) return "not_configured";
  return environment.SIGMA_OUTBOX_SCHEDULER_VERIFIED === "true"
    ? "verified"
    : "credential_configured_trigger_verification_required";
}

let cachedRuntime;

export function getProductionRuntime({ environment = process.env, fresh = false } = {}) {
  const state = productionRuntimeState(environment);
  if (state.state !== "configured") throw new ProductionRuntimeError(state.components);
  if (!fresh && cachedRuntime) return cachedRuntime;
  const database = new NeonProductionDatabase({ environment });
  const store = new PrivateBlobStore({ environment });
  const publisher = new VercelQueuePublisher({ environment });
  const controlPlane = new ProductionControlPlane({ database });
  const worker = statelessWorkerState(environment);
  const scheduler = productionOutboxSchedulerState(environment);
  const runtime = {
    database,
    store,
    publisher,
    controlPlane,
    service: new ProductionResearchService({ controlPlane, store, publisher }),
    jobSubmissionReady: worker === "configured" && scheduler === "verified",
    executionComponents: { statelessWorker: worker, outboxScheduler: scheduler },
  };
  if (!fresh) cachedRuntime = runtime;
  return runtime;
}
