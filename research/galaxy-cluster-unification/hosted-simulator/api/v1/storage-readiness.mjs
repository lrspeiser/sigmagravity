import { options, requireMethod, send } from "../../lib/http.mjs";
import { privateBlobStorageState } from "../../lib/private-blob-store.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  const objectStorage = privateBlobStorageState();
  send(response, 200, {
    schemaVersion: "sigma-production-storage-readiness/1",
    status: objectStorage === "configured" ? "durable_object_storage_connected" : "incomplete",
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
    queue: { state: "not_connected" },
    jobMetadataDatabase: { state: "not_connected" },
    statelessScientificContainer: { state: "not_connected" },
    productionExecution: "not_ready",
    boundary: "Durable private objects are one completed layer; they do not make the scientific job lifecycle durable by themselves.",
  });
}
