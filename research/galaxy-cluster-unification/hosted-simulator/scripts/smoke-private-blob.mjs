import assert from "node:assert/strict";
import { PrivateBlobStore, privateBlobStorageState } from "../lib/private-blob-store.mjs";

assert.equal(privateBlobStorageState(), "configured", "private Blob storage is not configured");
const store = new PrivateBlobStore();
const payload = Buffer.from(`${JSON.stringify({
  schemaVersion: "sigma-durable-storage-canary/1",
  purpose: "content-addressed private object storage acceptance",
  service: "sigma-gravity-research-simulator",
})}\n`, "utf8");
const first = await store.putImmutable({
  namespace: "deployment-canary",
  bytes: payload,
  mediaType: "application/json",
  extension: "json",
});
const second = await store.putImmutable({
  namespace: "deployment-canary",
  bytes: payload,
  mediaType: "application/json",
  extension: "json",
});
const downloaded = await store.getVerified(first);
assert.deepEqual(downloaded, payload);
assert.equal(second.pathname, first.pathname);
assert.equal(second.sha256, first.sha256);
assert.equal(second.etag, first.etag);

console.log(JSON.stringify({
  schemaVersion: "sigma-private-blob-smoke/1",
  state: "pass",
  provider: first.provider,
  access: first.access,
  pathname: first.pathname,
  sha256: first.sha256,
  bytes: first.bytes,
  idempotentReference: true,
  verifiedRead: true,
}, null, 2));
