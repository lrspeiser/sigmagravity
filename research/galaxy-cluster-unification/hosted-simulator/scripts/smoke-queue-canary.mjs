import assert from "node:assert/strict";

const base = process.env.SIMULATOR_URL ?? "http://127.0.0.1:4173";
const published = await fetch(`${base}/api/v1/queue-canary`, { method: "POST" });
const publishedBody = await published.json();
assert.equal(published.status, 202, JSON.stringify(publishedBody));
assert.equal(publishedBody.schemaVersion, "sigma-production-queue-canary-dispatch/1");
let status;
for (let attempt = 0; attempt < 30; attempt += 1) {
  const response = await fetch(`${base}/api/v1/queue-canary`, { cache: "no-store" });
  status = await response.json();
  assert.equal(response.status, 200, JSON.stringify(status));
  if (status.state === "verified_consumed") break;
  await new Promise((resolve) => setTimeout(resolve, 1000));
}
assert.equal(status.state, "verified_consumed", JSON.stringify(status));
assert.equal(status.deploymentIdentitySha256, publishedBody.deploymentIdentitySha256);
assert.equal(status.acknowledgementReference.sha256, publishedBody.acknowledgementReference.sha256);
console.log(JSON.stringify({
  schemaVersion: "sigma-production-queue-smoke/1",
  state: "pass",
  base,
  topic: status.topic,
  deploymentIdentitySha256: status.deploymentIdentitySha256,
  acknowledgementSha256: status.acknowledgementReference.sha256,
  privateVerifiedAcknowledgement: true,
}, null, 2));
