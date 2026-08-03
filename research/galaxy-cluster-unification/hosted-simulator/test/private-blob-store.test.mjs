import test from "node:test";
import assert from "node:assert/strict";
import {
  PrivateBlobStore,
  contentSha256,
  privateBlobStorageState,
  resolvePrivateBlobConfiguration,
} from "../lib/private-blob-store.mjs";

function fakeClient() {
  const objects = new Map();
  let puts = 0;
  return {
    objects,
    get puts() { return puts; },
    async head(pathname) {
      const found = objects.get(pathname);
      if (!found) {
        const error = new Error("missing");
        error.name = "BlobNotFoundError";
        throw error;
      }
      return { ...found.metadata };
    },
    async put(pathname, body, options) {
      if (objects.has(pathname) && !options.allowOverwrite) throw new Error("already exists");
      puts += 1;
      const bytes = Buffer.from(body);
      const metadata = {
        pathname,
        size: bytes.length,
        contentType: options.contentType,
        etag: `etag-${puts}`,
      };
      objects.set(pathname, { bytes, metadata });
      return metadata;
    },
    async get(pathname) {
      const found = objects.get(pathname);
      if (!found) return null;
      return {
        statusCode: 200,
        stream: new Blob([found.bytes]).stream(),
        blob: { ...found.metadata },
      };
    },
    async del(pathname) { objects.delete(pathname); },
  };
}

const environment = { BLOB_READ_WRITE_TOKEN: "x".repeat(48) };

test("private Blob configuration fails closed and never returns a partial credential", () => {
  assert.equal(resolvePrivateBlobConfiguration({}), null);
  assert.equal(privateBlobStorageState({}), "not_configured");
  assert.equal(privateBlobStorageState({ BLOB_READ_WRITE_TOKEN: "short" }), "misconfigured");
  assert.deepEqual(resolvePrivateBlobConfiguration(environment), {
    token: environment.BLOB_READ_WRITE_TOKEN,
  });
  assert.throws(
    () => resolvePrivateBlobConfiguration({ VERCEL_OIDC_TOKEN: "x".repeat(48) }),
    /configured together/,
  );
});

test("immutable private objects are content-addressed, idempotent, and rehashed", async () => {
  const client = fakeClient();
  const store = new PrivateBlobStore({ environment, client, maximumBytes: 1024 });
  const bytes = Buffer.from("gravity-independent durable object\n");
  const first = await store.putImmutable({
    namespace: "field-input",
    bytes,
    mediaType: "application/octet-stream",
    extension: "npz",
  });
  assert.equal(first.sha256, contentSha256(bytes));
  assert.equal(first.pathname, `sigma/v1/objects/field-input/sha256/${first.sha256}.npz`);
  assert.equal(client.puts, 1);
  const second = await store.putImmutable({
    namespace: "field-input",
    bytes,
    mediaType: "application/octet-stream",
    extension: "npz",
  });
  assert.equal(second.pathname, first.pathname);
  assert.equal(client.puts, 1);
  assert.deepEqual(await store.getVerified(first), bytes);
});

test("private object reads reject path, size, hash, and stored-byte mutations", async () => {
  const client = fakeClient();
  const store = new PrivateBlobStore({ environment, client, maximumBytes: 64 });
  const reference = await store.putImmutable({
    namespace: "artifact",
    bytes: Buffer.from("verified"),
    extension: "bin",
  });
  await assert.rejects(store.getVerified({ ...reference, pathname: "sigma/v1/objects/other" }), /pathname/);
  await assert.rejects(store.getVerified({ ...reference, bytes: reference.bytes + 1 }), /size/);
  await assert.rejects(store.getVerified({ ...reference, sha256: "0".repeat(64) }), /pathname/);
  client.objects.get(reference.pathname).bytes[0] ^= 0xff;
  await assert.rejects(store.getVerified(reference), /content verification/);
});

test("private object quotas and path grammar are enforced before upload", async () => {
  const client = fakeClient();
  const store = new PrivateBlobStore({ environment, client, maximumBytes: 4 });
  await assert.rejects(
    store.putImmutable({ namespace: "field-input", bytes: Buffer.alloc(5) }),
    /storage quota/,
  );
  await assert.rejects(
    store.putImmutable({ namespace: "../escape", bytes: Buffer.alloc(1) }),
    /namespace/,
  );
  assert.equal(client.puts, 0);
});
