import { createHash } from "node:crypto";
import {
  BlobNotFoundError,
  del as blobDelete,
  get as blobGet,
  head as blobHead,
  put as blobPut,
} from "@vercel/blob";

const DEFAULT_MAXIMUM_BYTES = 256 * 1024 * 1024;
const SEGMENT = /^[a-z0-9][a-z0-9-]{0,63}$/;
const EXTENSION = /^[a-z0-9]{1,12}$/;
const SHA256 = /^[0-9a-f]{64}$/;

function positiveInteger(value, fallback, label) {
  const result = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return result;
}

function tokenIsValid(value) {
  return typeof value === "string" && Buffer.byteLength(value, "utf8") >= 32;
}

export function resolvePrivateBlobConfiguration(environment = process.env) {
  const token = environment.BLOB_READ_WRITE_TOKEN;
  const oidcToken = environment.VERCEL_OIDC_TOKEN;
  const storeId = environment.BLOB_STORE_ID;
  if (!token && !oidcToken && !storeId) return null;
  if (token) {
    if (!tokenIsValid(token)) {
      throw new Error("BLOB_READ_WRITE_TOKEN must contain at least 32 UTF-8 bytes");
    }
    return { token };
  }
  if (!oidcToken || !storeId) {
    throw new Error("VERCEL_OIDC_TOKEN and BLOB_STORE_ID must be configured together");
  }
  if (!tokenIsValid(oidcToken) || !/^store_[A-Za-z0-9]+$/.test(storeId)) {
    throw new Error("Vercel Blob OIDC configuration is invalid");
  }
  return { oidcToken, storeId };
}

export function privateBlobStorageState(environment = process.env) {
  try {
    return resolvePrivateBlobConfiguration(environment) ? "configured" : "not_configured";
  } catch {
    return "misconfigured";
  }
}

export function contentSha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function validatedSegment(value, label) {
  if (typeof value !== "string" || !SEGMENT.test(value)) {
    throw new Error(`${label} must match ${SEGMENT}`);
  }
  return value;
}

function validatedExtension(value) {
  if (typeof value !== "string" || !EXTENSION.test(value)) {
    throw new Error(`extension must match ${EXTENSION}`);
  }
  return value;
}

function objectPath(namespace, sha256, extension) {
  return `sigma/v1/objects/${namespace}/sha256/${sha256}.${extension}`;
}

function isNotFound(error) {
  return error instanceof BlobNotFoundError || error?.name === "BlobNotFoundError";
}

async function streamBytes(stream, maximumBytes) {
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maximumBytes) {
      await reader.cancel("private blob exceeded configured read quota");
      throw new RangeError(`private blob exceeds the ${maximumBytes} byte read quota`);
    }
    chunks.push(Buffer.from(value));
  }
  return Buffer.concat(chunks, total);
}

export class PrivateBlobStore {
  constructor({
    environment = process.env,
    client = { del: blobDelete, get: blobGet, head: blobHead, put: blobPut },
    maximumBytes,
  } = {}) {
    const configuration = resolvePrivateBlobConfiguration(environment);
    if (!configuration) throw new Error("private Vercel Blob storage is not configured");
    this.authentication = configuration;
    this.client = client;
    this.maximumBytes = positiveInteger(
      maximumBytes ?? environment.SIMULATOR_BLOB_MAX_OBJECT_BYTES,
      DEFAULT_MAXIMUM_BYTES,
      "SIMULATOR_BLOB_MAX_OBJECT_BYTES",
    );
  }

  async #head(pathname) {
    try {
      return await this.client.head(pathname, this.authentication);
    } catch (error) {
      if (isNotFound(error)) return null;
      throw error;
    }
  }

  async putImmutable({ namespace, bytes, mediaType = "application/octet-stream", extension = "bin" }) {
    validatedSegment(namespace, "namespace");
    validatedExtension(extension);
    const content = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
    if (content.length > this.maximumBytes) {
      throw new RangeError(`object exceeds the ${this.maximumBytes} byte storage quota`);
    }
    if (typeof mediaType !== "string" || !/^[A-Za-z0-9.+-]+\/[A-Za-z0-9.+-]+$/.test(mediaType)) {
      throw new Error("mediaType must be a simple MIME type");
    }
    const sha256 = contentSha256(content);
    const pathname = objectPath(namespace, sha256, extension);
    let metadata = await this.#head(pathname);
    if (!metadata) {
      try {
        await this.client.put(pathname, content, {
          ...this.authentication,
          access: "private",
          addRandomSuffix: false,
          allowOverwrite: false,
          cacheControlMaxAge: 60,
          contentType: mediaType,
          maximumSizeInBytes: this.maximumBytes,
        });
      } catch (error) {
        metadata = await this.#head(pathname);
        if (!metadata) throw error;
      }
    }
    const reference = {
      schemaVersion: "sigma-private-blob-object/1",
      provider: "vercel_blob",
      access: "private",
      namespace,
      extension,
      pathname,
      sha256,
      bytes: content.length,
      mediaType,
    };
    const downloaded = await this.getVerified(reference);
    if (!downloaded.equals(content)) {
      throw new Error("content-addressed object bytes do not match the submitted bytes");
    }
    const verifiedMetadata = await this.#head(pathname);
    return { ...reference, etag: verifiedMetadata?.etag ?? null };
  }

  async getVerified(reference) {
    if (!reference || reference.schemaVersion !== "sigma-private-blob-object/1") {
      throw new Error("private blob reference schema is invalid");
    }
    const namespace = validatedSegment(reference.namespace, "namespace");
    const extension = validatedExtension(reference.extension);
    if (typeof reference.sha256 !== "string" || !SHA256.test(reference.sha256)) {
      throw new Error("private blob reference SHA-256 is invalid");
    }
    if (!Number.isSafeInteger(reference.bytes) || reference.bytes < 0 || reference.bytes > this.maximumBytes) {
      throw new Error("private blob reference byte count is invalid");
    }
    const expectedPath = objectPath(namespace, reference.sha256, extension);
    if (reference.pathname !== expectedPath) {
      throw new Error("private blob reference pathname is not content-addressed correctly");
    }
    const result = await this.client.get(reference.pathname, {
      ...this.authentication,
      access: "private",
      useCache: false,
    });
    if (!result || result.statusCode !== 200 || !result.stream) {
      throw new Error("private blob object is unavailable");
    }
    if (result.blob.size !== reference.bytes || result.blob.size > this.maximumBytes) {
      throw new Error("private blob object size does not match its immutable reference");
    }
    const content = await streamBytes(result.stream, this.maximumBytes);
    if (content.length !== reference.bytes || contentSha256(content) !== reference.sha256) {
      throw new Error("private blob object failed content verification");
    }
    return content;
  }

  async deleteForTest(reference) {
    await this.client.del(reference.pathname, this.authentication);
  }
}
