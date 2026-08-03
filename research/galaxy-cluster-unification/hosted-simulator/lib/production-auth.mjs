import { createHash, randomBytes } from "node:crypto";

const PROJECT_ID = /^project_[0-9a-f]{24}$/;
const CREDENTIAL_ID = /^key_[0-9a-f]{24}$/;
const TOKEN = /^sgp_[A-Za-z0-9_-]{43}$/;

function rows(result) {
  if (Array.isArray(result)) return result;
  return result?.rows ?? [];
}

function one(result) {
  return rows(result)[0] ?? null;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function header(request, name) {
  if (request?.headers?.get) return request.headers.get(name);
  const expected = name.toLowerCase();
  for (const [key, value] of Object.entries(request?.headers ?? {})) {
    if (key.toLowerCase() === expected) return Array.isArray(value) ? value[0] : value;
  }
  return undefined;
}

export class ProductionAuthError extends Error {
  constructor(code, message, statusCode = 401) {
    super(message);
    this.name = "ProductionAuthError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

export function bearerToken(request) {
  const authorization = header(request, "authorization");
  if (typeof authorization !== "string" || !authorization.startsWith("Bearer ")) {
    throw new ProductionAuthError("authentication_required", "A project bearer credential is required");
  }
  const token = authorization.slice(7);
  if (!TOKEN.test(token)) {
    throw new ProductionAuthError("invalid_credential", "The project bearer credential is invalid");
  }
  return token;
}

export async function createProjectCredential({
  database,
  projectId,
  label,
  randomBytesImpl = randomBytes,
} = {}) {
  if (!database?.query) throw new Error("a database is required");
  if (typeof projectId !== "string" || !PROJECT_ID.test(projectId)) {
    throw new ProductionAuthError("invalid_project", "project identifier is invalid", 422);
  }
  if (typeof label !== "string" || label.length < 1 || label.length > 120) {
    throw new ProductionAuthError("invalid_label", "credential label must contain 1-120 characters", 422);
  }
  const token = `sgp_${randomBytesImpl(32).toString("base64url")}`;
  if (!TOKEN.test(token)) throw new Error("generated project credential is invalid");
  const tokenSha256 = sha256(token);
  const credentialId = `key_${tokenSha256.slice(0, 24)}`;
  const result = await database.query(
    `INSERT INTO sigma_project_api_keys(credential_id, project_id, token_sha256, label)
     VALUES ($1, $2, $3, $4)
     RETURNING credential_id, project_id, label, state, created_at`,
    [credentialId, projectId, tokenSha256, label],
  );
  const record = one(result);
  if (!record) throw new Error("project credential was not created");
  return {
    token,
    credential: {
      id: record.credential_id,
      projectId: record.project_id,
      label: record.label,
      state: record.state,
      createdAt: new Date(record.created_at).toISOString(),
    },
  };
}

export async function authenticateProjectRequest(request, database) {
  if (!database?.query) throw new Error("a database is required");
  const tokenSha256 = sha256(bearerToken(request));
  const record = one(await database.query(
    `SELECT credentials.credential_id, credentials.project_id, credentials.label,
            projects.slug, projects.display_name, projects.state,
            projects.max_active_jobs, projects.max_upload_bytes, projects.max_attempts_per_job
       FROM sigma_project_api_keys AS credentials
       JOIN sigma_projects AS projects ON projects.project_id = credentials.project_id
      WHERE credentials.token_sha256 = $1
        AND credentials.state = 'active'
        AND projects.state = 'active'`,
    [tokenSha256],
  ));
  if (!record) {
    throw new ProductionAuthError("invalid_credential", "The project bearer credential is invalid");
  }
  await database.query(
    "UPDATE sigma_project_api_keys SET last_used_at = transaction_timestamp() WHERE credential_id = $1",
    [record.credential_id],
  );
  return {
    credentialId: record.credential_id,
    project: {
      id: record.project_id,
      slug: record.slug,
      displayName: record.display_name,
      state: record.state,
      limits: {
        maxActiveJobs: Number(record.max_active_jobs),
        maxUploadBytes: Number(record.max_upload_bytes),
        maxAttemptsPerJob: Number(record.max_attempts_per_job),
      },
    },
  };
}

export async function revokeProjectCredential({ database, projectId, credentialId }) {
  if (!PROJECT_ID.test(projectId ?? "") || !CREDENTIAL_ID.test(credentialId ?? "")) {
    throw new ProductionAuthError("invalid_credential", "credential identifier is invalid", 422);
  }
  const result = await database.query(
    `UPDATE sigma_project_api_keys
        SET state = 'revoked', revoked_at = transaction_timestamp()
      WHERE project_id = $1 AND credential_id = $2 AND state = 'active'
      RETURNING credential_id`,
    [projectId, credentialId],
  );
  return Boolean(one(result));
}

export { header as requestHeader };
