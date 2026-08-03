BEGIN;

ALTER TABLE sigma_projects
  ADD COLUMN IF NOT EXISTS max_active_jobs integer NOT NULL DEFAULT 8
    CHECK (max_active_jobs BETWEEN 1 AND 256),
  ADD COLUMN IF NOT EXISTS max_upload_bytes bigint NOT NULL DEFAULT 268435456
    CHECK (max_upload_bytes BETWEEN 1 AND 10737418240),
  ADD COLUMN IF NOT EXISTS max_attempts_per_job integer NOT NULL DEFAULT 4
    CHECK (max_attempts_per_job BETWEEN 1 AND 12);

CREATE TABLE IF NOT EXISTS sigma_project_api_keys (
  credential_id text PRIMARY KEY CHECK (credential_id ~ '^key_[0-9a-f]{24}$'),
  project_id text NOT NULL REFERENCES sigma_projects(project_id) ON DELETE CASCADE,
  token_sha256 text NOT NULL UNIQUE CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
  label text NOT NULL CHECK (char_length(label) BETWEEN 1 AND 120),
  state text NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'revoked')),
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  last_used_at timestamptz,
  revoked_at timestamptz,
  CHECK ((state = 'revoked') = (revoked_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS sigma_audit_events (
  audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id text NOT NULL REFERENCES sigma_projects(project_id) ON DELETE CASCADE,
  credential_id text REFERENCES sigma_project_api_keys(credential_id) ON DELETE SET NULL,
  action text NOT NULL CHECK (action ~ '^[a-z][a-z0-9_.-]{0,79}$'),
  resource_type text NOT NULL CHECK (resource_type ~ '^[a-z][a-z0-9_-]{0,39}$'),
  resource_id text CHECK (resource_id IS NULL OR char_length(resource_id) BETWEEN 1 AND 180),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp()
);

CREATE INDEX IF NOT EXISTS sigma_project_api_keys_project_idx
  ON sigma_project_api_keys(project_id, state);
CREATE INDEX IF NOT EXISTS sigma_audit_events_project_created_idx
  ON sigma_audit_events(project_id, created_at DESC, audit_id DESC);

INSERT INTO sigma_schema_migrations(migration_id)
VALUES ('production-research-api-v2')
ON CONFLICT (migration_id) DO NOTHING;

COMMIT;
