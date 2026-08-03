BEGIN;

CREATE TABLE IF NOT EXISTS sigma_schema_migrations (
  migration_id text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT transaction_timestamp()
);

CREATE TABLE IF NOT EXISTS sigma_projects (
  project_id text PRIMARY KEY CHECK (project_id ~ '^project_[0-9a-f]{24}$'),
  slug text NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9][a-z0-9-]{0,62}$'),
  display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 120),
  state text NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'suspended', 'deleted')),
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT transaction_timestamp()
);

CREATE TABLE IF NOT EXISTS sigma_models (
  project_id text NOT NULL REFERENCES sigma_projects(project_id),
  model_sha256 text NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
  canonical_object_ref jsonb NOT NULL,
  confirmation_object_ref jsonb NOT NULL,
  confirmed_by text NOT NULL CHECK (char_length(confirmed_by) BETWEEN 1 AND 160),
  confirmed_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  PRIMARY KEY (project_id, model_sha256),
  CHECK (canonical_object_ref->>'schemaVersion' = 'sigma-private-blob-object/1'),
  CHECK (confirmation_object_ref->>'schemaVersion' = 'sigma-private-blob-object/1')
);

CREATE TABLE IF NOT EXISTS sigma_uploads (
  upload_id text PRIMARY KEY CHECK (upload_id ~ '^upload_[0-9a-f]{24}$'),
  project_id text NOT NULL REFERENCES sigma_projects(project_id),
  state text NOT NULL CHECK (state IN ('pending', 'ready', 'rejected', 'deleted')),
  bundle_sha256 text NOT NULL CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
  archive_sha256 text NOT NULL CHECK (archive_sha256 ~ '^[0-9a-f]{64}$'),
  archive_bytes bigint NOT NULL CHECK (archive_bytes BETWEEN 1 AND 10737418240),
  manifest_object_ref jsonb NOT NULL,
  archive_object_ref jsonb,
  scientific_roles jsonb NOT NULL DEFAULT '[]'::jsonb,
  license jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  ready_at timestamptz,
  UNIQUE (project_id, bundle_sha256, archive_sha256),
  CHECK (manifest_object_ref->>'schemaVersion' = 'sigma-private-blob-object/1'),
  CHECK (archive_object_ref IS NULL OR archive_object_ref->>'schemaVersion' = 'sigma-private-blob-object/1')
);

CREATE TABLE IF NOT EXISTS sigma_jobs (
  job_id text PRIMARY KEY CHECK (job_id ~ '^job_[0-9a-f]{24}$'),
  project_id text NOT NULL REFERENCES sigma_projects(project_id),
  job_type text NOT NULL CHECK (job_type IN ('field', 'galaxy', 'observation', 'inverse_response', 'batch', 'advanced_plugin')),
  state text NOT NULL CHECK (state IN ('dispatch_pending', 'queued', 'running', 'cancel_requested', 'succeeded', 'failed', 'cancelled')),
  idempotency_key text NOT NULL CHECK (char_length(idempotency_key) BETWEEN 8 AND 160),
  request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
  request_object_ref jsonb NOT NULL,
  model_sha256 text CHECK (model_sha256 IS NULL OR model_sha256 ~ '^[0-9a-f]{64}$'),
  input_upload_id text REFERENCES sigma_uploads(upload_id),
  parameter_policy jsonb NOT NULL,
  current_attempt integer NOT NULL DEFAULT 0 CHECK (current_attempt >= 0),
  max_attempts integer NOT NULL DEFAULT 4 CHECK (max_attempts BETWEEN 1 AND 12),
  event_sequence bigint NOT NULL DEFAULT 1 CHECK (event_sequence >= 1),
  lease_token text,
  lease_expires_at timestamptz,
  cancellation_requested_at timestamptz,
  queue_message_id text,
  result_manifest_ref jsonb,
  error jsonb,
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  updated_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  started_at timestamptz,
  finished_at timestamptz,
  UNIQUE (project_id, job_type, idempotency_key),
  CHECK (request_object_ref->>'schemaVersion' = 'sigma-private-blob-object/1'),
  CHECK (result_manifest_ref IS NULL OR result_manifest_ref->>'schemaVersion' = 'sigma-private-blob-object/1'),
  CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL)),
  CHECK ((state IN ('running', 'cancel_requested')) OR lease_token IS NULL)
);

CREATE TABLE IF NOT EXISTS sigma_job_events (
  job_id text NOT NULL REFERENCES sigma_jobs(job_id) ON DELETE CASCADE,
  sequence bigint NOT NULL CHECK (sequence >= 1),
  event_type text NOT NULL CHECK (event_type ~ '^[a-z][a-z0-9_]{0,63}$'),
  state text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  PRIMARY KEY (job_id, sequence)
);

CREATE TABLE IF NOT EXISTS sigma_job_attempts (
  job_id text NOT NULL REFERENCES sigma_jobs(job_id) ON DELETE CASCADE,
  attempt integer NOT NULL CHECK (attempt BETWEEN 1 AND 12),
  lease_token text NOT NULL,
  queue_message_id text,
  delivery_count integer NOT NULL CHECK (delivery_count >= 1),
  worker_identity text NOT NULL CHECK (char_length(worker_identity) BETWEEN 1 AND 256),
  state text NOT NULL CHECK (state IN ('running', 'retryable_failure', 'failed', 'succeeded', 'cancelled', 'lease_expired')),
  started_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  lease_expires_at timestamptz NOT NULL,
  finished_at timestamptz,
  error jsonb,
  PRIMARY KEY (job_id, attempt),
  UNIQUE (lease_token)
);

CREATE TABLE IF NOT EXISTS sigma_job_artifacts (
  job_id text NOT NULL REFERENCES sigma_jobs(job_id) ON DELETE CASCADE,
  name text NOT NULL CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$'),
  object_ref jsonb NOT NULL,
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  bytes bigint NOT NULL CHECK (bytes BETWEEN 0 AND 10737418240),
  media_type text NOT NULL CHECK (char_length(media_type) BETWEEN 3 AND 160),
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  PRIMARY KEY (job_id, name),
  CHECK (object_ref->>'schemaVersion' = 'sigma-private-blob-object/1')
);

CREATE TABLE IF NOT EXISTS sigma_outbox (
  outbox_id text PRIMARY KEY CHECK (outbox_id ~ '^outbox_[0-9a-f]{24}$'),
  project_id text NOT NULL REFERENCES sigma_projects(project_id),
  job_id text NOT NULL REFERENCES sigma_jobs(job_id) ON DELETE CASCADE,
  topic text NOT NULL CHECK (topic ~ '^[A-Za-z0-9_-]+$'),
  idempotency_key text NOT NULL UNIQUE CHECK (char_length(idempotency_key) BETWEEN 8 AND 180),
  payload jsonb NOT NULL,
  state text NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'publishing', 'published', 'dead')),
  publish_attempts integer NOT NULL DEFAULT 0 CHECK (publish_attempts >= 0),
  publish_lease_token text,
  publish_lease_expires_at timestamptz,
  queue_message_id text,
  last_error jsonb,
  next_attempt_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
  published_at timestamptz,
  CHECK ((publish_lease_token IS NULL) = (publish_lease_expires_at IS NULL)),
  CHECK ((state = 'publishing') OR publish_lease_token IS NULL)
);

CREATE INDEX IF NOT EXISTS sigma_jobs_project_created_idx
  ON sigma_jobs(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS sigma_jobs_recoverable_lease_idx
  ON sigma_jobs(state, lease_expires_at) WHERE state IN ('running', 'cancel_requested');
CREATE INDEX IF NOT EXISTS sigma_outbox_dispatch_idx
  ON sigma_outbox(state, next_attempt_at, created_at)
  WHERE state IN ('pending', 'publishing');

INSERT INTO sigma_schema_migrations(migration_id)
VALUES ('production-control-plane-v1')
ON CONFLICT (migration_id) DO NOTHING;

COMMIT;
