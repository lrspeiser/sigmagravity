import { Pool } from "@neondatabase/serverless";

function databaseUrl(value) {
  if (typeof value !== "string" || value.length < 16 || value.length > 4096) {
    throw new Error("SIGMA_DATABASE_URL is invalid");
  }
  const url = new URL(value);
  if (!new Set(["postgres:", "postgresql:"]).has(url.protocol) || !url.hostname || !url.pathname.slice(1)) {
    throw new Error("SIGMA_DATABASE_URL must be a PostgreSQL connection URL");
  }
  return value;
}

export function resolveProductionDatabaseConfiguration(environment = process.env) {
  const connectionString = environment.SIGMA_DATABASE_URL;
  if (!connectionString) return null;
  return { connectionString: databaseUrl(connectionString) };
}

export function productionDatabaseState(environment = process.env) {
  try {
    return resolveProductionDatabaseConfiguration(environment) ? "configured" : "not_configured";
  } catch {
    return "misconfigured";
  }
}

export async function readProductionDatabaseReadiness({ environment = process.env, database = null } = {}) {
  const configured = productionDatabaseState(environment);
  if (configured !== "configured" && !database) return { state: configured, migrations: [] };
  const owned = database === null;
  const connection = database ?? new NeonProductionDatabase({ environment });
  try {
    const result = await connection.query(
      `SELECT migration_id FROM sigma_schema_migrations
        WHERE migration_id = ANY($1::text[]) ORDER BY migration_id`,
      [["production-control-plane-v1", "production-research-api-v2"]],
    );
    const migrations = (result.rows ?? result).map((row) => row.migration_id);
    return {
      state: migrations.length === 2 ? "verified_migrated" : "migration_required",
      migrations,
    };
  } catch {
    return { state: "verification_failed", migrations: [] };
  } finally {
    if (owned) await connection.close();
  }
}

export class NeonProductionDatabase {
  constructor({ environment = process.env, pool } = {}) {
    const configuration = resolveProductionDatabaseConfiguration(environment);
    if (!configuration && !pool) throw new Error("production Postgres is not configured");
    this.pool = pool ?? new Pool({ connectionString: configuration.connectionString });
  }

  query(text, parameters = []) {
    return this.pool.query(text, parameters);
  }

  async transaction(callback) {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const value = await callback({ query: (text, parameters = []) => client.query(text, parameters) });
      await client.query("COMMIT");
      return value;
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  async close() {
    await this.pool.end();
  }
}
