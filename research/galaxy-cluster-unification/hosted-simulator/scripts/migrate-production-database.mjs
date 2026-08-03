import { readFile } from "node:fs/promises";
import { NeonProductionDatabase, productionDatabaseState } from "../lib/production-database.mjs";

if (productionDatabaseState() !== "configured") {
  throw new Error("SIGMA_DATABASE_URL is not configured");
}
const database = new NeonProductionDatabase();
try {
  const migration = await readFile(new URL("../sql/production-control-plane-v1.sql", import.meta.url), "utf8");
  await database.query(migration);
  const migrationResult = await database.query(
    "SELECT migration_id FROM sigma_schema_migrations WHERE migration_id = $1",
    ["production-control-plane-v1"],
  );
  const tablesResult = await database.query(
    "SELECT count(*)::int AS table_count FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'sigma_%'",
  );
  if (migrationResult.rows?.length !== 1 || Number(tablesResult.rows?.[0]?.table_count) !== 9) {
    throw new Error("production control-plane migration verification failed");
  }
  console.log(JSON.stringify({
    schemaVersion: "sigma-production-database-migration/1",
    state: "pass",
    migrationId: "production-control-plane-v1",
    tableCount: 9,
  }, null, 2));
} finally {
  await database.close();
}
