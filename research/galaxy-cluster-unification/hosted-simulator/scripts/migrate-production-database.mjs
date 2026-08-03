import { readFile } from "node:fs/promises";
import { NeonProductionDatabase, productionDatabaseState } from "../lib/production-database.mjs";

if (productionDatabaseState() !== "configured") {
  throw new Error("SIGMA_DATABASE_URL is not configured");
}
const database = new NeonProductionDatabase();
try {
  const migrations = [
    "production-control-plane-v1",
    "production-research-api-v2",
  ];
  for (const migrationId of migrations) {
    const migration = await readFile(new URL(`../sql/${migrationId}.sql`, import.meta.url), "utf8");
    await database.query(migration);
  }
  const migrationResult = await database.query(
    "SELECT migration_id FROM sigma_schema_migrations WHERE migration_id = ANY($1::text[]) ORDER BY migration_id",
    [migrations],
  );
  const tablesResult = await database.query(
    "SELECT count(*)::int AS table_count FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'sigma_%'",
  );
  if (migrationResult.rows?.length !== migrations.length || Number(tablesResult.rows?.[0]?.table_count) !== 11) {
    throw new Error("production control-plane migration verification failed");
  }
  console.log(JSON.stringify({
    schemaVersion: "sigma-production-database-migration/1",
    state: "pass",
    migrationIds: migrations,
    tableCount: 11,
  }, null, 2));
} finally {
  await database.close();
}
