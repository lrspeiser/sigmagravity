import { sha256 } from "../lib/canonical.mjs";
import { createProjectCredential } from "../lib/production-auth.mjs";
import { ProductionControlPlane } from "../lib/production-control-plane.mjs";
import { NeonProductionDatabase, productionDatabaseState } from "../lib/production-database.mjs";

if (productionDatabaseState() !== "configured") throw new Error("SIGMA_DATABASE_URL is not configured");
const slug = process.env.SIGMA_BOOTSTRAP_PROJECT_SLUG;
const displayName = process.env.SIGMA_BOOTSTRAP_PROJECT_NAME;
if (typeof slug !== "string" || !/^[a-z0-9][a-z0-9-]{0,62}$/.test(slug)) {
  throw new Error("SIGMA_BOOTSTRAP_PROJECT_SLUG is invalid");
}
if (typeof displayName !== "string" || displayName.length < 1 || displayName.length > 120) {
  throw new Error("SIGMA_BOOTSTRAP_PROJECT_NAME must contain 1-120 characters");
}
const projectId = `project_${sha256({ slug }).slice(0, 24)}`;
const database = new NeonProductionDatabase();
try {
  const controlPlane = new ProductionControlPlane({ database });
  const project = await controlPlane.createProject({ projectId, slug, displayName });
  const created = await createProjectCredential({
    database,
    projectId,
    label: "initial operator credential",
  });
  console.log(JSON.stringify({
    schemaVersion: "sigma-production-project-bootstrap/1",
    project,
    credential: created.credential,
    bearerToken: created.token,
    warning: "This bearer token is shown once. Store it in a secret manager; never commit it.",
  }, null, 2));
} finally {
  await database.close();
}
