import { ProductionControlPlane } from "../../lib/production-control-plane.mjs";
import { NeonProductionDatabase } from "../../lib/production-database.mjs";
import { processProductionJobMessage } from "../../lib/production-job-consumer.mjs";
import { createControlPlaneQueueNodeHandler } from "../../lib/production-queue.mjs";
import { StatelessWorkerClient } from "../../lib/stateless-worker-client.mjs";

export default createControlPlaneQueueNodeHandler(async (message, metadata) => {
  const database = new NeonProductionDatabase();
  try {
    const controlPlane = new ProductionControlPlane({ database });
    const executor = new StatelessWorkerClient();
    await processProductionJobMessage(message, metadata, {
      controlPlane,
      executor,
      workerIdentity: process.env.VERCEL_DEPLOYMENT_ID ?? process.env.VERCEL_URL ?? "unknown-vercel-deployment",
    });
  } finally {
    await database.close();
  }
});
