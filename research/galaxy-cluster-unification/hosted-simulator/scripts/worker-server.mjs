import { resolve } from "node:path";
import { LocalFieldJobService } from "../lib/local-field-job-service.mjs";
import { createWorkerHttpServer } from "../lib/worker-http-server.mjs";

const hostedRoot = resolve(import.meta.dirname, "..");
const projectRoot = resolve(hostedRoot, "..");
const configuredStore = process.env.SIMULATOR_WORKER_STORE;
const storeRoot = resolve(configuredStore ?? resolve(projectRoot, "tmp", "container-field-worker"));
const host = process.env.HOST ?? "127.0.0.1";
const port = Number(process.env.PORT ?? 8787);

if (!Number.isSafeInteger(port) || port < 1 || port > 65535) {
  throw new Error("PORT must be an integer from 1 through 65535");
}

const service = new LocalFieldJobService({ root: storeRoot, projectRoot });
await service.initialize();
const server = createWorkerHttpServer({
  service,
  token: process.env.SIMULATOR_WORKER_TOKEN,
  persistentStoreConfigured: Boolean(configuredStore),
});

server.listen(port, host, () => {
  process.stdout.write(`Authenticated Sigma field worker listening on http://${host}:${port}\n`);
});

let closing = false;
async function close() {
  if (closing) return;
  closing = true;
  await new Promise((resolvePromise) => server.close(resolvePromise));
  await service.close();
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, async () => {
    await close();
    process.exit(0);
  });
}
