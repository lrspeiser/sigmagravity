import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import datasets from "../api/v1/datasets.mjs";
import formulasValidate from "../api/v1/formulas/validate.mjs";
import health from "../api/v1/health.mjs";
import runs from "../api/v1/runs.mjs";
import specification from "../api/v1/spec.mjs";
import syntheticGalaxies from "../api/v1/synthetic-galaxies.mjs";
import system from "../api/v1/system.mjs";
import systems from "../api/v1/systems.mjs";

const root = resolve(import.meta.dirname, "..");
const port = Number(process.env.PORT ?? 4173);
const host = process.env.HOST ?? "127.0.0.1";
const apiRoutes = new Map([
  ["/api/v1/health", health],
  ["/api/v1/datasets", datasets],
  ["/api/v1/systems", systems],
  ["/api/v1/formulas/validate", formulasValidate],
  ["/api/v1/synthetic-galaxies", syntheticGalaxies],
  ["/api/v1/runs", runs],
  ["/api/v1/openapi.json", specification],
]);
const staticFiles = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/assets/app.js", ["assets/app.js", "text/javascript; charset=utf-8"]],
  ["/assets/style.css", ["assets/style.css", "text/css; charset=utf-8"]],
]);

async function body(request) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.length;
    if (bytes > 1_000_000) throw new Error("request body exceeds 1 MB local limit");
    chunks.push(chunk);
  }
  if (!chunks.length) return undefined;
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function adaptResponse(response) {
  response.status = (code) => { response.statusCode = code; return response; };
  response.json = (payload) => { response.end(JSON.stringify(payload)); return response; };
  return response;
}

const server = createServer(async (request, rawResponse) => {
  const response = adaptResponse(rawResponse);
  const url = new URL(request.url, `http://${request.headers.host ?? `${host}:${port}`}`);
  const staticEntry = staticFiles.get(url.pathname);
  if (staticEntry) {
    const [path, contentType] = staticEntry;
    response.setHeader("Content-Type", contentType);
    response.setHeader("Cache-Control", "no-store");
    response.end(await readFile(resolve(root, path)));
    return;
  }
  if (url.pathname === "/favicon.ico") { response.statusCode = 204; response.end(); return; }

  let handler = apiRoutes.get(url.pathname);
  const detailMatch = url.pathname.match(/^\/api\/v1\/systems\/([^/]+)$/);
  if (detailMatch) handler = system;
  if (!handler) {
    response.statusCode = 404;
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.end(JSON.stringify({ error: "not_found" }));
    return;
  }
  try {
    request.query = Object.fromEntries(url.searchParams.entries());
    if (detailMatch) request.query.id = decodeURIComponent(detailMatch[1]);
    request.body = await body(request);
    handler(request, response);
  } catch (error) {
    if (!response.headersSent) response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.statusCode = 400;
    response.end(JSON.stringify({ error: "bad_request", message: error.message }));
  }
});

server.listen(port, host, () => {
  console.log(`Sigma Gravity simulator listening at http://${host}:${port}`);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
