import { listSystems } from "../../lib/catalog.mjs";
import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  const limit = Math.min(175, Math.max(1, Number(request.query.limit ?? 175)));
  const offset = Math.max(0, Number(request.query.offset ?? 0));
  const all = listSystems({
    query: request.query.q,
    morphology: request.query.morphology,
    quality: request.query.quality,
  });
  send(response, 200, {
    items: all.slice(offset, offset + limit),
    page: { total: all.length, offset, limit },
  });
}
