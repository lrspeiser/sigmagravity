import { datasetSummary } from "../../lib/catalog.mjs";
import { options, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "GET")) return;
  send(response, 200, { items: [datasetSummary()] });
}
