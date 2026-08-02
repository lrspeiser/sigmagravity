import { prepareFieldJob } from "../../../lib/field-job-preflight.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const result = prepareFieldJob(parseBody(request));
    send(response, result.valid ? 200 : 422, result);
  } catch (error) {
    fail(response, error);
  }
}
