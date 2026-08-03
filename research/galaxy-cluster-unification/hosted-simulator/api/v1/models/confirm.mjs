import { confirmFieldModel } from "../../../lib/field-model.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const body = parseBody(request);
    if (body?.schemaVersion !== "sigma-model-confirmation-request/1") {
      const error = new Error("model confirmation must use sigma-model-confirmation-request/1");
      error.code = "invalid_request";
      error.statusCode = 422;
      throw error;
    }
    const receipt = confirmFieldModel(body?.model, {
      expectedModelSha256: body?.expectedModelSha256,
      acknowledgement: body?.acknowledgement,
    });
    send(response, 200, receipt);
  } catch (error) {
    fail(response, error, error.statusCode ?? 400);
  }
}
