import { validateFieldModel } from "../../../lib/field-model.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const validation = validateFieldModel(parseBody(request));
    send(response, validation.valid ? 200 : 422, validation);
  } catch (error) {
    fail(response, error);
  }
}
