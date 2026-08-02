import { validateFormula } from "../../../lib/formula.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const validation = validateFormula(parseBody(request));
    const { evaluate: _evaluate, ...publicValidation } = validation;
    send(response, validation.valid ? 200 : 422, publicValidation);
  } catch (error) {
    fail(response, error);
  }
}
