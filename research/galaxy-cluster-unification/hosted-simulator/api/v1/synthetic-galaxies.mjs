import { createSyntheticGalaxy } from "../../lib/simulator.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    send(response, 200, createSyntheticGalaxy(parseBody(request)));
  } catch (error) {
    fail(response, error);
  }
}
