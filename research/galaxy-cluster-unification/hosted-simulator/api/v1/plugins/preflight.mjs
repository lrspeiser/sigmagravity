import { verifyAdvancedPluginManifest } from "../../../lib/advanced-plugin.mjs";
import { fail, options, parseBody, requireMethod, send } from "../../../lib/http.mjs";

export default function handler(request, response) {
  if (options(request, response) || !requireMethod(request, response, "POST")) return;
  try {
    const body = parseBody(request);
    if (body?.schemaVersion !== "sigma-advanced-plugin-preflight-request/1") {
      const error = new Error("plug-in preflight must use sigma-advanced-plugin-preflight-request/1");
      error.code = "invalid_request";
      throw error;
    }
    const verification = verifyAdvancedPluginManifest(body.manifest, {
      publicKeyPem: body.publicKeyPem,
      requireTrustedPublisher: false,
    });
    send(response, 200, {
      schemaVersion: "sigma-advanced-plugin-preflight/1",
      ...verification,
      publisherTrust: "self_signature_valid_not_operator_trusted",
      packageBytesVerified: false,
      executableInVercel: false,
      nextGate: "the isolated worker must rehash every package file and require an active operator trust-store record before container creation",
    });
  } catch (error) {
    fail(response, error, 422);
  }
}
