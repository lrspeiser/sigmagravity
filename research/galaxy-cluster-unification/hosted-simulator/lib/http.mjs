export function setCors(response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Content-Type", "application/json; charset=utf-8");
}

export function send(response, status, body) {
  setCors(response);
  response.status(status).json(body);
}

export function options(request, response) {
  if (request.method !== "OPTIONS") return false;
  setCors(response);
  response.status(204).end();
  return true;
}

export function requireMethod(request, response, method) {
  if (request.method === method) return true;
  response.setHeader("Allow", `OPTIONS, ${method}`);
  send(response, 405, { error: "method_not_allowed", allowed: [method] });
  return false;
}

export function parseBody(request) {
  if (request.body && typeof request.body === "object") return request.body;
  if (typeof request.body === "string" && request.body.length) return JSON.parse(request.body);
  return {};
}

export function fail(response, error, status = 422) {
  send(response, status, {
    error: status === 422 ? "invalid_request" : "server_error",
    message: error.message,
    details: error.details ?? [],
  });
}
