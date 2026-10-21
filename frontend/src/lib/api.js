// Shared API helpers for every page.
//
// Relative by default: the vite dev server proxies these paths in development
// and nginx proxies them in the Docker image, so no absolute host is baked in.
// Set VITE_API_BASE_URL to point at a backend on another origin.
export const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

/** Surfaces the backend's `detail` message so a 4xx/5xx is legible in the UI. */
export async function readError(res) {
  const body = await res.text().catch(() => "");
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.detail === "string") return `${res.status}: ${parsed.detail}`;
    // FastAPI validation errors: a list of {loc, msg}.
    if (Array.isArray(parsed?.detail)) {
      return `${res.status}: ${parsed.detail.map((d) => d.msg).join("; ")}`;
    }
  } catch {
    /* not JSON; fall through to the raw body */
  }
  return `${res.status} ${res.statusText}${body ? `: ${body}` : ""}`;
}

async function request(path, init) {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw new Error(await readError(res));
  return res.status === 204 ? null : res.json();
}

export function fetchJSON(path) {
  return request(path);
}

export function postJSON(path, payload, apiKey) {
  const headers = { "Content-Type": "application/json" };
  // Sent per request from a value the operator types in. Deliberately not read
  // from import.meta.env: a VITE_ variable is compiled into the bundle and
  // would be served to every visitor.
  if (apiKey) headers["X-API-Key"] = apiKey;
  return request(path, { method: "POST", headers, body: JSON.stringify(payload) });
}

export function postForm(path, formData) {
  return request(path, { method: "POST", body: formData });
}

export function deleteJSON(path) {
  return request(path, { method: "DELETE" });
}

export function fmtLatency(ms) {
  if (ms === null || ms === undefined) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}
