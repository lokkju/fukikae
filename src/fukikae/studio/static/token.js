// The studio's token. The start-up URL carries it in the fragment (/#t=<token>), which a
// browser never sends to a server; this moves it into sessionStorage (this tab only), clears
// the fragment from the address bar and history, and sends it as the X-Fukikae-Token header
// on every API and audio request. It never goes in a query string, and the server stores
// nothing in the browser. An <audio> element can't send a header, so audioUrl() fetches a
// clip with it and hands back a blob URL.

const KEY = "fukikae-studio-token";
const HEADER = "X-Fukikae-Token";

function readToken() {
  const m = /^#(?:t=|.*&t=)([^&]*)/.exec(window.location.hash);
  if (m) {
    try {
      window.sessionStorage.setItem(KEY, decodeURIComponent(m[1]));
    } catch (e) {
      // storage refused (a private window); the token lives only in this page load
      readToken.fallback = decodeURIComponent(m[1]);
    }
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  try {
    return window.sessionStorage.getItem(KEY) || readToken.fallback || null;
  } catch (e) {
    return readToken.fallback || null;
  }
}

export const token = readToken();

// fetch() with the token header; a plain-object body is sent as JSON.
export async function api(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (token) headers.set(HEADER, token);
  let body = opts.body;
  if (body !== undefined && body !== null && typeof body === "object" &&
      !(body instanceof Blob) && !(body instanceof FormData)) {
    body = JSON.stringify(body);
    headers.set("Content-Type", "application/json");
  }
  return fetch(path, { ...opts, headers, body, credentials: "omit", cache: "no-store" });
}

// A blob URL for a clip (a sample's clip field: <sha256>.wav); revoke it when done.
export async function audioUrl(clip) {
  const r = await api(`/audio/${encodeURIComponent(clip)}`);
  if (!r.ok) throw new Error(`audio ${clip}: HTTP ${r.status}`);
  return URL.createObjectURL(await r.blob());
}
