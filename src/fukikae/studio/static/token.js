// The studio's token. The start-up URL carries it in the fragment (/#t=<token>), which a
// browser never sends to a server; this moves it into sessionStorage (this tab only), clears
// the fragment from the address bar and history, and sends it as the X-Fukikae-Token header
// on every API and media request. It never goes in a query string or lasting storage, and this
// is the only module that calls fetch() or touches sessionStorage. An <audio> or <video>
// element can't send a header, so mediaUrl() fetches a clip with it into a blob URL.

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

// A media path the page plays: an /audio clip, a /video clip or an /audition render.
const MEDIA = /^\/(audio|video|audition)\/[^/?#]+$/;

// A media fetch the server refused: its HTTP status and {error, message} body.
export class MediaError extends Error {
  constructor(status, body) {
    super((body && body.message) || `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

// A blob URL for a media path (MEDIA), fetched with the token header: the only way the page
// gets audio or video, since an <audio> or <video> element can't send a header. Revoke it
// when done (lib.js counts who still plays it).
export async function mediaUrl(path) {
  if (!MEDIA.test(path)) throw new Error(`not a media path: ${path}`);
  const r = await api(path);
  if (!r.ok) {
    let body = null;
    try { body = await r.json(); } catch (e) { /* not JSON */ }
    throw new MediaError(r.status, body);
  }
  return URL.createObjectURL(await r.blob());
}

// A blob URL for a clip (a sample's clip field: <sha256>.wav); revoke it when done.
export function audioUrl(clip) {
  return mediaUrl(`/audio/${encodeURIComponent(clip)}`);
}

// A blob URL for a sample's video clip (its video field: <sha256>.mp4, cut by the server on
// first fetch, so the first one takes a second or so); revoke it when done.
export function videoUrl(clip) {
  return mediaUrl(`/video/${encodeURIComponent(clip)}`);
}
