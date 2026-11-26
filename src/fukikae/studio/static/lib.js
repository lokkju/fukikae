// Shared pieces of the studio page: the htm binding, API calls with their errors, formatting,
// and media (clips fetched with the token header into blob URLs, one playing at a time).

import { h } from "./vendor/preact.module.js";
import { useEffect, useRef, useState } from "./vendor/hooks.module.js";
import htm from "./vendor/htm.module.js";
import { api, mediaUrl, MediaError, token } from "./token.js";

export const html = htm.bind(h);
export { token };

// The register bands (fukikae.pitch MALE_BELOW and FEMALE_ABOVE; test_static pins them).
export const MALE_BELOW = 165;
export const FEMALE_ABOVE = 200;
// D32's clone floor and IndexTTS2's prompt length, for the reference meter's marks.
export const FLOOR_CLIPS = 2;
export const FLOOR_S = 10;
export const PROMPT_S = 15;

// -- the API ---------------------------------------------------------------------------------

// A failed API call: the HTTP status and the server's {error, message, ...} body.
export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.message) || `HTTP ${status}`);
    this.status = status;
    this.body = body || {};
    this.code = this.body.error || String(status);
  }
}

// JSON from an API route; ApiError on any non-2xx answer.
export async function call(path, opts = {}) {
  let r;
  try {
    r = await api(path, opts);
  } catch (e) {
    throw new ApiError(0, { error: "network", message: `the studio didn't answer (${e.message}); is fukikae studio still running?` });
  }
  let body = null;
  const text = await r.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (e) {
      body = { error: "not-json", message: text.slice(0, 200) };
    }
  }
  if (!r.ok) throw new ApiError(r.status, body);
  return body;
}

export const post = (path, body) => call(path, { method: "POST", body: body ?? {} });

// How the page explains an error, by status and code: a title, the server's message and, for
// unfinished decision records, the command that finishes them.
export function explain(err) {
  if (!(err instanceof ApiError)) return { title: "Error", message: String(err && err.message || err) };
  const b = err.body;
  const m = b.message || "";
  switch (err.status) {
    case 0: return { title: "No answer", message: m };
    case 403: return { title: "Not authorized", message: "This tab has no valid studio token. Open the URL fukikae studio printed (it ends in #t=...)." };
    case 409:
      if (b.error === "stale") return { title: "Changed since the page loaded it", message: m + " The character was reloaded; check it and decide again." };
      if (b.error === "version-changed") return { title: "The draft changed", message: m };
      if (b.error === "character-merged") return { title: "Merged away", message: m, merged_into: b.merged_into };
      if (b.error === "incomplete") return { title: "Landed, steps left", message: m, resume: b.resume };
      if (b.error === "unfinished") return { title: "Unfinished decision records", message: m, resume: b.resume };
      return { title: `Refused (${b.error || 409})`, message: m };
    case 422: return { title: b.error === "unchecked" ? "Unchecked" : "Refused", message: m };
    case 423: return { title: "Busy", message: m || "A fukikae run or another writer holds the studio lock; try again when it finishes." };
    case 503: return { title: "Busy", message: m };
    default: return { title: `HTTP ${err.status}${b.error ? " " + b.error : ""}`, message: m };
  }
}

// An error box: the explanation, the resume hint, and an action button when given.
export function ErrorNote({ err, onAction, actionLabel }) {
  if (!err) return null;
  const x = explain(err);
  return html`<div class="note bad" role="alert">
    <b>${err.status ? err.status + " " : ""}${x.title}</b>
    <span>${x.message}</span>
    ${x.resume && html`<span>Finish them with <code class="cmd">${x.resume}</code></span>`}
    ${onAction && html`<button class="btn" onClick=${onAction}>${actionLabel || "Reload"}</button>`}
  </div>`;
}

// -- formatting ------------------------------------------------------------------------------

export function fs(s) {
  if (s == null) return "";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.round(s - 60 * m)).padStart(2, "0")}s`;
}

export function tc(s) {
  const m = Math.floor(s / 60);
  return `${m}:${(s - 60 * m).toFixed(1).padStart(4, "0")}`;
}

export const GN = { f: "female", m: "male", u: "unclear" };

export function charName(c) {
  return c ? (c.name || `char ${c.id}`) : "?";
}

export function clusters(c) {
  return [...new Set(Object.values(c.episodes).flatMap((e) => e.clusters))].join(", ");
}

// "reruns characters on S00E01, S00E02 · tts on S00E03", or "reruns nothing".
export function reruns(invalidations) {
  if (!invalidations || !invalidations.length) return "reruns nothing";
  const by = new Map();
  for (const i of invalidations) {
    if (!by.has(i.stage)) by.set(i.stage, []);
    by.get(i.stage).push(i.episode);
  }
  return "reruns " + [...by].map(([st, eps]) => `${st} on ${eps.join(", ")}`).join(" · ");
}

export function cls(...xs) {
  return xs.filter(Boolean).join(" ");
}

// -- navigation ----------------------------------------------------------------------------

// Move to a screen (#/<screen>[/<character>[/<language>]]).
export function go(screen, id = null, lang = null) {
  let h = `#/${screen}`;
  if (id != null) h += `/${id}`;
  if (lang) h += `/${encodeURIComponent(lang)}`;
  if (window.location.hash !== h) window.location.hash = h;
}

// The audition worker's state in one line.
export function gpuText(g) {
  if (!g) return "GPU: no answer";
  let s = `GPU ${g.index}`;
  if (g.loaded_backend) s += ` · ${g.loaded_backend} ${g.state === "rendering" ? "rendering" : "loaded"}`;
  else if (g.state !== "idle") s += ` · ${g.state}`;
  else s += " · idle";
  if (g.holder && !g.ours) s += ` · held by ${g.holder}`;
  if (g.queued) s += ` · ${g.queued} queued`;
  if (g.unload_in_s != null && g.loaded_backend) s += ` · unloads in ${Math.round(g.unload_in_s)} s`;
  return s;
}

// -- media -----------------------------------------------------------------------------------

// Blob URLs by media path, each with how many mounted players use it. A URL in use is never
// revoked; past CACHE_MAX the least recently used unused ones are.
const CACHE_MAX = 80;
const blobs = new Map(); // path -> {url, users}, oldest use first
const pending = new Map();

function evict() {
  for (const [k, e] of blobs) {
    if (blobs.size <= CACHE_MAX) return;
    if (e.users > 0) continue;
    blobs.delete(k);
    URL.revokeObjectURL(e.url);
  }
}

// A blob URL for a media path (/audio/..., /video/..., /audition/...), fetched once through
// token.js and held for the caller: every acquire() is paired with one release(path).
export function acquire(path) {
  const hit = blobs.get(path);
  if (hit) {
    blobs.delete(path);
    blobs.set(path, hit); // most recently used
    hit.users++;
    return Promise.resolve(hit.url);
  }
  if (!pending.has(path)) {
    const p = mediaUrl(path).then((url) => {
      blobs.set(path, { url, users: 0 });
      return url;
    }, (e) => {
      throw e instanceof MediaError ? new ApiError(e.status, e.body) : e;
    }).finally(() => pending.delete(path));
    pending.set(path, p);
  }
  return pending.get(path).then(() => {
    const e = blobs.get(path);
    if (!e) return acquire(path); // evicted before this caller held it: fetch again
    e.users++;
    evict();
    return e.url;
  });
}

// The caller no longer plays the path's URL; unused URLs past CACHE_MAX are revoked.
export function release(path) {
  const e = blobs.get(path);
  if (e) e.users = Math.max(0, e.users - 1);
  evict();
}

// One clip plays at a time: starting one pauses the one before.
let playing = null;
export function claim(el) {
  if (playing && playing !== el) {
    try { playing.pause(); } catch (e) { /* gone */ }
  }
  playing = el;
}

// At most two video fetches at once (the server cuts each on first fetch, two at a time).
const queue = [];
let active = 0;
function limited(fn) {
  return new Promise((resolve, reject) => {
    queue.push(() => fn().then(resolve, reject).finally(() => { active--; next(); }));
    next();
  });
}
function next() {
  while (active < 2 && queue.length) { active++; queue.shift()(); }
}

// play() refused: autoplay policy, an unplayable clip. Pausing a clip that was starting (one
// clip at a time) rejects with AbortError, which is not a failure.
function playFailed(e) {
  return e && e.name !== "AbortError";
}

// A sample's player: its video clip (picture and the original audio) when it has one, else its
// audio clip. Click to play; one at a time. `preload` fetches the video so its first frame shows.
// The player holds its blob URL while mounted (acquire) and lets go on unmount or a new path.
export function Clip({ video, clip, media, seconds, label, preload = false, small = false }) {
  const [url, setUrl] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [on, setOn] = useState(false);
  const el = useRef(null);
  const held = useRef(null); // the path whose URL this player holds
  const mounted = useRef(true);
  const path = video ? `/video/${video}` : clip ? `/audio/${clip}` : media || null;
  const current = useRef(path);
  current.current = path;
  const isVideo = Boolean(video);

  // a fresh acquire of p: keep it if it is still this player's path, else give it back
  function take(p, u) {
    if (!mounted.current || current.current !== p) { release(p); return false; }
    if (held.current === p) release(p); // already held once
    else {
      if (held.current) release(held.current);
      held.current = p;
    }
    setUrl(u);
    return true;
  }

  useEffect(() => () => {
    mounted.current = false;
    if (held.current) release(held.current);
    held.current = null;
  }, []);

  useEffect(() => {
    setUrl(null);
    setErr(null);
    if (held.current && held.current !== path) { release(held.current); held.current = null; }
    if (!preload || !path || !isVideo) return;
    limited(() => acquire(path)).then((u) => take(path, u),
      (e) => mounted.current && current.current === path && setErr(e));
  }, [path, preload]);

  function start() {
    const v = el.current;
    if (!v) return;
    claim(v);
    v.play().then(() => setErr(null), (e) => { if (playFailed(e)) setErr(e); });
  }

  async function toggle() {
    if (!path) return;
    if (el.current && url) {
      if (el.current.paused) start();
      else el.current.pause();
      return;
    }
    setBusy(true);
    try {
      const u = await acquire(path);
      if (take(path, u)) requestAnimationFrame(start);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  const unplayable = () => setErr(new Error("the browser can't play this clip"));
  const title = err ? explain(err).message : (label || (isVideo ? "play the clip" : "play the audio"));
  const dur = seconds != null ? `${seconds.toFixed(1)}s` : "";
  return html`<button class=${cls("clip", isVideo ? "v" : "a", small && "small", on && "on", err && "err")}
      onClick=${toggle} title=${title} aria-label=${title} disabled=${!path}>
    ${isVideo && url && html`<video ref=${el} src=${url} preload="metadata" playsinline onError=${unplayable}
        onPlay=${() => setOn(true)} onPause=${() => setOn(false)} onEnded=${() => setOn(false)}></video>`}
    ${!isVideo && url && html`<audio ref=${el} src=${url} onError=${unplayable}
        onPlay=${() => setOn(true)} onPause=${() => setOn(false)} onEnded=${() => setOn(false)}></audio>`}
    <span class="pb" aria-hidden="true">${busy ? "…" : err ? "!" : on ? "❚❚" : "▶"}</span>
    ${dur && html`<span class="dur">${dur}</span>`}
    ${!isVideo && !small && html`<span class="kind">audio</span>`}
  </button>`;
}

// A small play button for a wav path (an audition's /audition/<key>.wav).
export function Play({ path, label, seconds }) {
  return html`<${Clip} media=${path} label=${label} seconds=${seconds} small=${true} />`;
}

// localStorage that may refuse (a private window): conveniences only.
export const store = {
  get(k, d) {
    try { const v = window.localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; }
  },
  set(k, v) {
    try { window.localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignored */ }
  },
};
