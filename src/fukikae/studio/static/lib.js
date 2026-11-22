// Shared pieces of the studio page: the htm binding, API calls with their errors, formatting,
// and media (clips fetched with the token header into blob URLs, one playing at a time).

import { h } from "./vendor/preact.module.js";
import { useEffect, useRef, useState } from "./vendor/hooks.module.js";
import htm from "./vendor/htm.module.js";
import { api, audioUrl, videoUrl, token } from "./token.js";

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

// Blob URLs by path, the oldest revoked past CACHE_MAX.
const CACHE_MAX = 80;
const blobs = new Map();
const pending = new Map();

async function fetchBlob(path) {
  const r = await api(path);
  if (!r.ok) {
    let body = null;
    try { body = await r.json(); } catch (e) { /* not JSON */ }
    throw new ApiError(r.status, body);
  }
  return URL.createObjectURL(await r.blob());
}

// A blob URL for a media path (/audio/..., /video/..., /audition/...), fetched once.
export function blobUrl(path) {
  if (blobs.has(path)) {
    const u = blobs.get(path);
    blobs.delete(path);
    blobs.set(path, u);
    return Promise.resolve(u);
  }
  if (pending.has(path)) return pending.get(path);
  let p;
  if (path.startsWith("/audio/")) p = audioUrl(path.slice(7));
  else if (path.startsWith("/video/")) p = videoUrl(path.slice(7));
  else p = fetchBlob(path);
  p = p.then((u) => {
    blobs.set(path, u);
    while (blobs.size > CACHE_MAX) {
      const [k, old] = blobs.entries().next().value;
      blobs.delete(k);
      URL.revokeObjectURL(old);
    }
    return u;
  }).finally(() => pending.delete(path));
  pending.set(path, p);
  return p;
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

// A sample's player: its video clip (picture and the original audio) when it has one, else its
// audio clip. Click to play; one at a time. `preload` fetches the video so its first frame shows.
export function Clip({ video, clip, src, seconds, label, preload = false, small = false }) {
  const [url, setUrl] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [on, setOn] = useState(false);
  const el = useRef(null);
  const path = video ? `/video/${video}` : clip ? `/audio/${clip}` : src || null;
  const isVideo = Boolean(video);

  useEffect(() => {
    setUrl(null);
    setErr(null);
    if (!preload || !path || !isVideo) return;
    let live = true;
    limited(() => blobUrl(path)).then((u) => live && setUrl(u), (e) => live && setErr(e));
    return () => { live = false; };
  }, [path, preload]);

  async function toggle() {
    if (!path) return;
    if (el.current && url) {
      if (el.current.paused) { claim(el.current); el.current.play().catch(() => {}); }
      else el.current.pause();
      return;
    }
    setBusy(true);
    try {
      const u = await blobUrl(path);
      setUrl(u);
      requestAnimationFrame(() => {
        if (el.current) { claim(el.current); el.current.play().catch(() => {}); }
      });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  const title = err ? explain(err).message : (label || (isVideo ? "play the clip" : "play the audio"));
  const dur = seconds != null ? `${seconds.toFixed(1)}s` : "";
  return html`<button class=${cls("clip", isVideo ? "v" : "a", small && "small", on && "on", err && "err")}
      onClick=${toggle} title=${title} aria-label=${title} disabled=${!path}>
    ${isVideo && url && html`<video ref=${el} src=${url} preload="metadata" playsinline
        onPlay=${() => setOn(true)} onPause=${() => setOn(false)} onEnded=${() => setOn(false)}></video>`}
    ${!isVideo && url && html`<audio ref=${el} src=${url}
        onPlay=${() => setOn(true)} onPause=${() => setOn(false)} onEnded=${() => setOn(false)}></audio>`}
    <span class="pb" aria-hidden="true">${busy ? "…" : on ? "❚❚" : "▶"}</span>
    ${dur && html`<span class="dur">${dur}</span>`}
    ${!isVideo && !small && html`<span class="kind">audio</span>`}
  </button>`;
}

// A small play button for a wav path (an audition's /audition/<key>.wav).
export function Play({ path, label, seconds }) {
  return html`<${Clip} src=${path} label=${label} seconds=${seconds} small=${true} />`;
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
