// The voice screen: one character's voice draft for one target language. References on the
// episode timeline and the working selection (with the floor meter), the candidates, the
// backend's own voice keys, versions by lines (each cell an audition), compare side by side or
// blind, the audition queue with the GPU, and accept. Every voice here is a clone of the
// character's own speech (D41): no control offers any other kind.

import { useCallback, useEffect, useMemo, useRef, useState } from "./vendor/hooks.module.js";
import {
  html, call, post, ErrorNote, Clip, Play, fs, tc, charName, clusters, reruns, cls, go, gpuText,
  store, FLOOR_CLIPS, FLOOR_S, PROMPT_S,
} from "./lib.js";

const PENDING = new Set(["queued", "waiting", "loading", "running"]);
const CAND_PAGE = 200;

const spanKey = (r) => `${r.episode}|${Number(r.start).toFixed(3)}|${Number(r.end).toFixed(3)}`;
const span = (c) => ({ episode: c.episode, start: c.start, end: c.end });
// An audition line as the API takes it (the grid's rows carry display fields beside it).
const plainLine = (l) => (l.kind === "real" ? { kind: "real", episode: l.episode, cue_id: l.cue_id }
  : { kind: "typed", text: l.text, emotion_from: l.emotion_from || null });
export const lineKey = (l) =>
  l.kind === "real" ? `r|${l.episode}|${l.cue_id}` : `t|${l.text}|${l.emotion_from ? l.emotion_from.episode + "/" + l.emotion_from.cue_id : ""}`;

const REASONS = {
  length: "too short or long", overlap: "overlaps another speaker", excluded: "song or tagged singing",
  foreign: "another character's subtitle name", no_text: "no transcript", sparse: "sparse text",
  off_register: "off register", unvoiced: "mostly unvoiced", song_like: "song-like", repeat: "audio already in a joined candidate",
  split: "outside the name-confirmed turns", scene: "same scene as a better clip",
};

// The selection against the floor, worked out here for an unsaved selection (the server's
// meter is the authority for a saved version).
function meter(refs) {
  const clips = refs.map((r) => r.end - r.start);
  const seconds = clips.reduce((a, x) => a + x, 0);
  const floor = clips.length >= FLOOR_CLIPS && seconds >= FLOOR_S;
  return { clips: clips.length, seconds, floor };
}

export function VoicePicker({ doc }) {
  const byId = new Map(doc.characters.map((c) => [c.id, c]));
  return html`<main class="screen picker">
    <div class="hd"><h2>Voices</h2><span class="dim">Pick a character to tune its voice; every voice is a clone of the character's own speech.</span></div>
    <div class="pane box">
      <h4>Queue</h4>
      <div class="scroll">
        <table class="tight">
          <thead><tr><th>character</th><th>clusters</th><th>speech</th><th>cards</th><th></th></tr></thead>
          <tbody>${doc.queue.map((id) => byId.get(id)).filter(Boolean).map((c) => html`<tr>
            <td>${charName(c)}</td><td class="mono">${clusters(c)}</td><td class="mono num">${fs(c.seconds)}</td>
            <td class="mono small">${c.cards.map((k) => `${k.language} ${k.chosen_by}${k.locked ? " locked" : ""}`).join(" · ") || "none"}</td>
            <td><button class="btn" onClick=${() => go("voice", c.id)}>Tune</button></td></tr>`)}</tbody>
        </table>
      </div>
    </div>
  </main>`;
}

function Timeline({ cands, chosen, showGated, onPick }) {
  const eps = [...new Set(cands.map((c) => c.episode))];
  const [ep, setEp] = useState(null);
  const cur = ep && eps.includes(ep) ? ep : eps[0];
  const here = cands.filter((c) => c.episode === cur && (showGated || c.rank != null));
  const len = Math.max(60, ...cands.filter((c) => c.episode === cur).map((c) => c.end));
  const step = len > 1800 ? 600 : len > 600 ? 300 : 60;
  const axis = [];
  for (let t = 0; t < len; t += step) axis.push(t);
  const chosenKeys = new Set(chosen.map(spanKey));
  return html`<div class="tlwrap">
    ${eps.length > 1 && html`<div class="row">${eps.map((e) => html`<button class=${cls("btn", e === cur && "pri")} onClick=${() => setEp(e)}>${e}</button>`)}</div>`}
    <div class="tl" role="img" aria-label=${`reference candidates on the ${cur || ""} timeline`}>
      ${here.map((c) => html`<button class=${cls("tk", chosenKeys.has(spanKey(c)) ? "ch" : c.rank != null ? "ok" : "gated")}
          style=${{ left: (100 * c.start) / len + "%", height: Math.min(60, 8 + c.seconds * 5) + "px" }}
          title=${`${c.episode} ${tc(c.start)} · ${c.seconds.toFixed(1)} s${c.rank != null ? " · rank " + c.rank : " · gated: " + c.reasons.join(", ")}`}
          aria-label=${`${tc(c.start)}, ${c.seconds.toFixed(1)} seconds`}
          onClick=${() => onPick(c)}></button>`)}
      ${axis.map((t) => html`<span class="ax" style=${{ left: (100 * t) / len + "%" }}>${Math.round(t / 60)}m</span>`)}
    </div>
  </div>`;
}

function KeyControl({ spec, value, onChange }) {
  const id = `vk-${spec.key}`;
  const set = value !== undefined && value !== null && value !== "";
  let input;
  if (spec.type === "choice") {
    input = html`<select id=${id} value=${set ? value : ""} onChange=${(e) => onChange(e.currentTarget.value || undefined)}>
      <option value="">default (${spec.default})</option>
      ${spec.choices.map((x) => html`<option value=${x}>${x}</option>`)}
    </select>`;
  } else {
    const isInt = spec.type === "int";
    const bounded = !isInt && spec.min != null && spec.max != null;
    const parse = (v) => (v === "" ? undefined : isInt ? parseInt(v, 10) : parseFloat(v));
    input = html`<span class="row">
      ${bounded && html`<input type="range" min=${spec.min} max=${spec.max} step=${(spec.max - spec.min) / 100}
          value=${set ? value : spec.default ?? spec.min} aria-label=${spec.key}
          onInput=${(e) => onChange(parse(e.currentTarget.value))} />`}
      <input id=${id} class="num" type="number" step=${isInt ? 1 : 0.01} min=${spec.min ?? undefined} max=${spec.max ?? undefined}
          placeholder=${spec.default != null ? `default ${spec.default}` : "default"} value=${set ? value : ""}
          onInput=${(e) => onChange(parse(e.currentTarget.value))} />
    </span>`;
  }
  return html`<tr><td><label for=${id}>${spec.key}</label></td><td>${input}</td>
    <td>${set && html`<button class="btn small" onClick=${() => onChange(undefined)} title="back to the default">reset</button>`}</td></tr>`;
}

export function Voice({ doc, id, lang, gpu, refreshGpu, reloadSeason }) {
  const c = doc.characters.find((x) => x.id === id) || null;
  const [language, setLanguage] = useState(lang);
  const [err, setErr] = useState(null);
  const [body, setBody] = useState(null); // {draft, meter, clips}
  const [backends, setBackends] = useState([]);
  const [cands, setCands] = useState({ list: [], total: 0, eligible: 0, err: null });
  const [showGated, setShowGated] = useState(false);
  const [lines, setLines] = useState({ list: [], total: 0, err: null });
  const [typed, setTyped] = useState([]);
  const [newTyped, setNewTyped] = useState("");
  const [sel, setSel] = useState(null); // working references
  const [backend, setBackend] = useState(null);
  const [params, setParams] = useState({});
  const [keepUnlisted, setKeepUnlisted] = useState(true);
  const [saving, setSaving] = useState({ busy: false, err: null });
  const [jobs, setJobs] = useState({}); // job id -> job
  const [cmp, setCmp] = useState({ a: null, b: null, line: null, blind: false, data: null, err: null, key: null });
  const [acc, setAcc] = useState({ version: null, preview: null, done: null, err: null, busy: false });
  const draft = body && body.draft;
  const versions = draft ? draft.versions : [];
  const latest = versions[versions.length - 1];

  // the language: the route's, else the workspace's target language (the lines route's default)
  useEffect(() => {
    if (lang) { setLanguage(lang); return; }
    call(`/api/characters/${id}/lines?limit=1`).then((b) => go("voice", id, b.language), setErr);
  }, [id, lang]);

  const loadDraft = useCallback(async () => {
    if (!language) return;
    try {
      const b = await call(`/api/drafts/${id}/${encodeURIComponent(language)}`);
      setBody(b);
      setErr(null);
      return b;
    } catch (e) {
      setErr(e);
    }
  }, [id, language]);

  useEffect(() => {
    if (!language) return;
    setBody(null); setSel(null);
    loadDraft();
    call(`/api/characters/${id}/candidates?gated=1&limit=${CAND_PAGE}`).then(
      (b) => setCands({ list: b.candidates, total: b.total, eligible: b.eligible, err: null }),
      (e) => setCands({ list: [], total: 0, eligible: 0, err: e }));
    call(`/api/characters/${id}/lines?language=${encodeURIComponent(language)}&limit=100`).then(
      (b) => setLines({ list: b.lines, total: b.total, err: null }),
      (e) => setLines({ list: [], total: 0, err: e }));
    setTyped(store.get(`typed:${id}.${language}`, []));
  }, [id, language]);

  // the backends, asked again while the server's background check hasn't answered for one
  useEffect(() => {
    let live = true, timer = null, tries = 0;
    const ask = () => call("/api/backends").then((b) => {
      if (!live) return;
      setBackends(b.backends);
      if (b.backends.some((x) => !x.checked) && ++tries < 40) timer = setTimeout(ask, 3000);
    }, () => live && setBackends([]));
    ask();
    return () => { live = false; clearTimeout(timer); };
  }, []);

  // the working selection, backend and keys start from the latest version
  useEffect(() => {
    if (!latest || sel !== null) return;
    setSel(latest.references);
    setBackend(latest.backend);
    setParams({ ...latest.params });
    setAcc((a) => ({ ...a, version: latest.n }));
  }, [latest && latest.n, latest && latest.card_id]);

  async function moreCands() {
    try {
      const b = await call(`/api/characters/${id}/candidates?gated=1&limit=${CAND_PAGE}&offset=${cands.list.length}`);
      setCands((x) => ({ ...x, list: [...x.list, ...b.candidates] }));
    } catch (e) { setCands((x) => ({ ...x, err: e })); }
  }
  async function moreLines() {
    try {
      const b = await call(`/api/characters/${id}/lines?language=${encodeURIComponent(language)}&limit=100&offset=${lines.list.length}`);
      setLines((x) => ({ ...x, list: [...x.list, ...b.lines] }));
    } catch (e) { setLines((x) => ({ ...x, err: e })); }
  }

  // -- jobs: polled while any is pending -----------------------------------------------------
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  useEffect(() => {
    const t = setInterval(async () => {
      const live = Object.values(jobsRef.current).filter((j) => PENDING.has(j.state));
      if (!live.length) return;
      let finished = false;
      for (const j of live) {
        try {
          const got = await call(`/api/auditions/${j.id}`);
          if (!PENDING.has(got.state)) finished = true;
          setJobs((x) => ({ ...x, [got.id]: got }));
        } catch (e) { /* the next poll tries again */ }
      }
      refreshGpu();
      if (finished) loadDraft();
    }, 1200);
    return () => clearInterval(t);
  }, [loadDraft]);

  async function audition(v, l) {
    const line = plainLine(l);
    try {
      const b = await post("/api/auditions", { character: id, language, version: v.n, line });
      setJobs((x) => ({ ...x, [b.job.id]: { ...b.job, error: null } }));
      refreshGpu();
    } catch (e) {
      const fake = `x${Date.now()}`;
      setJobs((x) => ({ ...x, [fake]: { id: fake, version: v.n, line, state: "failed", message: null, error: e, local: true, submitted: new Date().toISOString() } }));
    }
  }
  async function cancel(j) {
    try { const got = await post(`/api/auditions/${j.id}/cancel`); setJobs((x) => ({ ...x, [got.id]: got })); } catch (e) { /* shown by the next poll */ }
  }
  async function release() {
    try { await post("/api/gpu/release"); } finally { refreshGpu(); }
  }

  // -- the selection -------------------------------------------------------------------------
  const byKey = useMemo(() => new Map(cands.list.map((x) => [spanKey(x), x])), [cands.list]);
  const selKeys = new Set((sel || []).map(spanKey));
  const dirty = latest && sel && (
    JSON.stringify((sel || []).map(spanKey)) !== JSON.stringify(latest.references.map(spanKey)) ||
    backend !== latest.backend || JSON.stringify(clean(params)) !== JSON.stringify(clean(latest.params)));
  function clean(p) {
    return Object.fromEntries(Object.entries(p || {}).filter(([, v]) => v !== undefined && v !== null && v !== "").sort());
  }
  function pick(cand) {
    const k = spanKey(cand);
    setSel((s) => (s.some((r) => spanKey(r) === k) ? s.filter((r) => spanKey(r) !== k) : [...s, span(cand)]));
  }
  function move(i, d) {
    setSel((s) => { const x = [...s]; const j = i + d; if (j < 0 || j >= x.length) return s; [x[i], x[j]] = [x[j], x[i]]; return x; });
  }
  async function saveVersion() {
    setSaving({ busy: true, err: null });
    try {
      await post(`/api/drafts/${id}/${encodeURIComponent(language)}/versions`, {
        references: sel, backend, params: clean(params), keep_unlisted: keepUnlisted,
      });
      const b = await loadDraft();
      const v = b && b.draft.versions[b.draft.versions.length - 1];
      if (v) { setSel(v.references); setParams({ ...v.params }); setAcc((a) => ({ ...a, version: v.n, preview: null, done: null, err: null })); }
      setSaving({ busy: false, err: null });
    } catch (e) {
      setSaving({ busy: false, err: e });
    }
  }

  // -- compare -------------------------------------------------------------------------------
  const cmpRef = useRef(cmp);
  cmpRef.current = cmp;
  async function compareOnce(cfg, retry = false) {
    const q = new URLSearchParams({ a: cfg.a, b: cfg.b, line: JSON.stringify(cfg.line) });
    if (cfg.blind) q.set("blind", "1");
    if (retry) q.set("retry", "1");
    return call(`/api/drafts/${id}/${encodeURIComponent(language)}/compare?${q}`);
  }
  async function startCompare(blind, retry = false) {
    const cfg = { ...cmp, blind, data: null, err: null, key: null };
    if (!cfg.a || !cfg.b || !cfg.line) return;
    setCmp(cfg);
    try { const data = await compareOnce(cfg, retry); setCmp((x) => ({ ...x, data })); refreshGpu(); } catch (e) { setCmp((x) => ({ ...x, err: e })); }
  }
  useEffect(() => {
    const t = setInterval(async () => {
      const x = cmpRef.current;
      if (!x.data || x.err) return;
      const done = x.blind ? x.data.state !== "rendering" && x.data.state !== "missing"
        : ["a", "b"].every((s) => !PENDING.has(x.data.sides[s].state));
      if (done) return;
      try {
        const data = await compareOnce(x);
        setCmp((y) => (y.a === x.a && y.b === x.b && y.blind === x.blind ? { ...y, data } : y));
        if (!x.blind && ["a", "b"].every((s) => data.sides[s].state === "done")) loadDraft();
      } catch (e) { setCmp((y) => ({ ...y, err: e })); }
    }, 1500);
    return () => clearInterval(t);
  }, [id, language]);
  async function reveal() {
    try {
      const b = await post(`/api/drafts/${id}/${encodeURIComponent(language)}/reveal`, { a: cmp.a, b: cmp.b, line: cmp.line });
      setCmp((x) => ({ ...x, key: b.key }));
    } catch (e) { setCmp((x) => ({ ...x, err: e })); }
  }

  // -- accept --------------------------------------------------------------------------------
  const accV = versions.find((v) => v.n === acc.version) || latest;
  async function previewAccept() {
    setAcc((a) => ({ ...a, busy: true, err: null, preview: null, done: null }));
    try {
      const preview = await post(`/api/drafts/${id}/${encodeURIComponent(language)}/accept/preview`, { version: accV.n, card_id: accV.card_id });
      setAcc((a) => ({ ...a, busy: false, preview }));
    } catch (e) { setAcc((a) => ({ ...a, busy: false, err: e })); }
  }
  async function accept() {
    setAcc((a) => ({ ...a, busy: true, err: null }));
    try {
      const done = await post(`/api/drafts/${id}/${encodeURIComponent(language)}/accept`, { version: accV.n, card_id: accV.card_id });
      setAcc((a) => ({ ...a, busy: false, done, preview: null }));
      reloadSeason();
    } catch (e) { setAcc((a) => ({ ...a, busy: false, err: e })); }
  }

  // -- render --------------------------------------------------------------------------------
  if (!c) return html`<main class="screen"><div class="note bad" role="alert"><b>Character ${id} is not in the season index.</b>
    <button class="btn" onClick=${() => go("voice")}>Back to the queue</button></div></main>`;
  if (err) {
    return html`<main class="screen"><div class="hd"><h2>${charName(c)} · ${language || ""} voice</h2></div>
      <${ErrorNote} err=${err} onAction=${err.body && err.body.merged_into ? () => go("voice", err.body.merged_into, language) : loadDraft}
        actionLabel=${err.body && err.body.merged_into ? `Open character ${err.body.merged_into}` : "Retry"} /></main>`;
  }
  if (!draft || !sel) return html`<main class="screen"><p class="dim pad">Opening ${charName(c)}'s ${language || ""} draft…</p></main>`;

  const bk = backends.find((b) => b.name === backend);
  const m = meter(sel);
  const base = draft.base === "card" ? `opened from the current ${language} card`
    : draft.base === "derived" ? `started from the accepted ${draft.derived_from} card` : "the automatic proposal (assigned nowhere)";
  const allLines = [...typed.map((t) => ({ kind: "typed", text: t, emotion_from: null })),
    ...lines.list.map((l) => ({ kind: "real", episode: l.episode, cue_id: l.cue_id, _l: l }))];
  const jobFor = (v, line) => {
    const k = lineKey(line);
    return Object.values(jobs).filter((j) => j.version === v.n && lineKey(j.line) === k).sort((a, b) => (a.submitted < b.submitted ? 1 : -1))[0];
  };
  const auditionFor = (v, line) => {
    const k = lineKey(line);
    return [...v.auditions].reverse().find((a) => lineKey(a.line) === k && a.card_id === v.card_id);
  };
  const jobList = Object.values(jobs).sort((a, b) => (a.submitted < b.submitted ? 1 : -1));
  const cmpLine = cmp.line ? lineKey(cmp.line) : null;
  const shownCands = cands.list.filter((x) => showGated || x.rank != null);

  return html`<main class="screen voice-screen">
    <div class="hd">
      <h2>${charName(c)} · ${language} voice</h2>
      <span class="dim">draft ${base}${latest.card_id ? ` · v${latest.n} card ${latest.card_id}` : ""}</span>
      <span class="dim">· ${clusters(c)}</span>
      <button class="btn" onClick=${() => go("board", c.id)}>Identity</button>
    </div>
    <div class="vgrid3">
      <section class="pane cands">
        <h4>Candidates <span class="dim">· ${cands.eligible} eligible of ${cands.total}</span></h4>
        <label class="check"><input type="checkbox" checked=${showGated} onChange=${(e) => setShowGated(e.currentTarget.checked)} /> show gated</label>
        <${ErrorNote} err=${cands.err} />
        <div class="scroll">
          ${shownCands.map((x) => html`<div class=${cls("cand", x.rank == null && "gated", selKeys.has(spanKey(x)) && "chosen")}>
            <${Clip} clip=${x.clip} seconds=${x.seconds} small=${true} label=${`${x.episode} ${tc(x.start)}: ${x.transcript}`} />
            <div class="meta">
              <span class="mono small">${x.episode} ${tc(x.start)} · ${x.seconds.toFixed(1)} s${x.rank != null ? ` · #${x.rank}` : ""}${x.median ? ` · ${x.median.toFixed(0)} Hz` : ""}${x.emotion ? ` · ${x.emotion}` : ""}</span>
              <span class="ja small">${x.transcript}</span>
              ${x.reasons.length > 0 && html`<span class="small warnc">${x.reasons.map((r) => REASONS[r] || r).join("; ")}</span>`}
            </div>
            <button class="btn small" onClick=${() => pick(x)}>${selKeys.has(spanKey(x)) ? "Remove" : x.rank == null ? "Use anyway" : "Add"}</button>
          </div>`)}
          ${cands.list.length < cands.total && html`<button class="btn" onClick=${moreCands}>Load more</button>`}
        </div>
      </section>

      <div class="center">
        <section class="pane refs">
          <h4>References on the timeline <span class="dim">· orange: selected · indigo: eligible · grey: gated</span></h4>
          <${Timeline} cands=${cands.list} chosen=${sel} showGated=${showGated} onPick=${pick} />
          <div class=${cls("meter", m.floor ? "met" : "under")}>
            <span class="mono">${dirty ? "unsaved" : `v${latest.n}`}: ${m.clips} clips · ${m.seconds.toFixed(1)} s · floor ${FLOOR_CLIPS} clips / ${FLOOR_S} s ${m.floor ? "met" : "not met"} · prompt ${PROMPT_S} s${m.seconds > PROMPT_S ? " (over: only the first 15 s is heard)" : ""}</span>
            <span class="mbar"><i style=${{ width: Math.min(100, (100 * m.seconds) / (PROMPT_S * 1.4)) + "%" }}></i><b style=${{ left: (100 * FLOOR_S) / (PROMPT_S * 1.4) + "%" }}></b><b style=${{ left: (100 * PROMPT_S) / (PROMPT_S * 1.4) + "%" }}></b></span>
          </div>
          <div class="scroll">
            <table class="tight">
              <thead><tr><th>where</th><th>s</th><th>verdict</th><th>transcript</th><th></th></tr></thead>
              <tbody>${sel.map((r, i) => {
                const x = byKey.get(spanKey(r));
                const clip = (x && x.clip) || body.clips[`${r.episode} ${Number(r.start).toFixed(2)}-${Number(r.end).toFixed(2)}`];
                return html`<tr>
                  <td class="mono">${clip && html`<${Clip} clip=${clip} seconds=${r.end - r.start} small=${true} />`} ${r.episode} ${tc(r.start)}</td>
                  <td class="mono num">${(r.end - r.start).toFixed(1)}</td>
                  <td class="small">${!x ? html`<span class="dim">not a listed candidate</span>` : x.rank != null ? "own speech" : html`<span class="warnc">override: ${x.reasons.map((q) => REASONS[q] || q).join("; ")}</span>`}</td>
                  <td class="ja small">${x ? x.transcript : ""}</td>
                  <td class="nowrap"><button class="btn small" aria-label="move up" onClick=${() => move(i, -1)}>↑</button><button class="btn small" aria-label="move down" onClick=${() => move(i, 1)}>↓</button><button class="btn small" onClick=${() => pick(r)}>remove</button></td>
                </tr>`;
              })}</tbody>
            </table>
            ${latest.unlisted > 0 && html`<label class="check small"><input type="checkbox" checked=${keepUnlisted} onChange=${(e) => setKeepUnlisted(e.currentTarget.checked)} /> keep the ${latest.unlisted} reference${latest.unlisted > 1 ? "s" : ""} of v${latest.n}'s card with no episode span</label>`}
          </div>
          <div class="row">
            <button class="btn pri" disabled=${!dirty || saving.busy || !sel.length} onClick=${saveVersion}>Save as v${versions.length + 1}</button>
            ${dirty && html`<button class="btn" onClick=${() => { setSel(latest.references); setBackend(latest.backend); setParams({ ...latest.params }); }}>Revert to v${latest.n}</button>`}
            <span class="dim small">${dirty ? "references, backend or keys changed" : "make a change to save a new version"}</span>
          </div>
          <${ErrorNote} err=${saving.err} />
        </section>

        <section class="pane lines">
          <h4>Versions by line <span class="dim">· each cell an audition · tick two versions and a line to compare</span></h4>
          <div class="scroll">
            <div class="lgrid" style=${{ gridTemplateColumns: `minmax(220px, 1.6fr) repeat(${versions.length}, minmax(96px, 1fr))` }}>
              <div class="lh">line</div>
              ${versions.map((v) => html`<div class="lh">
                <label class="check"><input type="checkbox" checked=${cmp.a === v.n || cmp.b === v.n}
                  onChange=${(e) => setCmp((x) => {
                    if (e.currentTarget.checked) return x.a == null ? { ...x, a: v.n, data: null } : x.b == null ? { ...x, b: v.n, data: null } : { ...x, a: x.b, b: v.n, data: null };
                    return { ...x, a: x.a === v.n ? null : x.a, b: x.b === v.n ? null : x.b, data: null };
                  })} /> <b>v${v.n}</b></label>
                <div class="dim small">${v.backend} · ${v.tier}${Object.entries(v.params).filter(([, val]) => val != null).map(([k, val]) => ` · ${k} ${val}`).join("")}</div>
                ${!v.acceptable && html`<div class="warnc small" title=${v.accept_problems.join("; ")}>accept will refuse</div>`}
              </div>`)}
              ${allLines.map((l) => html`
                <div class=${cls("lc", cmpLine === lineKey(l) && "on")}>
                  <label class="check"><input type="radio" name="cmpline" checked=${cmpLine === lineKey(l)} onChange=${() => setCmp((x) => ({ ...x, line: plainLine(l), data: null }))} />
                  ${l.kind === "typed" ? html`<span class="dim">typed</span> ${l.text}` : html`<span class="mono small">${l.episode} ${l.cue_id}</span> ${l._l.text}`}</label>
                  ${l.kind === "real" && html`<div class="row small">${l._l.source_clip && html`<${Clip} clip=${l._l.source_clip} small=${true} label="the source line" />`}<span class="ja dim">${l._l.source_text}</span></div>`}
                  ${l.kind === "typed" && html`<button class="btn small" onClick=${() => { const t = typed.filter((x) => x !== l.text); setTyped(t); store.set(`typed:${id}.${language}`, t); }}>remove</button>`}
                </div>
                ${versions.map((v) => {
                  const a = auditionFor(v, l);
                  const j = jobFor(v, l);
                  if (a) return html`<div class="lv"><${Play} path=${`/audition/${a.wav}`} seconds=${a.seconds} label=${`v${v.n}`} />${a.suspect && html`<span class="warnc small" title="render guard">${a.suspect}</span>`}${a.gender_gate && html`<span class="small dim" title="the emotion prompt was dropped: the source line is the other register (A3)">A3</span>`}</div>`;
                  if (j && j.state === "done" && j.wav_url) return html`<div class="lv"><${Play} path=${j.wav_url} label=${`v${v.n}`} /></div>`;
                  if (j && PENDING.has(j.state)) return html`<div class="lv"><span class="state">${j.state}</span> <button class="btn small" onClick=${() => cancel(j)}>cancel</button></div>`;
                  return html`<div class="lv">${j && html`<span class="warnc small" title=${j.error ? j.error.message : j.message || ""}>${j.state}</span>`}<button class="btn small" onClick=${() => audition(v, l)}>${j ? "retry" : "render"}</button></div>`;
                })}
              `)}
            </div>
            ${lines.list.length < lines.total && html`<button class="btn" onClick=${moreLines}>More lines (${lines.total - lines.list.length})</button>`}
            ${!lines.list.length && !lines.err && html`<p class="dim small">No translated ${language} lines for this character yet: run <code class="cmd">fukikae run ${doc.workspace} &lt;episode&gt; --to translate</code> for real lines; typed lines work now.</p>`}
            <${ErrorNote} err=${lines.err} />
            <div class="row">
              <label class="sr" for="typed">typed line</label>
              <input id="typed" placeholder=${`a typed ${language} line`} value=${newTyped} maxlength="500" onInput=${(e) => setNewTyped(e.currentTarget.value)} />
              <button class="btn" disabled=${!newTyped.trim()} onClick=${() => { const t = [...typed.filter((x) => x !== newTyped.trim()), newTyped.trim()]; setTyped(t); store.set(`typed:${id}.${language}`, t); setNewTyped(""); }}>Add line</button>
            </div>
          </div>
          <div class="cmpbar">
            ${!(cmp.a && cmp.b && cmp.line) ? html`<span class="dim small">Compare: tick two versions and pick a line.</span>` : html`
              <span class="small">v${cmp.a} and v${cmp.b} on ${cmp.line.kind === "real" ? cmp.line.cue_id : "the typed line"}</span>
              <button class="btn" onClick=${() => startCompare(false)}>Side by side</button>
              <button class="btn" onClick=${() => startCompare(true)}>Blind A/B</button>`}
            ${cmp.data && !cmp.blind && html`<span class="row">${["a", "b"].map((s) => {
              const d = cmp.data.sides[s];
              return html`<span class="side"><b>v${d.version}</b> ${d.wav_url ? html`<${Play} path=${d.wav_url} label=${`v${d.version}`} />` : html`<span class="state">${d.state}</span>`}${d.message && html`<span class="warnc small">${d.message}</span>`}</span>`;
            })}${cmp.data.source_clip && html`<span class="side">source <${Clip} clip=${cmp.data.source_clip} small=${true} /></span>`}
              ${["a", "b"].some((s) => ["failed", "cancelled"].includes(cmp.data.sides[s].state)) && html`<button class="btn small" onClick=${() => startCompare(false, true)}>retry failed</button>`}</span>`}
            ${cmp.data && cmp.blind && html`<span class="row">
              ${cmp.data.state === "ready" ? ["A", "B"].map((s) => html`<span class="side"><b>${s}</b> <${Play} path=${cmp.data.sides[s].wav_url} label=${`take ${s}`} />${cmp.key && html` <span class="dim">= v${cmp.key[s]}</span>`}</span>`)
                : html`<span class="state">${cmp.data.state}</span>${cmp.data.message && html`<span class="small dim">${cmp.data.message}</span>`}`}
              ${cmp.data.source_clip && html`<span class="side">source <${Clip} clip=${cmp.data.source_clip} small=${true} /></span>`}
              ${cmp.data.state === "failed" && html`<button class="btn small" onClick=${() => startCompare(true, true)}>retry</button>`}
              ${cmp.data.state === "ready" && !cmp.key && html`<button class="btn" onClick=${reveal}>Reveal</button>`}
            </span>`}
            <${ErrorNote} err=${cmp.err} />
          </div>
        </section>
      </div>

      <div class="right">
        <section class="pane keys">
          <h4>Backend and voice keys</h4>
          <div class="row">
            <label for="backend">clone backend</label>
            <select id="backend" value=${backend || ""} onChange=${(e) => { setBackend(e.currentTarget.value); setParams({}); }}>
              ${backends.map((b) => html`<option value=${b.name}>${b.name}${b.default ? " (workspace default)" : ""}${b.checked && !b.installed ? " (can't run here)" : ""}</option>`)}
              ${!backends.some((b) => b.name === backend) && backend && html`<option value=${backend}>${backend}</option>`}
            </select>
          </div>
          ${bk && bk.checked && !bk.installed && html`<div class="note warn small">${bk.name} can't render here: ${bk.problem}</div>`}
          ${bk && !bk.checked && html`<div class="dim small">checking whether ${bk.name} can run here…</div>`}
          <div class="scroll">
            <table class="tight keys">
              <tbody>${(bk ? bk.voice_keys : []).map((spec) => html`<${KeyControl} spec=${spec} value=${params[spec.key]}
                onChange=${(v) => setParams((p) => ({ ...p, [spec.key]: v }))} />`)}</tbody>
            </table>
            ${bk && !bk.voice_keys.length && html`<p class="dim small">${bk.name} has no voice keys.</p>`}
          </div>
        </section>

        <section class="pane accept">
          <h4>Accept</h4>
          <div class="row">
            <label for="accv">version</label>
            <select id="accv" value=${accV.n} onChange=${(e) => setAcc({ version: Number(e.currentTarget.value), preview: null, done: null, err: null, busy: false })}>
              ${versions.map((v) => html`<option value=${v.n}>v${v.n} · ${v.card_id}</option>`)}
            </select>
          </div>
          ${accV.acceptable ? html`<p class="small">v${accV.n} becomes the ${language} card: studio, locked, for every episode where ${charName(c)} speaks. ${accV.auditions.length} audition${accV.auditions.length === 1 ? "" : "s"}.</p>`
            : html`<div class="note bad small"><b>Accept will refuse v${accV.n}:</b> ${accV.accept_problems.map((p) => html`<div>${p}</div>`)}</div>`}
          <div class="row">
            <button class="btn" disabled=${!accV.acceptable || acc.busy} onClick=${previewAccept}>Preview</button>
            <button class="btn pri" disabled=${!accV.acceptable || acc.busy || !acc.preview || acc.preview.stale} onClick=${accept}>Accept v${accV.n}</button>
          </div>
          ${acc.preview && html`<div class="small">
            <div class="pv">${reruns(acc.preview.invalidations)}</div>
            ${acc.preview.lines.map((l) => html`<div>${l}</div>`)}
            ${acc.preview.stale && html`<div class="note bad">Stale: ${acc.preview.stale}. A run re-picked the card since v${accV.n} was made; make a new version from the current card.</div>`}
            ${acc.preview.unchecked && html`<div class="note warn">v${accV.n} records no digest; make a new version to accept.</div>`}
            ${acc.preview.cached ? Object.entries(acc.preview.cached).map(([e, ls]) => html`<div class="dim">${e}: ${ls.length} line${ls.length === 1 ? "" : "s"} already rendered</div>`)
              : acc.preview.cached_error && html`<div class="dim">cached lines unknown: ${acc.preview.cached_error}</div>`}
          </div>`}
          ${acc.done && html`<div class="note ok small">Accepted v${accV.n}. ${reruns(acc.done.invalidations)}. ${acc.done.lines.map((l) => html`<div>${l}</div>`)}${acc.done.warning && html`<div>${acc.done.warning}</div>`}</div>`}
          <${ErrorNote} err=${acc.err} onAction=${() => { setAcc({ version: null, preview: null, done: null, err: null, busy: false }); setSel(null); loadDraft(); }} actionLabel="Reload the draft" />
        </section>

        <section class="pane queue">
          <h4>Auditions and GPU</h4>
          <div class="row"><span class="mono small">${gpuText(gpu)}</span>
            <button class="btn small" onClick=${release} title="cancel queued auditions, unload the model and let go of the GPU">Release GPU</button></div>
          ${gpu && gpu.leftover && html`<div class="note warn small">${gpu.holder}</div>`}
          <div class="scroll">
            ${jobList.length ? jobList.map((j) => html`<div class="job">
              <span class=${cls("state", j.state)}>${j.state}</span>
              <span class="small">v${j.version} · ${j.line.kind === "real" ? `${j.line.episode} ${j.line.cue_id}` : `“${j.line.text}”`}</span>
              ${j.waited_s != null && html`<span class="dim small">waited ${j.waited_s.toFixed(0)} s</span>`}
              ${PENDING.has(j.state) && !j.local && html`<button class="btn small" onClick=${() => cancel(j)}>cancel</button>`}
              ${j.message && html`<div class="small warnc">${j.message}</div>`}
              ${j.error && html`<${ErrorNote} err=${j.error} />`}
            </div>`) : html`<p class="dim small">No auditions from this page yet. Render a cell in the grid.</p>`}
          </div>
        </section>
      </div>
    </div>
  </main>`;
}
