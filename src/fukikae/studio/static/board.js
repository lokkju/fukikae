// The season board: characters by episodes (bar length the seconds of speech in the episode,
// colour the register heard there), grouped by their worst flag in the review queue's order.
// Selecting a row opens the drawer beneath: samples with their video clips, the register
// against the 165/200 Hz bands, similarity to the cast, and the decision area, where each
// identity decision is a question with its answers; an answer is previewed (what it reruns)
// and applied only from the pinned decision box.

import { useEffect, useRef, useState } from "./vendor/hooks.module.js";
import {
  html, post, ErrorNote, Clip, fs, tc, GN, charName, clusters, reruns, cls, go,
  MALE_BELOW, FEMALE_ABOVE,
} from "./lib.js";

// The queue's flag order (fukikae.studio.season.SEVERITY), then the rest.
const SEVERITY = ["gender", "thin", "stock", "impure", "bimodal", "split", "identity", "gated", "names", "floor"];
const GROUP = {
  gender: "Gender conflict", thin: "Thin reference", stock: "Stock card kept", impure: "Impure references",
  bimodal: "Bimodal: split?", split: "Split skipped", identity: "New from a split", gated: "Gated lines",
  names: "Names disagreement", floor: "Below the floor", other: "Other flags", none: "No flags",
  ignored: "Not a character",
};
const GLYPH = {
  gender: "G", thin: "T", stock: "S", impure: "I", bimodal: "B", split: "P", identity: "N1",
  gated: "A3", names: "N", floor: "F", dropped: "H", stale: "X",
};
const SEV = {
  gender: "hi", thin: "hi", stock: "hi", impure: "hi", bimodal: "mid", split: "mid", identity: "mid",
  gated: "mid", names: "mid", floor: "mid", stale: "mid", dropped: "lo",
};

function worst(c) {
  if (c.kind === "ignored") return "ignored";
  const ranks = c.flags.map((f) => (SEVERITY.includes(f.kind) ? SEVERITY.indexOf(f.kind) : SEVERITY.length));
  if (!ranks.length) return "none";
  const r = Math.min(...ranks);
  return r < SEVERITY.length ? SEVERITY[r] : "other";
}

function grouped(doc) {
  const byId = new Map(doc.characters.map((c) => [c.id, c]));
  const keys = [...SEVERITY, "other", "none", "ignored"];
  const out = keys.map((k) => ({ k, items: [] }));
  for (const id of doc.queue) {
    const c = byId.get(id);
    if (c) out.find((g) => g.k === worst(c)).items.push(c);
  }
  return out.filter((g) => g.items.length);
}

function Glyph({ kind }) {
  return html`<span class=${cls("glyph", SEV[kind] || "lo")} title=${kind}>${GLYPH[kind] || "?"}</span>`;
}

// The m/u/f seconds as one bar.
function RegisterBar({ g, width = 110 }) {
  const s = g.seconds || {};
  const tot = (s.m || 0) + (s.u || 0) + (s.f || 0) || 1;
  return html`<span class="rbar" style=${{ width: width + "px" }} title=${`m ${fs(s.m || 0)} · u ${fs(s.u || 0)} · f ${fs(s.f || 0)}`}>
    <i class="m" style=${{ flex: (s.m || 0) / tot }}></i><i class="u" style=${{ flex: (s.u || 0) / tot }}></i><i class="f" style=${{ flex: (s.f || 0) / tot }}></i>
  </span>`;
}

function BoardGrid({ doc, sel, compact }) {
  const eps = doc.episodes;
  const max = Math.max(1, ...doc.characters.flatMap((c) => Object.values(c.episodes).map((v) => v.seconds)));
  const cols = `minmax(150px, 210px) repeat(${eps.length}, minmax(${eps.length > 6 ? 56 : 120}px, 1fr)) 80px 124px 120px`;
  const rowRef = useRef(null);
  useEffect(() => {
    if (rowRef.current) rowRef.current.scrollIntoView({ block: "nearest" });
  }, [sel]);
  function open(id) { go("board", id === sel ? null : id); }
  function key(e, id) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(id); }
    else if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "j" || e.key === "k") {
      e.preventDefault();
      const rows = [...e.currentTarget.parentElement.querySelectorAll(".brow.c")];
      const i = rows.indexOf(e.currentTarget) + (e.key === "ArrowDown" || e.key === "j" ? 1 : -1);
      if (rows[i]) rows[i].focus();
    } else if (e.key === "Escape") go("board");
  }
  return html`<div class=${cls("board pane", compact && "compact")} role="grid" aria-label="characters by episode">
    <div class="brow h" role="row" style=${{ gridTemplateColumns: cols }}>
      <div class="cell">character</div>
      ${eps.map((e) => html`<div class="cell" title=${e.ready ? e.id : `${e.id}: at ${e.last_stage || "nothing"}; ${e.command}`}>${e.id}${e.ready ? "" : " · not ready"}</div>`)}
      <div class="cell">total</div><div class="cell">register m/u/f</div><div class="cell">flags</div>
    </div>
    ${grouped(doc).map((g) => html`
      <div class="brow band" role="row">${GROUP[g.k]} · ${g.items.length}</div>
      ${g.items.map((c) => html`<div class=${cls("brow c", c.id === sel && "sel")} role="row" tabindex="0"
          ref=${c.id === sel ? rowRef : null}
          aria-selected=${c.id === sel ? "true" : "false"}
          onClick=${() => open(c.id)} onKeyDown=${(e) => key(e, c.id)}
          style=${{ gridTemplateColumns: cols }} title="open the drawer">
        <div class="cell who">${charName(c)}<small>${clusters(c)}${c.locked ? " · locked" : ""}${c.reviewed ? " · reviewed" : ""}</small></div>
        ${eps.map((e) => {
          const v = c.episodes[e.id];
          if (!e.ready) return html`<div class="cell"><span class="na"></span></div>`;
          return html`<div class="cell" title=${v ? `${e.id}: ${fs(v.seconds)}, ${v.turns} turns, ${v.cues} cues` : `${e.id}: silent`}>
            ${v && html`<span class=${"bar " + (c.gender.per_episode[e.id] || "u")} style=${{ width: Math.max(3, (100 * v.seconds) / max) + "%" }}></span>`}
          </div>`;
        })}
        <div class="cell mono num">${fs(c.seconds)}</div>
        <div class="cell"><${RegisterBar} g=${c.gender} /></div>
        <div class="cell">${[...new Set(c.flags.map((f) => f.kind))].map((k) => html`<${Glyph} kind=${k} />`)}</div>
      </div>`)}
    `)}
  </div>`;
}

function Legend() {
  return html`<div class="legend">
    <span>bar: seconds in the episode · colour: register there (<b class="gf">f</b> <b class="gm">m</b> <b class="gu">u</b>)</span>
    <span>${Object.entries(GLYPH).map(([k, v]) => html`<span class="lg"><span class=${cls("glyph", SEV[k] || "lo")}>${v}</span>${k}</span>`)}</span>
  </div>`;
}

function Overview({ doc }) {
  const ready = doc.episodes.filter((e) => e.ready);
  const waiting = doc.episodes.filter((e) => !e.ready);
  return html`<div class="overview">
    <section class="pane box">
      <h4>Extras by episode</h4>
      <div class="scroll">
        ${ready.map((e) => {
          const xs = doc.extras.filter((x) => x.episode === e.id);
          return html`<div class="extras">
            <b>${e.id}</b> <span class="dim">${xs.length} clusters, ${fs(xs.reduce((a, x) => a + x.seconds, 0))}; their voices are automatic, episode-local clones</span>
            ${e.video_note && html`<div class="dim small">${e.video_note}</div>`}
            ${xs.map((x) => html`<div class="xrow"><span class="mono">${x.label}</span><span class="dim mono">${x.tier || "no tts yet"}</span>
              ${x.samples.map((s) => html`<${Clip} video=${s.video} clip=${s.clip} seconds=${s.end - s.start} small=${true} label=${`${x.label} ${tc(s.start)}: ${s.transcript}`} />`)}
            </div>`)}
          </div>`;
        })}
      </div>
    </section>
    <section class="pane box">
      <h4>Not in the studio yet</h4>
      <div class="scroll">
        ${waiting.length ? waiting.map((e) => html`<p><b class="mono">${e.id}</b> <span class="dim">at ${e.last_stage || "nothing"}</span><br /><code class="cmd">${e.command}</code></p>`)
          : html`<p class="dim">Every episode has run through characters.</p>`}
        ${doc.episodes.filter((e) => e.note).map((e) => html`<p class="dim">${e.id}: ${e.note}</p>`)}
      </div>
    </section>
  </div>`;
}

// The register chart: the median against the bands.
function RegisterChart({ g }) {
  const W = 240, lo = 80, hi = 450;
  const x = (hz) => Math.max(0, Math.min(W, ((hz - lo) / (hi - lo)) * W));
  return html`<svg class="rchart" viewBox="0 0 240 62" role="img"
      aria-label=${`median ${g.median == null ? "unknown" : g.median.toFixed(0) + " Hz"}; male below ${MALE_BELOW} Hz, female above ${FEMALE_ABOVE} Hz`}>
    <rect x="0" y="22" width=${W} height="14" class="track" />
    <rect x="0" y="22" width=${x(MALE_BELOW)} height="14" class="bm" />
    <rect x=${x(FEMALE_ABOVE)} y="22" width=${W - x(FEMALE_ABOVE)} height="14" class="bf" />
    ${g.median != null && html`<line x1=${x(g.median)} x2=${x(g.median)} y1="14" y2="44" class="med" />
      <text x=${x(g.median)} y="10" text-anchor="middle" class="lab">${g.median.toFixed(0)} Hz</text>`}
    <text x=${x(MALE_BELOW) - 2} y="56" text-anchor="end" class="tick">${MALE_BELOW}</text>
    <text x=${x(FEMALE_ABOVE) + 2} y="56" class="tick">${FEMALE_ABOVE}</text>
  </svg>`;
}

function Samples({ c }) {
  return html`<section class="pane">
    <h4>Samples across the season <span class="dim">· ${c.samples.length}</span></h4>
    <div class="scroll samples">
      ${c.samples.length ? c.samples.map((s) => html`<div class="sample">
        <${Clip} video=${s.video} clip=${s.clip} seconds=${s.end - s.start} preload=${true} label=${`${s.episode} ${tc(s.start)}: ${s.transcript}`} />
        <div class="meta"><span class="mono">${s.episode} ${tc(s.start)}</span><span class="ja">${s.transcript || html`<i class="dim">no transcript</i>`}</span></div>
      </div>`) : html`<p class="dim">No samples: no clean turn, or the episode has no vocals stem.</p>`}
    </div>
  </section>`;
}

function Evidence({ c, doc }) {
  const g = c.gender;
  return html`<section class="pane">
    <h4>Register and evidence</h4>
    <div class="scroll">
      <p>Stored <b>${g.stored ? GN[g.stored] : "not set"}</b>${g.stored_by ? html` <span class="dim">(${g.stored_by})</span>` : ""} · estimated <b class=${"g" + g.estimate}>${GN[g.estimate]}</b> <span class="mono">${g.confidence.toFixed(2)}</span></p>
      <${RegisterChart} g=${g} />
      <div><${RegisterBar} g=${g} width=${200} /></div>
      <div class="mono dim small">m ${fs(g.seconds.m || 0)} · u ${fs(g.seconds.u || 0)} · f ${fs(g.seconds.f || 0)}</div>
      <h5>Per episode</h5>
      <table class="tight">
        <thead><tr><th>episode</th><th>clusters</th><th>speech</th><th>clean</th><th>cues</th><th>reg.</th></tr></thead>
        <tbody>${Object.entries(c.episodes).map(([e, v]) => html`<tr><td class="mono">${e}</td><td class="mono">${v.clusters.join(", ")}</td><td class="mono num">${fs(v.seconds)}</td><td class="mono num">${fs(v.clean_seconds)}</td><td class="mono num">${v.cues}</td><td>${g.per_episode[e] || ""}</td></tr>`)}</tbody>
      </table>
      <h5>Voice cards</h5>
      ${c.cards.length ? c.cards.map((k) => html`<div class="card">
        <span class="mono">${k.language}</span> <span class="mono">${k.card_id || "none"}</span>
        <span class="dim">${[k.source, k.chosen_by, k.locked ? "locked" : null, k.tier].filter(Boolean).join(" · ")}</span>
        ${k.flags.map((f) => html`<${Glyph} kind=${f} />`)}
        <button class="btn" onClick=${() => go("voice", c.id, k.language)}>Tune ${k.language} voice</button>
      </div>`) : html`<p class="dim">No card yet.</p>`}
      <button class="btn" onClick=${() => go("voice", c.id)}>Open the voice screen</button>
      <h5>Flags</h5>
      ${c.flags.length ? c.flags.map((f) => html`<div class="flag"><${Glyph} kind=${f.kind} /><span>${f.text}</span></div>`) : html`<p class="dim">None.</p>`}
      ${c.names.length > 0 && html`<details><summary>Names evidence (${c.names.length})</summary>${c.names.map((n) => html`<div class="mono small">${n}</div>`)}</details>`}
      ${doc.cast_error && html`<p class="note warn small">cast.toml: ${doc.cast_error}</p>`}
    </div>
  </section>`;
}

function Similarity({ c, doc }) {
  const merges = c.suggestions.filter((s) => s.kind === "merge" && s.cosine != null);
  const hints = c.suggestions.filter((s) => s.kind === "merge" && s.cosine == null);
  const splits = c.suggestions.filter((s) => s.kind === "split");
  const byId = new Map(doc.characters.map((x) => [x.id, x]));
  return html`<section class="pane">
    <h4>Similarity to the cast</h4>
    <div class="scroll">
      ${merges.length ? html`<div class="sim">${merges.map((s) => html`<div class="simrow">
        <span class="mono">${charName(byId.get(s.other))}</span><span class="simbar"><i style=${{ width: s.cosine * 100 + "%" }}></i></span><span class="mono num">${s.cosine.toFixed(2)}</span>
      </div>`)}</div>` : html`<p class="dim">No other character at or over 0.50 cosine.</p>`}
      ${hints.map((s) => html`<p class="small">Subtitle names in ${s.episode} suggest the same person as <b>${charName(byId.get(s.other))}</b>.</p>`)}
      <h5>Split by pitch</h5>
      ${splits.length ? splits.map((s) => {
        const p = s.parts || {};
        const sec = (k) => (p[k] && p[k].seconds) || 0;
        return html`<div class="split">
          <div class="mono">${s.episode} ${s.cluster}</div>
          <span class="rbar wide"><i class="m" style=${{ flex: sec("m") }}></i><i class="u" style=${{ flex: sec("u") }}></i><i class="f" style=${{ flex: sec("f") }}></i></span>
          ${["m", "f"].map((k) => html`<div class="part"><span class="mono small">~voice-${k} · ${fs(sec(k))}</span>
            ${((p[k] && p[k].samples) || []).map((x) => html`<${Clip} video=${x.video} clip=${x.clip} seconds=${x.end - x.start} small=${true} label=${`~voice-${k} ${tc(x.start)}: ${x.transcript}`} />`)}</div>`)}
        </div>`;
      }) : html`<p class="dim">No cluster of this character is bimodal.</p>`}
    </div>
  </section>`;
}

// -- decisions -------------------------------------------------------------------------------

function Answer({ label, sub, rec, on, onClick, disabled }) {
  return html`<button class=${cls("opt", rec && "rec", on && "on")} onClick=${onClick} disabled=${disabled} aria-pressed=${on ? "true" : null}>
    ${label}${rec && html`<span class="tag">suggested</span>`}${sub && html`<small>${sub}</small>`}
  </button>`;
}

function Decisions({ c, doc, applied, reload }) {
  const [pending, setPending] = useState(null); // {label, sent, preview, err, busy, done}
  const [name, setName] = useState("");
  const [alias, setAlias] = useState("");
  const [other, setOther] = useState("");
  const [keepThis, setKeepThis] = useState(true);
  useEffect(() => { setPending(null); setName(""); setAlias(""); setOther(""); }, [c.id]);

  const base = { character: c.id, digest: c.digest, key: c.key, external_id: c.external_id };
  async function propose(label, decision) {
    setPending({ label, sent: decision, busy: true });
    try {
      const preview = await post("/api/decisions/preview", decision);
      setPending({ label, sent: decision, preview });
    } catch (err) {
      setPending({ label, sent: decision, err });
      if (err.status === 409 && err.code === "stale") reload();
    }
  }
  async function apply() {
    const p = pending;
    setPending({ ...p, busy: true, err: null });
    try {
      const body = await post("/api/decisions", p.preview.decision);
      setPending({ ...p, busy: false, done: body });
      applied(body);
      if (p.sent.op === "merge" && p.sent.drop === c.id) go("board", p.sent.keep);
    } catch (err) {
      setPending({ ...p, busy: false, err });
      if (err.status === 409) reload();
    }
  }

  const g = c.gender;
  const splits = c.suggestions.filter((s) => s.kind === "split");
  const merges = c.suggestions.filter((s) => s.kind === "merge");
  const others = doc.characters.filter((x) => x.id !== c.id);
  const byId = new Map(doc.characters.map((x) => [x.id, x]));
  const castEntry = c.cast ? doc.cast[c.cast.entry] : null;
  // a merge carries both characters' rows as the index showed them (each its own digest), so
  // a change to either since the page loaded makes it stale, naming which
  const merge = (otherId, keepSelf) => {
    const keep = keepSelf ? c : byId.get(otherId);
    const drop = keepSelf ? byId.get(otherId) : c;
    return {
      op: "merge", keep: keep.id, drop: drop.id, key: keep.key, external_id: keep.external_id,
      digests: { keep: keep.digest, drop: drop.digest },
    };
  };
  const p = pending;

  return html`<section class="pane decide">
    <h4>Decisions</h4>
    <div class="scroll asks">
      ${splits.length > 0 && html`<div class="ask">
        <h5>Is this one voice?</h5>
        ${splits.map((s) => html`<p class="small">${s.episode} ${s.cluster}: ${fs((s.parts.m || {}).seconds || 0)} low, ${fs((s.parts.f || {}).seconds || 0)} high.</p>
          <div class="opts">
            <${Answer} label="One voice" sub="mark reviewed" onClick=${() => propose("one voice: reviewed", { op: "reviewed", ...base })} />
            <${Answer} label=${`Two voices: split ${s.cluster} by pitch`} rec=${true}
              onClick=${() => propose(`split ${s.episode} ${s.cluster} by pitch`, { op: "split", ...base, episode: s.episode, cluster: s.cluster, by: "pitch", male_below: MALE_BELOW, female_above: FEMALE_ABOVE })} />
          </div>`)}
      </div>`}
      <div class="ask">
        <h5>Which register should the voice keep?</h5>
        <p class="small">Estimated ${GN[g.estimate]} (${g.confidence.toFixed(2)}${g.median != null ? `, median ${g.median.toFixed(0)} Hz` : ""}); stored ${g.stored ? GN[g.stored] : "not set"}.</p>
        <div class="opts">
          <${Answer} label="Female" on=${g.stored === "f"} rec=${!g.stored && g.estimate === "f"} onClick=${() => propose("voice: female", { op: "gender", ...base, gender: "f" })} />
          <${Answer} label="Male" on=${g.stored === "m"} rec=${!g.stored && g.estimate === "m"} onClick=${() => propose("voice: male", { op: "gender", ...base, gender: "m" })} />
          ${g.stored && html`<${Answer} label="Clear" sub="back to the estimate" onClick=${() => propose("voice: clear", { op: "gender", ...base, gender: null })} />`}
          ${c.cast && c.cast.gender && html`<${Answer} label=${`Cast says ${GN[c.cast.gender]}`} sub=${`cast.toml: ${castEntry ? castEntry.name : ""}`}
            onClick=${() => propose(`voice: ${GN[c.cast.gender]} (cast)`, { op: "gender", ...base, gender: c.cast.gender })} />`}
        </div>
      </div>
      <div class="ask">
        <h5>Same person as someone else?</h5>
        ${merges.length ? html`<div class="opts">${merges.map((s) => html`<${Answer}
            label=${`Same as ${charName(byId.get(s.other))}`} sub=${s.cosine != null ? `cosine ${s.cosine.toFixed(2)}` : `names in ${s.episode}`}
            onClick=${() => propose(`merge with ${charName(byId.get(s.other))}`, merge(s.other, keepThis))} />`)}
            <${Answer} label="Different people" sub="mark reviewed" onClick=${() => propose("different people: reviewed", { op: "reviewed", ...base })} />
          </div>` : html`<p class="small dim">No suggestion. Any character:</p>`}
        <div class="row">
          <label class="sr" for="merge-other">merge with</label>
          <select id="merge-other" value=${other} onChange=${(e) => setOther(e.currentTarget.value)}>
            <option value="">merge with…</option>
            ${others.map((x) => html`<option value=${x.id}>${charName(x)} (${clusters(x)})</option>`)}
          </select>
          <label class="check"><input type="checkbox" checked=${keepThis} onChange=${(e) => setKeepThis(e.currentTarget.checked)} /> keep this one</label>
          <button class="btn" disabled=${!other} onClick=${() => propose(`merge with ${charName(byId.get(Number(other)))}`, merge(Number(other), keepThis))}>Merge</button>
        </div>
      </div>
      <div class="ask">
        <h5>What is it?</h5>
        <div class="opts">
          <${Answer} label="A character" on=${c.kind === "character"} onClick=${() => propose("kind: character", { op: "kind", ...base, kind: "character" })} />
          <${Answer} label="A narrator" on=${c.kind === "narrator"} sub="its own card; never paired with dialogue" onClick=${() => propose("kind: narrator", { op: "kind", ...base, kind: "narrator" })} />
          <${Answer} label="Not a character" on=${c.kind === "ignored"} sub="never voiced; keeps the original audio" onClick=${() => propose("kind: not a character", { op: "kind", ...base, kind: "ignored" })} />
        </div>
      </div>
      <div class="ask">
        <h5>Which clusters are this character?</h5>
        ${Object.entries(c.episodes).flatMap(([e, v]) => v.clusters.map((cl) => html`<div class="row">
          <span class="mono">${e} ${cl}</span>
          <button class="btn" onClick=${() => propose(`split ${e} ${cl} out`, { op: "split", ...base, episode: e, cluster: cl, by: "cluster" })}>Not this character: split out</button>
        </div>`))}
      </div>
      <div class="ask">
        <h5>Name</h5>
        ${castEntry && castEntry.name !== c.name && html`<div class="opts"><${Answer} label=${`Cast: ${castEntry.name}`} sub=${`matched by ${c.cast.by}${c.cast.ambiguous_with.length ? `; also matches ${c.cast.ambiguous_with.map((i) => "char " + i).join(", ")}` : ""}`}
          onClick=${() => propose(`name: ${castEntry.name}`, { op: "name", ...base, name: castEntry.name })} /></div>`}
        <div class="row">
          <label class="sr" for="c-name">name</label>
          <input id="c-name" list="cast-names" placeholder=${c.name || "name"} value=${name} onInput=${(e) => setName(e.currentTarget.value)} />
          <button class="btn" disabled=${!name.trim()} onClick=${() => propose(`name: ${name.trim()}`, { op: "name", ...base, name: name.trim() })}>Rename</button>
        </div>
        <datalist id="cast-names">${(doc.cast || []).map((x) => html`<option value=${x.name} />`)}</datalist>
        <div class="row">
          <label class="sr" for="c-alias">alias</label>
          <input id="c-alias" placeholder="alias (a subtitle name)" value=${alias} onInput=${(e) => setAlias(e.currentTarget.value)} />
          <button class="btn" disabled=${!alias.trim()} onClick=${() => propose(`alias: ${alias.trim()}`, { op: "alias", ...base, name: alias.trim() })}>Add alias</button>
        </div>
      </div>
      <div class="ask">
        <h5>Done with this one?</h5>
        <div class="opts">
          <${Answer} label="Looks right" sub=${c.reviewed ? `reviewed ${c.reviewed.slice(0, 10)}` : "mark reviewed"} onClick=${() => propose("reviewed, no change", { op: "reviewed", ...base })} />
          <${Answer} label=${c.locked ? "Locked" : "Lock identity"} on=${c.locked} disabled=${c.locked} sub="no automatic rematch moves it" onClick=${() => propose("lock", { op: "lock", ...base })} />
        </div>
      </div>
    </div>
    <div class=${cls("pending", p && "open")} aria-live="polite">
      ${!p && html`<p class="dim small">Pick an answer above; it is previewed here before anything is written.</p>`}
      ${p && html`<div>
        <b>${p.label}</b>
        ${p.busy && html`<span class="dim"> · working…</span>`}
        ${p.preview && !p.done && html`<div class="pv">${reruns(p.preview.invalidations)}</div>
          ${p.preview.lines.map((l) => html`<div class="small">${l}</div>`)}
          ${p.preview.stale && html`<div class="note warn small">Stale: ${p.preview.stale}. Applying would be refused; reload and decide again.</div>
            <button class="btn" onClick=${() => { reload(); setPending(null); }}>Reload and dismiss</button>`}
          ${p.preview.unchecked && html`<div class="note warn small">This decision carried no digest; the preview filled it from the registry now.</div>`}
          <div class="row"><button class="btn pri" onClick=${apply} disabled=${p.busy || Boolean(p.preview.stale)}>Apply</button><button class="btn" onClick=${() => setPending(null)}>Cancel</button></div>`}
        ${p.done && html`<div class="note ok small">Applied. ${reruns(p.done.invalidations)}.
          ${p.done.lines.map((l) => html`<div>${l}</div>`)}
          ${p.done.warning && html`<div>${p.done.warning}</div>`}</div>
          <button class="btn" onClick=${() => setPending(null)}>Done</button>`}
        ${p.err && html`<${ErrorNote} err=${p.err} onAction=${() => { reload(); setPending(null); }} actionLabel="Reload and dismiss" />`}
      </div>`}
    </div>
  </section>`;
}

function Drawer({ c, doc, applied, reload }) {
  return html`<div class="drawer" aria-label=${`${charName(c)}: evidence and decisions`}>
    <${Samples} c=${c} />
    <${Evidence} c=${c} doc=${doc} />
    <${Similarity} c=${c} doc=${doc} />
    <${Decisions} c=${c} doc=${doc} applied=${applied} reload=${reload} />
  </div>`;
}

export function Board({ doc, id, applied, reload }) {
  useEffect(() => {
    const on = (e) => {
      if (e.key !== "Escape" || id == null) return;
      if (e.target.closest && e.target.closest("input, select, textarea")) return;
      go("board");
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [id]);
  const c = id != null ? doc.characters.find((x) => x.id === id) : null;
  const flagged = doc.characters.filter((x) => x.flags.length).length;
  const unreviewed = doc.characters.filter((x) => !x.reviewed).length;
  const ready = doc.episodes.filter((e) => e.ready).length;
  const speaking = doc.characters.filter((x) => x.kind !== "ignored").length;
  return html`<main class=${cls("screen board-screen", c && "with-drawer")}>
    <div class="hd">
      ${c ? html`<h2>${charName(c)}</h2><span class="mono dim">${clusters(c)} · actor ${c.actor_id} · ${c.kind} · ${c.locked ? "locked" : "unlocked"} · ${c.reviewed ? "reviewed" : "unreviewed"} · ${fs(c.seconds)} over ${Object.keys(c.episodes).length} ep.</span>
            <button class="btn" onClick=${() => go("board")} title="close the drawer (Esc)">Close</button>`
        : html`<h2>Season board</h2>`}
      <span class="dim">${doc.characters.length} characters · ${flagged} flagged · ${unreviewed} unreviewed · ${ready}/${doc.episodes.length} episodes ready</span>
      ${doc.cast && doc.cast.length > 0 && html`<span class="dim">· cast lists ${doc.cast.length} speaking roles; this season has ${speaking} characters</span>`}
      ${id != null && !c && html`<span class="note warn small">Character ${id} is not in the index (merged away?).</span>`}
    </div>
    <${BoardGrid} doc=${doc} sel=${c ? c.id : null} compact=${Boolean(c)} />
    <${Legend} />
    ${c ? html`<${Drawer} c=${c} doc=${doc} applied=${applied} reload=${reload} />` : html`<${Overview} doc=${doc} />`}
  </main>`;
}
