// The studio page: a fixed-height app over the studio's JSON API. The top bar holds the
// navigation; each screen fills the rest, and every list on it is its own scrolling pane, so
// headers, the open drawer and the decision area stay in view.
//
// Screens are a table (SCREENS): the season board (identity) and the voice screen are built;
// "Episodes" (intake) and "Script" (transcript and English lines) are planned in a later plan
// and show a placeholder, so adding them is a new entry here and a new module, no restructuring.
// Routes live in the fragment after the token is gone (token.js clears #t=): #/board,
// #/board/<character>, #/voice, #/voice/<character>[/<language>], #/episodes, #/script.

import { render } from "./vendor/preact.module.js";
import { useCallback, useEffect, useState } from "./vendor/hooks.module.js";
import { html, call, token, ErrorNote, cls, gpuText } from "./lib.js";
import { Board } from "./board.js";
import { Voice, VoicePicker } from "./voice.js";

const SCREENS = [
  { id: "episodes", label: "Episodes", planned: "Intake: each episode's source and subtitle track, and its stages run with progress." },
  { id: "board", label: "Season board" },
  { id: "script", label: "Script", planned: "Edit each episode's transcript and its English lines before the voices." },
  { id: "voice", label: "Voices" },
];

function parseRoute() {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const screen = SCREENS.some((s) => s.id === parts[0]) ? parts[0] : "board";
  const id = parts[1] && /^\d+$/.test(parts[1]) ? Number(parts[1]) : null;
  return { screen, id, lang: parts[2] || null };
}

function useRoute() {
  const [route, setRoute] = useState(parseRoute);
  useEffect(() => {
    const on = () => setRoute(parseRoute());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

// The GPU line in the top bar, polled while the page is visible.
function useGpu() {
  const [gpu, setGpu] = useState(null);
  const refresh = useCallback(() => {
    if (document.hidden) return;
    call("/api/gpu").then(setGpu, () => setGpu(null));
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);
  return [gpu, refresh];
}

function Planned({ screen }) {
  return html`<main class="screen planned">
    <div class="box">
      <h2>${screen.label}</h2>
      <p class="dim">Planned in the next plan; not built yet. ${screen.planned}</p>
    </div>
  </main>`;
}

function App() {
  const route = useRoute();
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState(null);
  const [gpu, refreshGpu] = useGpu();

  const reload = useCallback(async () => {
    try {
      setDoc(await call("/api/season"));
      setErr(null);
    } catch (e) {
      setErr(e);
    }
  }, []);
  useEffect(() => { reload(); }, []);

  // A decision's answer: the rebuilt character folded in at once, then the whole index read
  // again, since a decision touches more than its own row (a merge's survivor and every
  // character on the episodes it reruns, a split's new characters, which join the queue).
  const applied = useCallback((body) => {
    if (body && body.character && body.queue) {
      setDoc((d) => {
        if (!d) return d;
        const known = new Set(body.queue);
        const characters = d.characters
          .filter((c) => known.has(c.id))
          .map((c) => (c.id === body.character.id ? body.character : c));
        return { ...d, characters, queue: body.queue.filter((id) => characters.some((c) => c.id === id)) };
      });
    }
    reload();
  }, [reload]);

  const screen = SCREENS.find((s) => s.id === route.screen);
  const ws = doc ? doc.workspace.split("/").filter(Boolean).pop() : "";
  let body;
  if (!token) {
    body = html`<main class="screen"><div class="note bad" role="alert"><b>No studio token.</b>
      <span>Open the address fukikae studio printed (it ends in <code>#t=...</code>); the token stays in this tab only.</span></div></main>`;
  } else if (screen.planned) {
    body = html`<${Planned} screen=${screen} />`;
  } else if (!doc) {
    body = html`<main class="screen">${err ? html`<${ErrorNote} err=${err} onAction=${reload} />` : html`<p class="dim pad">Loading the season…</p>`}</main>`;
  } else if (route.screen === "voice" && route.id != null) {
    body = html`<${Voice} key=${route.id} doc=${doc} id=${route.id} lang=${route.lang} gpu=${gpu} refreshGpu=${refreshGpu} reloadSeason=${reload} />`;
  } else if (route.screen === "voice") {
    body = html`<${VoicePicker} doc=${doc} />`;
  } else {
    body = html`<${Board} doc=${doc} id=${route.id} applied=${applied} reload=${reload} />`;
  }

  return html`<div class="app">
    <header class="top">
      <span class="brand">fukikae studio</span>
      ${ws && html`<span class="ws mono" title=${doc.workspace}>${ws}</span>`}
      <nav aria-label="screens">
        ${SCREENS.map((s) => html`<a href=${"#/" + s.id}
            class=${cls("tab", s.id === route.screen && "on", s.planned && "planned")}
            aria-current=${s.id === route.screen ? "page" : null}
            title=${s.planned ? "planned: " + s.planned : null}>${s.label}${s.planned && html`<small> planned</small>`}</a>`)}
      </nav>
      <span class="gpu mono" title="the audition worker and who holds the card">${gpuText(gpu)}</span>
    </header>
    ${body}
  </div>`;
}

const root = document.getElementById("app");
root.textContent = ""; // the loading note
render(html`<${App} />`, root);
