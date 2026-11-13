"""Score the spike's renders: WeSpeaker similarity to the card's references (and, for thin cards,
to the character's other original lines), emotion2vec label of each render (CPU), arousal
proxies (RMS level and median F0). Adds fields to each row and writes <in>.scored.jsonl.

    uv run --directory <wt> python score.py <ws> <results.jsonl> [--heldout]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig
from fukikae.eval import tts as ev
from fukikae.models import Turn
from fukikae.workspace import Workspace

TMP = Path("/nvme/fukikae/tmp/zonos3/score-16k")


def sixteen(path: Path) -> Path:
    dest = TMP / (str(path).replace("/", "_") + ".16k.wav")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        ev.to_16k(path, dest)
    return dest


def level_f0(path: Path) -> tuple[float, float | None]:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    rms = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9))
    try:
        import librosa

        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=600, sr=sr, frame_length=2048)
        f = f0[voiced] if voiced is not None else np.array([])
        med = float(np.median(f)) if f.size else None
    except Exception:  # noqa: BLE001
        med = None
    return rms, med


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ws", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument(
        "--heldout", action="store_true", help="also score against the character's other lines"
    )
    p.add_argument("--no-tag", action="store_true")
    a = p.parse_args()
    ws = Workspace(a.ws)
    store = CardStore(ws.root)
    rows = [json.loads(x) for x in a.results.read_text().splitlines() if x.strip()]
    episode = rows[0].get("episode") or next(
        iter(ws.config.__dict__.get("episodes", []) or []), None
    )
    # card references per character: the zonos3 clone card from eval's references, else the row's card
    refs: dict[int, list[Path]] = {}
    sample_cards: dict[int, str] = {}
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("spike", Path(__file__).with_name("spike.py"))
        spike = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(spike)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        spike = None
    for r in rows:
        cid = r.get("card") or r["character"]
        if cid in refs:
            continue
        card_id = r.get("card")
        if card_id is None:
            if not sample_cards and spike is not None:
                ep = r.get("episode")
                _, _, crefs = spike.sample(ws, ep)
                sample_cards.update(spike.clone_cards(ws, "zonos3", crefs))
            card_id = sample_cards.get(r["character"])
        if card_id is None:
            continue
        refs[cid] = [
            sixteen(store.sidecar_path(x.clip)) for x in store.get(card_id).voice.references
        ]
    held: dict = {}
    if a.heldout:
        for r in rows:
            key = r["card"]
            if key in held or not r.get("heldout_spans"):
                continue
            ep = r["episode"]
            v16 = ws.work_dir(ep) / "01-separate" / "vocals_16k.wav"
            if not v16.exists():
                ev.to_16k(ws.work_dir(ep) / "01-separate" / "vocals.wav", v16)
            held[key] = [(v16, tuple(sp)) for sp in r["heldout_spans"]]
    cfg = ws.config
    gpu = "cuda:0"
    with ev._model("embed", ev.EMBEDDER, gpu, cfg) as emb:
        ref_vecs = {
            cid: np.concatenate([emb.embed(pth, [(0.0, ev.wav_seconds_16k(pth))]) for pth in ps])
            for cid, ps in refs.items()
        }
        held_vecs = {
            k: np.concatenate([emb.embed(pth, [sp]) for pth, sp in items])
            for k, items in held.items()
            if items
        }
        for r in rows:
            s16 = sixteen(Path(r["path"]))
            vec = emb.embed(s16, [(0.0, ev.wav_seconds_16k(s16))])
            cid = r.get("card") or r["character"]
            if cid in ref_vecs:
                r["sim_refs"] = round(ev.similarity(vec, ref_vecs[cid]), 4)
            if r.get("card") in held_vecs:
                r["sim_heldout"] = round(ev.similarity(vec, held_vecs[r["card"]]), 4)
                r["heldout_lines"] = len(held[r["card"]])
            if r.get("source_path"):
                pass
    if not a.no_tag:
        with ev._model("tag", BackendConfig(ev.DEFAULT_TAGGER, {}), "cpu", cfg) as tg:
            for r in rows:
                s16 = sixteen(Path(r["path"]))
                secs = ev.wav_seconds_16k(s16)
                tags = tg.tag(s16, [Turn(start=0.0, end=secs, speaker="render")])
                r["render_emotion"] = ev._label(tags, 0.0)
    for r in rows:
        rms, f0 = level_f0(Path(r["path"]))
        r["rms_db"] = round(rms, 2)
        r["f0_median"] = round(f0, 1) if f0 else None
    out = a.results.with_suffix(".scored.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
