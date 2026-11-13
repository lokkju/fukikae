"""ZONOS3-Mini spike driver (lokkju/fukikae#43). Run with the worktree's fukikae environment:

    uv run --directory <wt> python spike.py <arm> <ws> <episode> [options]

Arms (each writes JSON lines to /nvme/fukikae/tmp/zonos3/results/<arm>-<set>.jsonl and renders
under /nvme/fukikae/tmp/zonos3/renders/<arm>/<set>/):

- speed: the first N cues of eval tts's sample, per backend config; seconds per line (the
  worker's load and the first line apart), audio seconds, worker VRAM (nvidia-smi) per line.
- pace: zonos3 with pace on (target = the cue's slot) against zonos3 and indextts2 without:
  trimmed render length over the slot.
- emotion: ~20 emotional cues: indextts2 (emotion audio), zonos3 plain, zonos3 steered by the
  cue's emotion label, zonos3 cloned from the card's references plus the line itself.
- thin: thin-tier and below-floor pipeline cards (references as they are), rendered by
  indextts2 and by zonos3 from the same references.

Every card clones a character's own speech (D41); nothing here uses a stock voice.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fukikae.backends import loader
from fukikae.cards.card import Reference
from fukikae.cards.store import CardStore
from fukikae.eval import tts as ev
from fukikae.models import DubCues
from fukikae.stages.tts import trim_silence
from fukikae.workspace import Workspace

ROOT = Path("/nvme/fukikae/tmp/zonos3")
ZONOS_HF = "/nvme/fukikae/hf-cache"

# The emotion label -> ZONOS3 steering controls (emotion_vector keys, value 0..1 scaled by the
# worker's dose, default 3). Chosen from the bundle's sets before listening: the EARS mood
# directions for the emotion, plus the delivery style the label implies.
STEER = {
    "angry": {"emotions.angry": 0.6, "delivery.loud": 0.5},
    "sad": {"emotions.sad": 0.7},
    "fearful": {"emotions.fearful": 0.7},
    "surprised": {"emotions.surprised": 0.7},
    "happy": {"emotions.happy": 0.6, "mood.valence": 0.4},
    "disgusted": {"emotions.disgusted": 0.7},
}


def gpu_mem_by_pid() -> dict[int, int]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    mem: dict[int, int] = {}
    for line in out.strip().splitlines():
        try:
            pid, mb = (int(x.strip()) for x in line.split(","))
            mem[pid] = mb
        except ValueError:
            continue
    return mem


def sample(ws: Workspace, episode: str, per_character: int = 8, seed: int = 0):
    """eval tts's own sample and its clone references per character."""
    docs = ev._cues(ws, [episode], ws.config.target_language)
    picked = ev.sample_cues(docs, per_character, seed, ev.MIN_SLOT_S)
    chars = sorted({ev._cid(c) for _, c in picked})
    cands = {cid: ev.character_candidates(ws, [episode], cid, {}) for cid in chars}
    refs = {
        cid: ev.character_references(ws, [episode], cid, candidates=f) for cid, f in cands.items()
    }
    return docs, picked, refs


def clone_cards(ws: Workspace, backend: str, refs) -> dict[int, str]:
    cls = loader.backend_class("tts", backend)
    store = CardStore(ws.root)
    src = ws.config.source_language
    return ev._clone_cards(ws, ws.config, store, cls, refs, src)


def synth_for(backend: str, options: dict[str, Any]):
    if backend == "zonos3":
        os.environ["HF_HOME"] = ZONOS_HF
    else:
        os.environ.pop("HF_HOME", None)
    cls = loader.backend_class("tts", backend)
    return cls(options, "cuda:0")


def render(
    synth,
    store: CardStore,
    card_id: str,
    text: str,
    dest: Path,
    *,
    backend: str,
    emotion_audio: Path | None = None,
    emotion_vector: dict | None = None,
    target_seconds: float | None = None,
    voice=None,
) -> tuple[float, float]:
    """(seconds after trim, wall seconds)."""
    card = store.get(card_id)
    v = voice or card.voice
    if v.backend != backend:
        rev = loader.backend_class("tts", backend).revision
        v = dataclasses.replace(v, backend=backend, model_revision=rev)
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    synth.synthesize(
        text,
        v,
        card.target_language,
        dest,
        resolve=store.sidecar_path,
        emotion_audio=emotion_audio,
        emotion_vector=emotion_vector,
        emotion_strength=v.emotion_strength,
        target_seconds=target_seconds,
    )
    wall = time.perf_counter() - t
    trim_silence(dest)
    import soundfile as sf

    info = sf.info(str(dest))
    return info.frames / info.samplerate, wall


def cut(vocals: Path, span: tuple[float, float], dest: Path) -> Path:
    from fukikae.cards.select import cut_clip

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        cut_clip(vocals, span[0], span[1], dest)
    return dest


ARMS_SPEED = [
    ("indextts2", {"tf32": True}),
    ("zonos3", {"mode": "reference", "dtype": "bf16"}),
    ("zonos3", {"mode": "reference", "dtype": "f32"}),
    ("zonos3", {"mode": "embedding", "dtype": "bf16"}),
]


def arm_speed(ws: Workspace, episode: str, set_name: str, n: int) -> None:
    docs, picked, refs = sample(ws, episode)
    store = CardStore(ws.root)
    jobs = picked[:n]
    out = ROOT / "results" / f"speed-{set_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for backend, opts in ARMS_SPEED:
        cards = clone_cards(ws, backend, refs)
        label = backend + (
            "-" + "-".join(f"{k}={v}" for k, v in sorted(opts.items())) if opts else ""
        )
        synth = synth_for(backend, opts)
        t = time.perf_counter()
        synth.load()
        load_s = time.perf_counter() - t
        peak = 0
        rows = []
        try:
            for i, (e, c) in enumerate(jobs):
                cid = ev._cid(c)
                if cid not in cards:
                    continue
                dest = ROOT / "renders" / "speed" / set_name / label / f"{e}-{c.id}.wav"
                emo = None
                if backend == "indextts2":
                    d = c.delivery.reference
                    if d[1] - d[0] >= 0.5:
                        emo = cut(
                            ws.work_dir(e) / "01-separate" / "vocals.wav",
                            d,
                            ROOT / "renders" / "speed" / set_name / "ref" / f"{e}-{c.id}.wav",
                        )
                secs, wall = render(
                    synth, store, cards[cid], c.text, dest, backend=backend, emotion_audio=emo
                )
                mem = gpu_mem_by_pid().get(synth.pid or -1, 0)
                peak = max(peak, mem)
                rows.append({"i": i, "cue": c.id, "seconds": secs, "wall": wall, "vram_mb": mem})
                print(label, i, c.id, f"{secs:.2f}s in {wall:.2f}s, {mem} MiB", flush=True)
        finally:
            synth.unload()
        steady = [r for r in rows if r["i"] > 0]
        summary = {
            "set": set_name,
            "arm": label,
            "load_s": round(load_s, 1),
            "first_line_s": round(rows[0]["wall"], 2) if rows else None,
            "lines": len(steady),
            "s_per_line": round(sum(r["wall"] for r in steady) / max(1, len(steady)), 3),
            "rtf": round(
                sum(r["wall"] for r in steady) / max(1e-9, sum(r["seconds"] for r in steady)), 3
            ),
            "audio_s": round(sum(r["seconds"] for r in steady), 1),
            "peak_vram_mb": peak,
            "rows": rows,
        }
        with out.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


def arm_pace(ws: Workspace, episode: str, set_name: str, n: int) -> None:
    docs, picked, refs = sample(ws, episode)
    store = CardStore(ws.root)
    jobs = picked[:n]
    out = ROOT / "results" / f"pace-{set_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cards = clone_cards(ws, "zonos3", refs)
    for label, opts, use_target in [
        ("zonos3-plain", {"mode": "reference"}, False),
        ("zonos3-pace", {"mode": "reference", "pace": True}, True),
    ]:
        synth = synth_for("zonos3", opts)
        synth.load()
        try:
            for e, c in jobs:
                cid = ev._cid(c)
                if cid not in cards:
                    continue
                slot = c.end - c.start
                dest = ROOT / "renders" / "pace" / set_name / label / f"{e}-{c.id}.wav"
                secs, wall = render(
                    synth,
                    store,
                    cards[cid],
                    c.text,
                    dest,
                    backend="zonos3",
                    target_seconds=slot if use_target else None,
                )
                row = {
                    "set": set_name,
                    "arm": label,
                    "cue": c.id,
                    "character": cid,
                    "slot": slot,
                    "seconds": secs,
                    "ratio": secs / slot,
                    "text": c.text,
                    "path": str(dest),
                }
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    label,
                    c.id,
                    f"slot {slot:.2f} render {secs:.2f} ratio {secs / slot:.2f}",
                    flush=True,
                )
        finally:
            synth.unload()


def emotional_cues(docs: dict[str, DubCues], picked_chars: set[int], n: int):
    """Up to n cues with a non-neutral tag, spread over labels and characters, slot >= 1 s."""
    pool = []
    for e, d in docs.items():
        for c in d.cues:
            if c.character_id is None or c.character_id not in picked_chars or not c.text:
                continue
            emo = c.delivery.emotion
            if emo not in STEER or (c.end - c.start) < 1.0:
                continue
            conf = c.delivery.emotion_confidence or 0.0
            pool.append((emo, -conf, e, c))
    pool.sort(key=lambda x: (x[0], x[1], x[3].id))
    by_label: dict[str, list] = {}
    for emo, _, e, c in pool:
        by_label.setdefault(emo, []).append((e, c))
    out = []
    while len(out) < n and any(by_label.values()):
        for emo in sorted(by_label):
            if by_label[emo] and len(out) < n:
                out.append(by_label[emo].pop(0))
    return out


def arm_emotion(ws: Workspace, episode: str, set_name: str, n: int) -> None:
    docs, picked, refs = sample(ws, episode)
    store = CardStore(ws.root)
    zcards = clone_cards(ws, "zonos3", refs)
    icards = clone_cards(ws, "indextts2", refs)
    lines = emotional_cues(docs, set(zcards), n)
    out = ROOT / "results" / f"emotion-{set_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = ROOT / "renders" / "emotion" / set_name
    vocals = {e: ws.work_dir(e) / "01-separate" / "vocals.wav" for e in docs}
    srcs = {}
    for e, c in lines:
        srcs[c.id] = cut(
            vocals[e], tuple(c.delivery.reference), base / "source" / f"{e}-{c.id}.wav"
        )
        _row(
            out,
            set_name,
            "source",
            e,
            c,
            srcs[c.id],
            c.delivery.reference[1] - c.delivery.reference[0],
        )
    # indextts2 with the source line as emotion audio (the default path)
    synth = synth_for("indextts2", {"tf32": True})
    synth.load()
    try:
        for e, c in lines:
            dest = base / "indextts2" / f"{e}-{c.id}.wav"
            secs, _ = render(
                synth,
                store,
                icards[ev._cid(c)],
                c.text,
                dest,
                backend="indextts2",
                emotion_audio=srcs[c.id],
            )
            _row(out, set_name, "indextts2", e, c, dest, secs)
    finally:
        synth.unload()
    synth = synth_for("zonos3", {"mode": "reference"})
    synth.load()
    try:
        for e, c in lines:
            card_id = zcards[ev._cid(c)]
            for label, vec, voice in [
                ("zonos3-plain", None, None),
                ("zonos3-steer", STEER[c.delivery.emotion], None),
                ("zonos3-lineref", None, _with_line(store, card_id, srcs[c.id], c)),
            ]:
                dest = base / label / f"{e}-{c.id}.wav"
                secs, _ = render(
                    synth,
                    store,
                    card_id,
                    c.text,
                    dest,
                    backend="zonos3",
                    emotion_vector=vec,
                    voice=voice,
                )
                _row(out, set_name, label, e, c, dest, secs)
    finally:
        synth.unload()


def _with_line(store: CardStore, card_id: str, line: Path, c) -> Any:
    """The card's voice with the cue's own source line first among its references (the
    character's own speech: a reference of its own for the delivery)."""
    card = store.get(card_id)
    ref = store.add_sidecar(line, "clip")
    first = Reference(
        clip=ref, transcript=c.source_text or "", language=card.voice.references[0].language
    )
    return dataclasses.replace(card.voice, references=(first, *card.voice.references))


def _row(out: Path, set_name: str, arm: str, e: str, c, dest: Path, secs: float) -> None:
    row = {
        "set": set_name,
        "arm": arm,
        "episode": e,
        "cue": c.id,
        "character": c.character_id,
        "emotion": c.delivery.emotion,
        "confidence": c.delivery.emotion_confidence,
        "text": c.text,
        "source_text": c.source_text if hasattr(c, "source_text") else None,
        "slot": c.end - c.start,
        "seconds": secs,
        "path": str(dest),
    }
    with out.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(arm, c.id, c.delivery.emotion, f"{secs:.2f}s", flush=True)


def arm_thin(
    ws: Workspace, episode: str, set_name: str, per_card: int, card_ids: list[str]
) -> None:
    """The named pipeline cards (thin tier, under the floor, the shouted-only one), each rendered
    on up to per_card of the cues the tts plan gave it, from its own references, by indextts2
    (as carded) and zonos3. Each row carries the spans of the card's cues that overlap none of
    its references, for a held-out similarity."""
    store = CardStore(ws.root)
    docs = ev._cues(ws, [episode], ws.config.target_language)
    cues = {c.id: c for c in docs[episode].cues}
    plan = json.loads((ws.work_dir(episode) / "07-tts" / "plan.json").read_text())
    by_card: dict[str, list] = {}
    for pc in plan["cues"]:
        if pc.get("card_id") in card_ids:
            by_card.setdefault(pc["card_id"], []).append(cues[pc["cue_id"]])
    out = ROOT / "results" / f"thin-{set_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = ROOT / "renders" / "thin" / set_name
    vocals = ws.work_dir(episode) / "01-separate" / "vocals.wav"
    jobs = []
    held: dict[str, list] = {}
    for card_id in card_ids:
        card = store.get(card_id)
        origins = [(r.origin or {}) for r in card.voice.references]
        mine = by_card.get(card_id, [])
        held[card_id] = [
            list(c.delivery.reference)
            for c in mine
            if (c.delivery.reference[1] - c.delivery.reference[0]) >= 0.8
            and not any(
                o.get("start", -9) < c.delivery.reference[1]
                and o.get("end", -9) > c.delivery.reference[0]
                for o in origins
            )
        ]
        for c in [c for c in mine if c.text and (c.end - c.start) >= 1.0][:per_card]:
            jobs.append((card_id, c))
    for backend, opts in [("indextts2", {"tf32": True}), ("zonos3", {"mode": "reference"})]:
        label = backend if backend == "indextts2" else "zonos3-reference"
        synth = synth_for(backend, opts)
        synth.load()
        try:
            for card_id, c in jobs:
                emo = None
                if backend == "indextts2":
                    emo = cut(
                        vocals,
                        tuple(c.delivery.reference),
                        base / "source" / f"{episode}-{c.id}.wav",
                    )
                dest = base / label / f"{episode}-{c.id}.wav"
                secs, _ = render(
                    synth, store, card_id, c.text, dest, backend=backend, emotion_audio=emo
                )
                row = {
                    "set": set_name,
                    "arm": label,
                    "episode": episode,
                    "cue": c.id,
                    "character": c.character_id if c.character_id is not None else c.extra,
                    "card": card_id,
                    "seconds": secs,
                    "slot": c.end - c.start,
                    "text": c.text,
                    "path": str(dest),
                    "heldout_spans": held[card_id],
                }
                with out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(label, card_id, c.id, f"{secs:.2f}s", flush=True)
        finally:
            synth.unload()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["speed", "pace", "emotion", "thin"])
    p.add_argument("ws", type=Path)
    p.add_argument("episode")
    p.add_argument("set_name")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--cards", default="")
    a = p.parse_args()
    ws = Workspace(a.ws)
    if a.arm == "speed":
        arm_speed(ws, a.episode, a.set_name, a.n)
    elif a.arm == "pace":
        arm_pace(ws, a.episode, a.set_name, a.n)
    elif a.arm == "emotion":
        arm_emotion(ws, a.episode, a.set_name, a.n)
    else:
        arm_thin(ws, a.episode, a.set_name, a.n, [x for x in a.cards.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
