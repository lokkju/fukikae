r"""The per-turn F0 spread guard, measured without changing the pipeline (read-only; the
2026-11-13 controller ruling, docs/results/2026-11-13-pitch-spread-guard.md).

The candidate guard: a turn whose voiced frames spread over ``LIMIT`` semitones from p10 to
p90 is classed ``u`` instead of by its median. It lives here, not in ``fukikae.pitch``: the
measurement rejected it. ``--guard ST`` sets it (``none``: today's pipeline) by wrapping
`fukikae.pitch.turn_classes` for the run, which every per-turn reader goes through (the season
index's split suggestions, the tts line gate and speaker registers, the reference gates'
majority register). Pitch tracks go to the cache under ``<cache-root>``, never the
workspace's own, and the registry is read from a copy there.

    uv run [--with librosa] python scripts/pitch-guard/measure.py turns \
        <workspace> <cache-root> <out.jsonl> [--guard ST] [--pyin ST]
    uv run python scripts/pitch-guard/measure.py gate \
        <workspace> <episode> <cache-root> <out.json> [--guard ST]

``turns``: for every ready episode and every speaking character the registry doesn't ignore
(as `studio.season.build_season` reads them), the clusters `studio.season.bimodal_clusters`
flags under the guard, and one JSON row per turn: episode, character, cluster, start,
seconds, voiced seconds, median, p10, p90, spread in semitones and its class under the guard.
``--pyin ST`` adds pYIN (librosa, 60 to 800 Hz, 2048-sample frames, as Task 15 measured) for
every turn of at least 0.4 s voiced whose spread is ``ST`` or more (needs ``--with librosa``).

``gate``: `stages.tts.gender_gate` for the episode's spoken cues (operator genders from a
scratch copy of the registry, as `fukikae eval gender` reads them), once with no guard and
once under ``--guard``: each cue's line class and speaker register both ways, and the cues
gated (line in the other band from the register) both ways. `fukikae eval gender` reports the
gated lines its plan.json recorded at render time and bands every source line on its median
itself, so it can't show this.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np

from fukikae import pitch


def spread_st(s: pitch.F0Summary) -> float | None:
    """p90 over p10 of a summary's voiced frames, in semitones; None without a median."""
    if s.p10 is None or s.p90 is None or s.p10 <= 0:
        return None
    return 12 * math.log2(s.p90 / s.p10)


_PLAIN = pitch.turn_classes


def guarded_classes(limit: float | None):
    """`pitch.turn_classes` with the guard at ``limit`` semitones (None: unchanged)."""

    def classes(track, bands, min_voiced_s=pitch.LINE_MIN_VOICED_S):
        out = _PLAIN(track, bands, min_voiced_s)
        if limit is None:
            return out
        for i, est in enumerate(out):
            sp = spread_st(pitch.summarize(track, i))
            if est.gender != "u" and sp is not None and sp > limit:
                out[i] = pitch.GenderEstimate("u", 0.0, est.median, est.voiced_s)
        return out

    return classes


@contextlib.contextmanager
def guard(limit: float | None) -> Iterator[None]:
    from fukikae.stages import tts

    fn = guarded_classes(limit)
    with mock.patch.object(pitch, "turn_classes", fn), mock.patch.object(tts, "turn_classes", fn):
        yield


def limit_of(text: str) -> float | None:
    return None if text == "none" else float(text)


def pyin_row(x: np.ndarray, rate: int) -> dict[str, float | None]:
    import librosa

    f0, flag, _ = librosa.pyin(x, fmin=60, fmax=800, sr=rate, frame_length=2048)
    v = f0[flag & ~np.isnan(f0)]
    out: dict[str, float | None] = {"pyin_voiced_s": round(len(v) * 512 / rate, 3)}
    for k, q in (("pyin_p10", 10), ("pyin_median", 50), ("pyin_p90", 90)):
        out[k] = round(float(np.percentile(v, q)), 1) if len(v) else None
    return out


def turns(a: argparse.Namespace) -> None:
    from fukikae.registry.store import Registry
    from fukikae.studio import season
    from fukikae.workspace import Workspace

    ws = Workspace(a.workspace)
    cfg = ws.config
    a.cache.mkdir(parents=True, exist_ok=True)
    reg_copy = a.cache / f"registry-{a.workspace.name}.sqlite"
    shutil.copyfile(ws.registry_path, reg_copy)
    with Registry(reg_copy) as reg:
        kinds = {c.id: c.kind for c in reg.characters()}
    bands = (pitch.MALE_BELOW, pitch.FEMALE_ABOVE)
    rows = []
    with guard(limit_of(a.guard)):
        for e in season.episode_ids(ws):
            if not season._assignments_path(ws, e).exists():
                continue
            ep = season._load(ws, cfg, e)
            if not ep.vocals.exists():
                continue
            for cid, clusters in sorted(ep.speech.items()):
                if cid not in kinds or kinds[cid] == "ignored":
                    continue
                ts = sorted(
                    (t for c in clusters.values() for t in c), key=lambda t: (t.start, t.end)
                )
                spans = [(t.start, t.end) for t in ts]
                track = pitch.cached_track(a.cache, ep.vocals, spans)
                for cl, secs in season.bimodal_clusters(track, clusters).items():
                    print(f"bimodal {e} char {cid} {cl}: m {secs['m']:.1f} u {secs['u']:.1f} "
                          f"f {secs['f']:.1f} s")  # fmt: skip
                classes = pitch.turn_classes(track, bands)
                segs = rate = None
                for i, t in enumerate(ts):
                    s = pitch.summarize(track, i)
                    sp = spread_st(s)
                    row = {
                        "episode": e, "char": cid, "cluster": t.speaker,
                        "start": round(t.start, 2), "seconds": round(t.end - t.start, 3),
                        "voiced_s": round(s.voiced_s, 3),
                        "median": None if s.median is None else round(s.median, 1),
                        "p10": None if s.p10 is None else round(s.p10, 1),
                        "p90": None if s.p90 is None else round(s.p90, 1),
                        "spread_st": None if sp is None else round(sp, 2),
                        "class": classes[i].gender,
                    }  # fmt: skip
                    if a.pyin is not None and sp is not None and sp >= a.pyin and s.voiced_s >= 0.4:
                        if segs is None:
                            segs, rate = pitch.read_spans(ep.vocals, spans)
                        row |= pyin_row(segs[i], rate)
                    rows.append(row)
    with a.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} turns -> {a.out}")


def gate(a: argparse.Namespace) -> None:
    from fukikae.artifacts import read_json
    from fukikae.eval.gender import _operator_genders
    from fukikae.models import DubCues
    from fukikae.stages import tts
    from fukikae.workspace import Workspace

    ws = Workspace(a.workspace)
    work = ws.work_dir(a.episode)
    cues = read_json(work / "06-translate" / f"cues.{ws.config.target_language}.json", DubCues)
    spoken = [c for c in cues.cues if tts.TtsStage._skipped(c) is None]
    operator = _operator_genders(a.workspace, pitch.cache_dir(a.cache) / "registry")
    with guard(None):
        plain = tts.gender_gate(work, spoken, a.cache, operator=operator)
    with guard(limit_of(a.guard)):
        guarded = tts.gender_gate(work, spoken, a.cache, operator=operator)
    rows = []
    for c in spoken:
        before, after = plain.line(c.id), guarded.line(c.id)
        rows.append({
            "cue": c.id, "start": float(c.start), "reference": list(c.delivery.reference),
            "character": c.character_id, "extra": c.extra,
            "median": None if after is None else after.median,
            "line_before": None if before is None else before.gender,
            "line_after": None if after is None else after.gender,
            "voice_before": plain.voice(c.id), "voice_after": guarded.voice(c.id),
            "gated_before": tts.wrong_gender(before, plain.voice(c.id)),
            "gated_after": tts.wrong_gender(after, guarded.voice(c.id)),
        })  # fmt: skip
    summary = {
        "episode": a.episode,
        "guard": a.guard,
        "spoken": len(rows),
        "gated_before": sum(r["gated_before"] for r in rows),
        "gated_after": sum(r["gated_after"] for r in rows),
        "line_changed": sum(r["line_before"] != r["line_after"] for r in rows),
        "voice_changed": sorted(
            {
                (r["character"], r["extra"], r["voice_before"], r["voice_after"])
                for r in rows
                if r["voice_before"] != r["voice_after"]
            },
            key=str,
        ),  # fmt: skip
        "gate_changed": [r for r in rows if r["gated_before"] != r["gated_after"]],
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    print(f"{a.episode} guard {a.guard}: {summary['spoken']} spoken cues; gated "
          f"{summary['gated_before']} -> {summary['gated_after']}; {summary['line_changed']} "
          f"line classes changed; registers changed: {summary['voice_changed']}")  # fmt: skip
    for r in summary["gate_changed"]:
        print(f"  {r['cue']} {r['start']:.1f}s char {r['character']} {r['extra']}: line "
              f"{r['line_before']}->{r['line_after']} voice {r['voice_before']}->"
              f"{r['voice_after']} ({r['median']:.0f} Hz)")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("turns")
    t.add_argument("workspace", type=Path)
    t.add_argument("cache", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--guard", default="none")
    t.add_argument("--pyin", type=float, default=None)
    g = sub.add_parser("gate")
    g.add_argument("workspace", type=Path)
    g.add_argument("episode")
    g.add_argument("cache", type=Path)
    g.add_argument("out", type=Path)
    g.add_argument("--guard", default="20")
    a = ap.parse_args()
    turns(a) if a.cmd == "turns" else gate(a)


if __name__ == "__main__":
    main()
