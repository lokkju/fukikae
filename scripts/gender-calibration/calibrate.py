"""Gender band calibration (identification workflow, Task 2): the A0 estimator
(``fukikae.pitch``) against pYIN and against a voice-register truth file.

``measure`` reads one workspace's episodes and writes JSON lines, one record per ground-truth
line, per character (speech pooled over the given episodes) and per diarize cluster, each
with the A0 median and voiced seconds and, with ``--pyin``, pYIN's. ``report`` reads those
records, classes every median at the bands given, and prints the confusion tables, the share
of main-character speech in the unknown band, line agreement with pYIN, and the acceptance
verdict. Bands are applied only in ``report``, so a band can be tried without measuring again.

Lines come from the set's ground-truth subtitles (``--gt-kind ass`` or ``colour``, read with
``fukikae.eval.groundtruth``), or with ``--gt-kind none`` from the ASR transcript, which has
no names (Sol Bianca: A0 against pYIN only).

    uv run --directory $REPO --with librosa python scripts/gender-calibration/calibrate.py \\
        measure ~/fukikae-ws/kaze S01E01 S01E02 S01E03 --set kaze --gt-kind colour \\
        --gt-file ~/fukikae-ws/subs/kaze-kaoru/aligned --pyin --out kaze.jsonl
    uv run --directory $REPO python scripts/gender-calibration/calibrate.py report \\
        --truth scripts/gender-calibration/truth.toml ft.jsonl bbt.jsonl kaze.jsonl

Reads workspaces only (CPU, nothing written into them). librosa is imported only for
``--pyin`` and is never a fukikae dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from fukikae.pitch import (
    FEMALE_ABOVE,
    HOP_S,
    LINE_MIN_VOICED_S,
    MALE_BELOW,
    SPEAKER_MIN_VOICED_S,
    F0Summary,
    classify,
    f0_track,
    read_spans,
)

MAIN_S = 30.0  # a main character: this much ground-truth speech over the set's episodes
UNKNOWN_MAX = 0.10  # acceptance: main characters' seconds classed unknown
LINE_AGREEMENT_MIN = 0.90  # acceptance: per-line agreement with pYIN where both are known
CLUSTER_CAP_S = 120.0  # as cards.select.MAX_SECONDS
PYIN_FMIN, PYIN_FMAX = 60.0, 1000.0


def fold(name: str) -> str:
    return " ".join(name.split()).casefold()


# -- measuring ---------------------------------------------------------------------------------


def _pyin_frames(args: tuple[str, list[tuple[float, float]]]) -> list[np.ndarray]:
    """pYIN's voiced F0 frames per span (20 ms hop, as A0's)."""
    import librosa

    vocals, spans = args
    pieces, rate = read_spans(Path(vocals), spans)
    hop = int(HOP_S * rate)
    out = []
    for x in pieces:
        if len(x) < 2048:
            out.append(np.empty(0))
            continue
        f, voiced, _ = librosa.pyin(
            x, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=rate, frame_length=2048, hop_length=hop
        )
        out.append(f[voiced & ~np.isnan(f)])
    return out


def pyin_frames(
    vocals: Path, spans: Sequence[tuple[float, float]], workers: int
) -> list[np.ndarray]:
    chunks = [list(spans[i : i + 16]) for i in range(0, len(spans), 16)]
    with ProcessPoolExecutor(workers) as ex:
        parts = ex.map(_pyin_frames, [(str(vocals), c) for c in chunks])
    return [frames for part in parts for frames in part]


def a0_frames(vocals: Path, spans: Sequence[tuple[float, float]]) -> list[np.ndarray]:
    """A0's voiced F0 frames per span."""
    track = f0_track(vocals, spans)
    out = []
    for i in range(len(spans)):
        f = track.f0[track.span_index == i]
        out.append(f[~np.isnan(f)])
    return out


def stats(frames: np.ndarray | None) -> dict[str, float | None]:
    if frames is None:
        return {"median": None, "voiced_s": None}
    med = float(np.median(frames)) if len(frames) >= 10 else None
    return {"median": med, "voiced_s": round(len(frames) * HOP_S, 3)}


def gt_lines(ws: Path, episode: str, kind: str, gt_file: Path | None, n: int) -> list[dict]:
    """Ground-truth lines (start, end, speaker), or the ASR transcript's (speaker None)."""
    if kind == "none":
        tr = json.loads(
            (ws / ".fukikae" / "episodes" / episode / "03-asr" / "transcript.json").read_text()
        )
        return [{"start": s["start"], "end": s["end"], "speaker": None} for s in tr["segments"]]
    from fukikae.eval.autolabel import gt_path_for
    from fukikae.eval.groundtruth import read_ground_truth
    from fukikae.eval.groundtruth.base import load_alias_file
    from fukikae.eval.groundtruth.colour import workspace_colour_map

    assert gt_file is not None
    aliases = (
        load_alias_file(ws / "labels-aliases.yaml")
        if (ws / "labels-aliases.yaml").exists()
        else None
    )
    lines = read_ground_truth(
        gt_path_for(gt_file, episode, n),
        kind,  # type: ignore[arg-type]
        aliases,
        episode=episode,
        colour_names=workspace_colour_map(ws, kind, None),
    )
    return [
        {"start": g.start, "end": g.end, "speaker": g.speaker} for g in lines if g.end > g.start
    ]


def diarize_clusters(ep: Path) -> dict[str, list[tuple[float, float]]]:
    d = json.loads((ep / "04-diarize" / "turns.json").read_text())
    turns = d["turns"] if isinstance(d, dict) else d
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for t in turns:
        by[t["speaker"]].append((float(t["start"]), float(t["end"])))
    return by


def capped(
    spans: Iterable[tuple[float, float]], cap: float = CLUSTER_CAP_S
) -> list[tuple[float, float]]:
    out, total = [], 0.0
    for a, b in sorted(spans):
        if total >= cap:
            break
        b = min(b, a + cap - total)
        if b > a:
            out.append((a, b))
            total += b - a
    return out


def overlap_names(spans: Sequence[tuple[float, float]], lines: Sequence[dict]) -> dict[str, float]:
    secs: dict[str, float] = defaultdict(float)
    for a, b in spans:
        for ln in lines:
            if ln["speaker"] is None:
                continue
            o = min(b, ln["end"]) - max(a, ln["start"])
            if o > 0:
                secs[ln["speaker"]] += o
    return dict(secs)


def measure(args: argparse.Namespace) -> None:
    if args.out:
        with open(args.out, "w") as out:
            _measure(args, out)
    else:
        _measure(args, sys.stdout)


def _measure(args: argparse.Namespace, out: TextIO) -> None:
    ws = Path(args.ws).expanduser()
    gt_file = Path(args.gt_file).expanduser() if args.gt_file else None
    pooled: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    seconds: dict[str, float] = defaultdict(float)

    def emit(rec: dict) -> None:
        out.write(json.dumps({"set": args.set, **rec}, ensure_ascii=False) + "\n")
        out.flush()

    for episode in args.episodes:
        ep = ws / ".fukikae" / "episodes" / episode
        vocals = ep / "01-separate" / "vocals_16k.wav"
        lines = gt_lines(ws, episode, args.gt_kind, gt_file, len(args.episodes))
        spans = [(ln["start"], ln["end"]) for ln in lines]
        a0 = a0_frames(vocals, spans)
        py = pyin_frames(vocals, spans, args.workers) if args.pyin else [None] * len(spans)
        for ln, fa, fp in zip(lines, a0, py, strict=True):
            emit(
                {
                    "kind": "line",
                    "episode": episode,
                    "speaker": ln["speaker"],
                    "start": round(ln["start"], 3),
                    "end": round(ln["end"], 3),
                    "a0": stats(fa),
                    "pyin": stats(fp),
                }
            )
            if ln["speaker"] is not None:
                pooled[ln["speaker"]]["a0"].append(fa)
                if fp is not None:
                    pooled[ln["speaker"]]["pyin"].append(fp)
                seconds[ln["speaker"]] += ln["end"] - ln["start"]
        for cluster, cspans in sorted(diarize_clusters(ep).items()):
            cs = capped(cspans)
            names = overlap_names(cspans, lines)
            fa = np.concatenate(a0_frames(vocals, cs) or [np.empty(0)])
            fp = np.concatenate(pyin_frames(vocals, cs, args.workers)) if args.pyin else None
            emit(
                {
                    "kind": "cluster",
                    "episode": episode,
                    "cluster": cluster,
                    "seconds": round(sum(b - a for a, b in cspans), 2),
                    "names": names,
                    "a0": stats(fa),
                    "pyin": stats(fp),
                }
            )
        print(f"{args.set} {episode}: {len(lines)} lines", file=sys.stderr)
    for speaker, f in sorted(pooled.items()):
        fa = np.concatenate(f["a0"]) if f["a0"] else np.empty(0)
        fp = np.concatenate(f["pyin"]) if f["pyin"] else None
        emit(
            {
                "kind": "character",
                "speaker": speaker,
                "seconds": round(seconds[speaker], 2),
                "a0": stats(fa),
                "pyin": stats(fp),
            }
        )


# -- reporting ---------------------------------------------------------------------------------


def band(
    median: float | None, voiced_s: float | None, bands: tuple[float, float], minimum: float
) -> str:
    if median is None or voiced_s is None:
        return "u"
    s = F0Summary(median, median, median, median, voiced_s, voiced_s, 1.0)
    return classify(s, bands[0], bands[1], min_voiced_s=minimum).gender


CLASSES = ("m", "u", "f")


@dataclass
class Table:
    """One set's verdict at one pair of bands."""

    main_in_other_class: int = 0  # main characters (MAIN_S or more) in the opposite class
    unknown_share: float = 0.0  # main characters' seconds whose character median is unknown
    line_agreement: float | None = None  # A0 against pYIN, lines where both are known
    confusion: dict[str, dict[str, float]] = field(default_factory=dict)  # truth -> A0 -> seconds
    mains: list[dict] = field(default_factory=list)
    lines_compared: int = 0
    line_unknown_share: float = 0.0  # share of line seconds A0 classes unknown
    line_truth_agreement: float | None = None  # A0 line class against the speaker's truth


def accept(t: Table) -> bool:
    """The plan's rule: no main character in the other class, at most 10% of main
    characters' seconds unknown, and 90% line agreement with pYIN where both are known."""
    return (
        t.main_in_other_class == 0
        and t.unknown_share <= UNKNOWN_MAX
        and t.line_agreement is not None
        and t.line_agreement >= LINE_AGREEMENT_MIN
    )


def table(
    records: Sequence[Mapping], truth: Mapping[str, str], bands: tuple[float, float]
) -> Table:
    t = Table(confusion={c: dict.fromkeys(CLASSES, 0.0) for c in ("m", "f")})
    main_s = unknown_s = 0.0
    for r in records:
        if r["kind"] != "character" or r["seconds"] < MAIN_S:
            continue
        want = truth.get(fold(r["speaker"]))
        got = band(r["a0"]["median"], r["a0"]["voiced_s"], bands, SPEAKER_MIN_VOICED_S)
        t.mains.append(
            {
                **r,
                "truth": want,
                "a0_class": got,
                "pyin_class": band(
                    r["pyin"]["median"], r["pyin"]["voiced_s"], bands, SPEAKER_MIN_VOICED_S
                ),
            }
        )
        if want not in ("m", "f"):
            continue
        t.confusion[want][got] += r["seconds"]
        main_s += r["seconds"]
        if got == "u":
            unknown_s += r["seconds"]
        elif got != want:
            t.main_in_other_class += 1
    t.unknown_share = unknown_s / main_s if main_s else 0.0
    both = agree = 0
    line_s = line_u = 0.0
    tagree = tboth = 0
    for r in records:
        if r["kind"] != "line":
            continue
        a = band(r["a0"]["median"], r["a0"]["voiced_s"], bands, LINE_MIN_VOICED_S)
        p = band(r["pyin"]["median"], r["pyin"]["voiced_s"], bands, LINE_MIN_VOICED_S)
        dur = r["end"] - r["start"]
        line_s += dur
        line_u += dur if a == "u" else 0.0
        if a != "u" and p != "u":
            both += 1
            agree += a == p
        want = truth.get(fold(r["speaker"])) if r["speaker"] else None
        if want in ("m", "f") and a != "u":
            tboth += 1
            tagree += a == want
    t.lines_compared = both
    t.line_agreement = agree / both if both else None
    t.line_unknown_share = line_u / line_s if line_s else 0.0
    t.line_truth_agreement = tagree / tboth if tboth else None
    return t


def sweep(
    records: Sequence[Mapping], truth: Mapping[str, str], step: float = 5.0
) -> list[tuple[float, float, Table]]:
    """Every (male_below, female_above) on a ``step`` Hz grid from 120 to 320 Hz that passes
    ``accept`` on these records."""
    grid = np.arange(120.0, 320.0 + step / 2, step)
    out = []
    for lo in grid:
        for hi in grid[grid >= lo]:
            t = table(records, truth, (float(lo), float(hi)))
            if accept(t):
                out.append((float(lo), float(hi), t))
    return out


def nearest(passing: Sequence[tuple[float, float, Table]], to: tuple[float, float]):
    """The passing band nearest ``to`` (sum of the two edges' distances in Hz), or None."""
    if not passing:
        return None
    return min(passing, key=lambda p: (abs(p[0] - to[0]) + abs(p[1] - to[1]), p[0] - p[1]))


def load_truth(path: Path) -> dict[str, dict[str, str]]:
    raw = tomllib.loads(Path(path).read_text())
    out: dict[str, dict[str, str]] = {}
    for set_name, names in raw.items():
        for name, g in names.items():
            if g not in ("f", "m"):
                raise ValueError(f'{path}: [{set_name}] {name!r} = {g!r}; use "f" or "m"')
        out[set_name] = {fold(n): g for n, g in names.items()}
    return out


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _hz(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def report(args: argparse.Namespace) -> None:
    truth = load_truth(Path(args.truth))
    bands = tuple(float(x) for x in args.bands.split(","))
    assert len(bands) == 2
    by_set: dict[str, list[dict]] = defaultdict(list)
    for p in args.records:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            by_set[r["set"]].append(r)
    for set_name, recs in by_set.items():
        t = table(recs, truth.get(set_name, {}), bands)  # type: ignore[arg-type]
        print(f"\n### {set_name} at {bands[0]:g}/{bands[1]:g} Hz\n")
        if truth.get(set_name):
            print("Main characters (30 s or more of ground-truth speech), medians in Hz:\n")
            row(
                "character",
                "truth",
                "seconds",
                "A0 median",
                "A0 class",
                "pYIN median",
                "pYIN class",
            )
            row(*["---"] * 7)
            for m in sorted(t.mains, key=lambda m: m["a0"]["median"] or 0):
                row(m["speaker"], m["truth"] or "?", f"{m['seconds']:.0f}", _hz(m["a0"]["median"]),
                    m["a0_class"], _hz(m["pyin"]["median"]), m["pyin_class"])  # fmt: skip
            print("\nConfusion, main characters' seconds (truth rows, A0 columns):\n")
            row("truth", *CLASSES)
            row(*["---"] * 4)
            for c in ("m", "f"):
                row(c, *(f"{t.confusion[c][k]:.0f}" for k in CLASSES))
            print(f"\n- main characters in the other class: {t.main_in_other_class}")
            print(f"- main characters' seconds unknown: {_pct(t.unknown_share)}")
            truthful = _pct(t.line_truth_agreement)
            print(f"- A0 line class against the speaker's truth (A0 known): {truthful}")
        agreement = _pct(t.line_agreement)
        print(f"- line agreement with pYIN (both known, {t.lines_compared} lines): {agreement}")
        unknown = _pct(t.line_unknown_share)
        print(f"- line seconds A0 classes unknown (band or under 0.4 s voiced): {unknown}")
        if truth.get(set_name):
            print(f"- verdict: {'pass' if accept(t) else 'FAIL'}")
            if args.sweep:
                passing = sweep(recs, truth[set_name])
                best = nearest(passing, bands)  # type: ignore[arg-type]
                if best is None:
                    print("- sweep (5 Hz grid, 120 to 320 Hz): no band passes")
                else:
                    lo, hi, bt = best
                    print(f"- sweep: {len(passing)} bands pass; nearest {lo:g}/{hi:g} Hz "
                          f"(unknown {_pct(bt.unknown_share)}, "
                          f"line agreement {_pct(bt.line_agreement)})")  # fmt: skip
        clusters = [r for r in recs if r["kind"] == "cluster" and r["seconds"] >= MAIN_S]
        if args.clusters and clusters:
            print("\nDiarize clusters (30 s or more):\n")
            row("episode", "cluster", "seconds", "majority name (share)", "truth",
                "A0", "class", "pYIN", "class")  # fmt: skip
            row(*["---"] * 9)
            for r in clusters:
                names = r["names"]
                top = max(names, key=names.get) if names else None
                share = _pct(names[top] / sum(names.values())) if top else "-"
                want = truth.get(set_name, {}).get(fold(top)) if top else None
                a0, py = r["a0"], r["pyin"]
                row(r["episode"], r["cluster"], f"{r['seconds']:.0f}", f"{top or '-'} ({share})",
                    want or "-", _hz(a0["median"]),
                    band(a0["median"], a0["voiced_s"], bands, SPEAKER_MIN_VOICED_S),
                    _hz(py["median"]),
                    band(py["median"], py["voiced_s"], bands, SPEAKER_MIN_VOICED_S))  # fmt: skip


def row(*cells: object) -> None:
    print("| " + " | ".join(str(c) for c in cells) + " |")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="A0 (and pYIN) medians per line, character and cluster")
    m.add_argument("ws")
    m.add_argument("episodes", nargs="+")
    m.add_argument("--set", required=True, help="the set's name, as in truth.toml")
    m.add_argument("--gt-kind", choices=("ass", "colour", "none"), required=True)
    m.add_argument("--gt-file", help="ground-truth subtitle file, or a directory matched by SxxEyy")
    m.add_argument("--pyin", action="store_true", help="also pYIN (needs `uv run --with librosa`)")
    m.add_argument("--workers", type=int, default=16)
    m.add_argument("--out")
    r = sub.add_parser("report", help="tables and the acceptance verdict at the bands given")
    r.add_argument("records", nargs="+")
    r.add_argument("--truth", required=True)
    r.add_argument("--bands", default=f"{MALE_BELOW},{FEMALE_ABOVE}")
    r.add_argument("--clusters", action="store_true", help="also the per-cluster table")
    r.add_argument("--sweep", action="store_true", help="also the passing bands on a 5 Hz grid")
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        if args.gt_kind != "none" and not args.gt_file:
            ap.error("--gt-file is needed unless --gt-kind none")
        measure(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
