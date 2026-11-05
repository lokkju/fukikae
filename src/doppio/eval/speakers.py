"""Speaker evaluation: export clips per diarized cluster for blind labeling, then score the
labels against the registry's assignments."""

import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from doppio.artifacts import atomic_path, read_json
from doppio.models import Assignments
from doppio.workspace import Workspace, episode_id

MAX_CLIP_SECONDS = 8.0
IGNORED_LABELS = {"", "?", "extra"}
PASS_FRACTION = 0.90
TUNE_FRACTION = 0.75
SAMPLE_RATE = 16000

_HEADER = """\
# Fill in `label` for every clip with the speaker's name, listening to the wav at `clip`
# (relative to this file). Clips are named by episode cluster, not by the system's answer.
#   - Use the same name for the same person across episodes.
#   - Use `?` when you can't tell, and `extra` for a one-off speaker who won't recur.
#   - Leave `label` empty (or `?`) to skip a clip; it is not scored.
"""


@dataclass(frozen=True)
class LabeledClip:
    episode: str
    cluster: str
    clip: str
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class SpeakerScore:
    recurring: int
    consistent: int
    merged_pairs: list[tuple[str, str]]
    fraction: float
    # The character id each merged pair shares, parallel to merged_pairs.
    merged_ids: list[int] | None = None


def _crop(span: tuple[float, float]) -> tuple[float, float]:
    start, end = span
    if end - start <= MAX_CLIP_SECONDS:
        return start, end
    mid = (start + end) / 2
    return mid - MAX_CLIP_SECONDS / 2, mid + MAX_CLIP_SECONDS / 2


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono audio")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.ascontiguousarray(pcm, dtype=np.int16).tobytes())


def labels_have_content(path: Path) -> bool:
    """True when the labels file exists and any clip already carries a label."""
    return path.is_file() and any(c.label.strip() for c in load_labels(path))


def sample(root: Path, episodes: list[str], per_cluster: int, out: Path) -> list[LabeledClip]:
    """Export the `per_cluster` longest clean spans of each cluster as clips; write labels.yaml."""
    ws = Workspace(root)
    clips: list[LabeledClip] = []
    for ref in episodes:
        eid = episode_id(Path(ref))
        work = ws.work_dir(eid)
        npz = work / "04-diarize" / "clusters.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"{eid}: no diarization at {npz}")
        pcm = _read_wav(work / "01-separate" / "vocals_16k.wav")
        with np.load(npz) as z:
            for label in [str(x) for x in z["labels"]]:
                spans = [
                    (float(a), float(b)) for a, b in np.asarray(z[f"{label}/spans"]).reshape(-1, 2)
                ]
                longest = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)[:per_cluster]
                for k, span in enumerate(sorted(longest)):
                    start, end = _crop(span)
                    rel = f"{eid}/{label}-{k}.wav"
                    lo = max(0, round(start * SAMPLE_RATE))
                    hi = min(len(pcm), lo + round((end - start) * SAMPLE_RATE))
                    _write_wav(out / rel, pcm[lo:hi])
                    clips.append(LabeledClip(eid, label, rel, start, end, ""))
    write_labels(out / "labels.yaml", clips)
    return clips


def write_labels(path: Path, clips: list[LabeledClip]) -> None:
    rows = [
        {
            "episode": c.episode,
            "cluster": c.cluster,
            "clip": c.clip,
            "start": round(c.start, 3),
            "end": round(c.end, 3),
            "label": c.label,
        }
        for c in clips
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        tmp.write_text(
            _HEADER + yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def load_labels(path: Path) -> list[LabeledClip]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of clips")
    try:
        return [
            LabeledClip(
                str(d["episode"]),
                str(d["cluster"]),
                str(d["clip"]),
                float(d["start"]),
                float(d["end"]),
                "" if d.get("label") is None else str(d["label"]).strip(),
            )
            for d in data
        ]
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path}: malformed clip entry ({e})") from e


def load_assignments(root: Path, episodes: set[str]) -> dict[str, Assignments]:
    ws = Workspace(root)
    out: dict[str, Assignments] = {}
    for eid in sorted(episodes):
        path = ws.work_dir(eid) / "05-characters" / "assignments.json"
        if not path.is_file():
            raise FileNotFoundError(f"{eid}: no assignments at {path}")
        out[eid] = read_json(path, Assignments)
    return out


def score(labels: list[LabeledClip], assignments: dict[str, Assignments]) -> SpeakerScore:
    """Score labels against assignments. A clip whose cluster maps to no character counts as
    "no character": it makes its label inconsistent and never contributes to a merge."""
    mapping = {
        (eid, c.cluster): c.character_id for eid, a in assignments.items() for c in a.clusters
    }
    episodes: dict[str, set[str]] = defaultdict(set)
    chars: dict[str, set[int | None]] = defaultdict(set)
    for clip in labels:
        if clip.label in IGNORED_LABELS:
            continue
        episodes[clip.label].add(clip.episode)
        chars[clip.label].add(mapping.get((clip.episode, clip.cluster)))
    recurring = [lab for lab, eps in episodes.items() if len(eps) >= 2]
    consistent = [lab for lab in recurring if len(chars[lab]) == 1 and None not in chars[lab]]
    by_char: dict[int, set[str]] = defaultdict(set)
    for lab, cids in chars.items():
        for cid in cids:
            if cid is not None:
                by_char[cid].add(lab)
    pairs: list[tuple[str, str]] = []
    ids: list[int] = []
    for cid in sorted(by_char):
        names = sorted(by_char[cid])
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs.append((a, b))
                ids.append(cid)
    fraction = len(consistent) / len(recurring) if recurring else 0.0
    return SpeakerScore(len(recurring), len(consistent), pairs, fraction, ids)


def verdict(s: SpeakerScore) -> str:
    if s.fraction < TUNE_FRACTION:
        return "rework"
    if s.fraction >= PASS_FRACTION and not s.merged_pairs:
        return "pass"
    return "tune"
