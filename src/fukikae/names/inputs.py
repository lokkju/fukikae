"""Which turns and clusters identity reads: the names stage's refined pair when its
`names.json` says names were used or a pitch split relabelled turns, else the diarize stage's
(Task 4's helper for the characters stage, `registry resolve`, `reembed`, `adapt` and
`eval identity`); and the operator pitch splits the names stage reads
(``<ws>/studio/splits/<episode>.json``, written by `fukikae studio apply`)."""

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fukikae.artifacts import read_json
from fukikae.names.models import NamesResult

if TYPE_CHECKING:
    from fukikae.models import Turn

NAMES_DIR = "04c-names"
DIARIZE_DIR = "04-diarize"


def names_path(work: Path) -> Path:
    return Path(work) / NAMES_DIR / "names.json"


def names_report_path(work: Path) -> Path:
    """The characters stage's `NamesReport` for a names-used episode."""
    return Path(work) / "05-characters" / "names-report.json"


def identity_inputs(work: Path) -> tuple[Path, Path, NamesResult | None]:
    """(turns.json, clusters.npz, names result) for an episode's work directory: the refined
    pair under `04c-names/` when names were used or a pitch split relabelled turns
    (`NamesResult.refined`), else `04-diarize/`'s, with the result (None when the names stage
    never ran)."""
    path = names_path(work)
    result = read_json(path, NamesResult) if path.exists() else None
    d = Path(work) / (NAMES_DIR if result is not None and result.refined else DIARIZE_DIR)
    return d / "turns.json", d / "clusters.npz", result


# ---- operator pitch splits ----


def splits_path(workspace_root: Path, episode: str) -> Path:
    """``<ws>/studio/splits/<episode>.json``: ``{cluster: {"by": "pitch", "male_below": Hz,
    "female_above": Hz, "fingerprint": {...}}}``, the fingerprint (`cluster_fingerprint`) of
    the diarize cluster the decision split; files written before it existed have none."""
    return Path(workspace_root) / "studio" / "splits" / f"{episode}.json"


FINGERPRINT_KEYS = frozenset({"turns", "seconds", "sha256"})


def cluster_fingerprint(turns: "Iterable[Turn]", cluster: str) -> dict[str, Any] | None:
    """What a pitch split was decided on: the cluster's turn count, seconds and the sha256 of
    its turn spans (start order, millisecond precision) in a diarization; None when the
    cluster has no turn there. A diarize rerun that relabels or reshapes the cluster changes
    it, so an old split never lands on a different cluster."""
    spans = sorted((round(t.start, 3), round(t.end, 3)) for t in turns if t.speaker == cluster)
    if not spans:
        return None
    digest = hashlib.sha256(json.dumps([list(sp) for sp in spans]).encode()).hexdigest()
    seconds = round(sum(e - s for s, e in spans), 3)
    return {"turns": len(spans), "seconds": seconds, "sha256": digest}


def describe_fingerprint(fp: dict[str, Any] | None) -> str:
    if fp is None:
        return "no turns"
    return f"{fp['turns']} turns, {fp['seconds']:g} s"


@dataclass(frozen=True)
class PitchSplit:
    male_below: float
    female_above: float
    fingerprint: dict[str, Any] | None = None  # None: a split file from before fingerprints


class SplitFileError(ValueError):
    """An operator split file the names stage can't use, or a split it can't measure."""


def splits_digest(workspace_root: Path, episode: str) -> str | None:
    """sha256 of the episode's split file, or None when it has none."""
    path = splits_path(workspace_root, episode)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def operator_splits(workspace_root: Path, episode: str) -> dict[str, PitchSplit]:
    """The episode's pitch splits by cluster; empty when it has no split file. SplitFileError
    naming the file for anything else than an object of clusters, each ``by = "pitch"`` with
    finite positive bands, ``male_below`` at most ``female_above``."""
    path = splits_path(workspace_root, episode)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise SplitFileError(f"{path}: not a split file ({e})") from e
    if not isinstance(raw, dict):
        raise SplitFileError(f"{path}: expected an object of clusters, not {type(raw).__name__}")
    out: dict[str, PitchSplit] = {}
    for cluster, body in sorted(raw.items()):
        if not isinstance(body, dict) or body.get("by") != "pitch":
            raise SplitFileError(f'{path}: {cluster} is not a split with by = "pitch"')
        bands = []
        for k in ("male_below", "female_above"):
            v = body.get(k)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                raise SplitFileError(f"{path}: {cluster} {k} must be a number, not {v!r}")
            if v <= 0:
                raise SplitFileError(f"{path}: {cluster} {k} must be above 0 Hz, not {v!r}")
            bands.append(float(v))
        if bands[0] > bands[1]:
            raise SplitFileError(
                f"{path}: {cluster} male_below {bands[0]:g} Hz is above female_above "
                f"{bands[1]:g} Hz"
            )
        fp = body.get("fingerprint")
        if fp is not None and not _fingerprint_ok(fp):
            raise SplitFileError(
                f"{path}: {cluster} fingerprint must be an object of turns, seconds and "
                f"sha256, not {fp!r}"
            )
        out[cluster] = PitchSplit(*bands, fingerprint=fp)
    return out


def _fingerprint_ok(fp: Any) -> bool:
    if not isinstance(fp, dict) or set(fp) != FINGERPRINT_KEYS:
        return False
    n, secs, digest = fp["turns"], fp["seconds"], fp["sha256"]
    return (
        isinstance(n, int)
        and not isinstance(n, bool)
        and n > 0
        and isinstance(secs, int | float)
        and not isinstance(secs, bool)
        and math.isfinite(secs)
        and isinstance(digest, str)
        and len(digest) == 64
    )


# What a held-out episode's names.json may say: it read no names (`holdout`), or names were
# off for the whole run (a baseline arm). A missing file is a workspace from before the stage.
HELD_OUT_VERDICTS = frozenset({"holdout", "off"})


class HeldOutNamesError(ValueError):
    """A held-out episode whose names stage read its subtitle names."""


def check_held_out(work: Path, episode: str) -> None:
    """Refuse to grade `episode` as held out when it read subtitle names (Review Focus 3).
    Every eval path that scores a held-out episode calls this first. Refused when:

    - its names stage read names (`names.json` verdict other than `holdout` or `off`);
    - its `05-characters/assignments.json` holds a name decision (a cluster with a key, or
      one whose source is a name), since the characters run behind it read names whatever
      `names.json` says now;
    - those assignments record other turns than `04-diarize/turns.json` (the names stage's
      refined turns: subtitle names, or an operator pitch split's voice parts, which the
      message names as such; either way the grade would not be the diarizer's clusters).

    A missing `names.json` passes the first test, because it means one of two things this
    guard can't tell apart: the names stage never ran (a workspace from before it), so
    identity read `04-diarize`'s inputs and no names; or its last run failed, since a run
    removes the old file first. The assignments tests catch the second case when the
    assignments came from a run that read names. Missing assignments, or assignments from
    before they recorded turns, pass those tests (the eval reports a missing file itself)."""
    from fukikae.models import Assignments, Diarization

    path = names_path(work)
    if path.exists():
        result = read_json(path, NamesResult)
        if result.verdict not in HELD_OUT_VERDICTS:
            what = (
                "its subtitles (verdict none)"
                if result.verdict == "none"
                else f"its subtitle names (names verdict {result.verdict!r})"
            )
            raise HeldOutNamesError(
                f"held-out episode {episode} read {what}; add it to [names] holdout_episodes "
                "and rerun it from names, or score it as a tuning episode"
            )
    asg_path = Path(work) / "05-characters" / "assignments.json"
    if not asg_path.exists():
        return
    asg = read_json(asg_path, Assignments)
    rerun = f"rerun it from names (`fukikae run <ws> {episode} --from names`) before grading it"
    named = sorted(c.cluster for c in asg.clusters if c.name is not None or c.source == "name")
    if named:
        shown = ", ".join(named[:4]) + (f" and {len(named) - 4} more" if len(named) > 4 else "")
        raise HeldOutNamesError(
            f"held-out episode {episode}'s assignments hold name decisions ({shown}): its "
            f"characters stage read subtitle names; {rerun}"
        )
    diar = Path(work) / DIARIZE_DIR / "turns.json"
    if asg.turns is not None and diar.exists() and asg.turns != read_json(diar, Diarization).turns:
        from fukikae.names.align import VOICE_PREFIX

        split = sorted(
            {t.speaker.split("~", 1)[0] for t in asg.turns if f"~{VOICE_PREFIX}" in t.speaker}
        )
        named = [
            t.speaker for t in asg.turns if "~" in t.speaker and f"~{VOICE_PREFIX}" not in t.speaker
        ]
        if split and not named:  # the refined turns are an operator split's, not names'
            raise HeldOutNamesError(
                f"held-out episode {episode}'s assignments record an operator pitch split of "
                f"{', '.join(split)} (studio/splits/{episode}.json), not "
                f"{DIARIZE_DIR}/turns.json: a held-out grade reads the diarizer's own clusters; "
                f"remove the split and {rerun}, or score it as a tuning episode"
            )
        raise HeldOutNamesError(
            f"held-out episode {episode}'s assignments record other turns than "
            f"{DIARIZE_DIR}/turns.json (the names stage's refined turns); {rerun}"
        )
