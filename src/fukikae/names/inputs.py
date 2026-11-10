"""Which turns and clusters identity reads: the names stage's refined pair when its
`names.json` says names were used, else the diarize stage's (Task 4's helper for the characters
stage, `registry resolve`, `reembed`, `adapt` and `eval identity`)."""

from pathlib import Path

from fukikae.artifacts import read_json
from fukikae.names.models import NamesResult

NAMES_DIR = "04c-names"
DIARIZE_DIR = "04-diarize"


def names_path(work: Path) -> Path:
    return Path(work) / NAMES_DIR / "names.json"


def names_report_path(work: Path) -> Path:
    """The characters stage's `NamesReport` for a names-used episode."""
    return Path(work) / "05-characters" / "names-report.json"


def identity_inputs(work: Path) -> tuple[Path, Path, NamesResult | None]:
    """(turns.json, clusters.npz, names result) for an episode's work directory: the refined
    pair under `04c-names/` when names were used, else `04-diarize/`'s, with the result (None
    when the names stage never ran)."""
    path = names_path(work)
    result = read_json(path, NamesResult) if path.exists() else None
    d = Path(work) / (NAMES_DIR if result is not None and result.used else DIARIZE_DIR)
    return d / "turns.json", d / "clusters.npz", result


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
      refined turns).

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
            raise HeldOutNamesError(
                f"held-out episode {episode} read its subtitle names (names verdict "
                f"{result.verdict!r}); add it to [names] holdout_episodes and rerun it from "
                "names, or score it as a tuning episode"
            )
    asg_path = Path(work) / "05-characters" / "assignments.json"
    if not asg_path.exists():
        return
    asg = read_json(asg_path, Assignments)
    rerun = f"rerun it from names (`fukikae run <ws> {episode} --from names`) before grading it"
    named = sorted(c.cluster for c in asg.clusters if c.name is not None or c.source == "name")
    if named:
        raise HeldOutNamesError(
            f"held-out episode {episode}'s assignments hold name decisions ({', '.join(named)}): "
            f"its characters stage read subtitle names; {rerun}"
        )
    diar = Path(work) / DIARIZE_DIR / "turns.json"
    if asg.turns is not None and diar.exists() and asg.turns != read_json(diar, Diarization).turns:
        raise HeldOutNamesError(
            f"held-out episode {episode}'s assignments record other turns than "
            f"{DIARIZE_DIR}/turns.json (the names stage's refined turns); {rerun}"
        )
