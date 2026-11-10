"""`names.inputs.check_held_out`: a held-out episode read no names, by its names.json and by
the assignments it is graded on."""

from pathlib import Path

import pytest

from fukikae.artifacts import write_json
from fukikae.models import Assignments, ClusterAssignment, Diarization, Turn
from fukikae.names.inputs import HeldOutNamesError, check_held_out, names_path
from fukikae.names.models import NamesResult

DIARIZE = [Turn(start=0.0, end=4.0, speaker="S0"), Turn(start=5.0, end=9.0, speaker="S1")]


def _holdout_names(work: Path, verdict: str = "holdout") -> None:
    names_path(work).parent.mkdir(parents=True, exist_ok=True)
    write_json(
        names_path(work),
        NamesResult(
            episode="S01E14", used=False, verdict=verdict, reason="held out", source=None,  # type: ignore[arg-type]
            offset_s=0.0, coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[],
            excluded=[], dropped_row_seconds=0.0, suggestions=[],
        ),
    )  # fmt: skip


def _work(tmp_path: Path, clusters: list[dict], turns: list[Turn] | None = DIARIZE) -> Path:
    work = tmp_path / "S01E14"
    (work / "04-diarize").mkdir(parents=True)
    (work / "05-characters").mkdir(parents=True)
    write_json(work / "04-diarize" / "turns.json", Diarization(backend="x", turns=DIARIZE))
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode="S01E14", threshold=0.6,
            clusters=[ClusterAssignment.model_validate(
                {"actor_id": 1, "character_id": 1, "score": None, "clean_seconds": 4.0,
                 "status": "matched", **c}) for c in clusters],
            turns=turns,
        ),
    )  # fmt: skip
    _holdout_names(work)
    return work


def test_acoustic_assignments_on_the_diarize_turns_pass(tmp_path: Path):
    check_held_out(_work(tmp_path, [{"cluster": "S0"}, {"cluster": "S1"}]), "S01E14")


def test_assignments_without_turns_pass(tmp_path: Path):
    """Assignments written before they recorded turns read the diarize pair."""
    check_held_out(_work(tmp_path, [{"cluster": "S0"}], turns=None), "S01E14")


def test_names_built_assignments_are_refused_though_names_json_says_holdout(tmp_path: Path):
    """The reviewer's reproduction: names.json rewritten to verdict holdout while the
    assignments still come from a characters run that read the names."""
    refined = [Turn(start=0.0, end=4.0, speaker="S0~natsu"), DIARIZE[1]]
    work = _work(
        tmp_path,
        [{"cluster": "S0~natsu", "name": "natsu", "source": "name"}, {"cluster": "S1"}],
        turns=refined,
    )
    with pytest.raises(HeldOutNamesError, match="S01E14.*assignments"):
        check_held_out(work, "S01E14")


@pytest.mark.parametrize(
    "cluster",
    [
        {"cluster": "S0", "name": "natsu"},
        {"cluster": "S0", "name": "natsu", "source": "lock"},
        {"cluster": "S0", "source": "name"},
    ],
)
def test_any_name_decision_is_refused(tmp_path: Path, cluster: dict):
    with pytest.raises(HeldOutNamesError, match="name decisions"):
        check_held_out(_work(tmp_path, [cluster, {"cluster": "S1"}]), "S01E14")


def test_a_lock_without_a_key_passes(tmp_path: Path):
    """A lock keeps a cluster whatever the names say; with no key it read none."""
    check_held_out(_work(tmp_path, [{"cluster": "S0", "source": "lock"}]), "S01E14")


def test_other_turns_than_the_diarizer_are_refused(tmp_path: Path):
    moved = [Turn(start=0.0, end=3.0, speaker="S0"), DIARIZE[1]]
    with pytest.raises(HeldOutNamesError, match="turns"):
        check_held_out(_work(tmp_path, [{"cluster": "S0"}], turns=moved), "S01E14")


def test_a_names_verdict_is_still_refused_first(tmp_path: Path):
    work = _work(tmp_path, [{"cluster": "S0"}])
    _holdout_names(work, "full")
    with pytest.raises(HeldOutNamesError, match="names verdict 'full'"):
        check_held_out(work, "S01E14")


def test_verdict_none_says_it_read_the_subtitles_not_their_names(tmp_path: Path):
    """No named track was found, so no names were read; the subtitles were."""
    work = _work(tmp_path, [{"cluster": "S0"}])
    _holdout_names(work, "none")
    with pytest.raises(HeldOutNamesError) as e:
        check_held_out(work, "S01E14")
    assert "read its subtitles (verdict none)" in str(e.value)
    assert "subtitle names" not in str(e.value)
