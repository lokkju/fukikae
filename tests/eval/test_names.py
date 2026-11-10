"""`fukikae eval names` (Task 10): cue attribution and reference purity, per protocol."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import read_json, write_json
from fukikae.cli import app
from fukikae.eval.names import names_eval
from fukikae.names.inputs import names_path
from fukikae.names.models import NamesResult
from fukikae.registry.store import Registry
from fukikae.workspace import Workspace
from tests.cards.test_auto import _named_episode
from tests.names.test_cli import HEAD
from tests.stages.test_names import _stamp

runner = CliRunner()

# (start, end, cluster, noise, ground-truth name): noise 0.001 is loud, 0.02 quiet
TUNE = [
    (0.0, 5.0, "S0~natsu", 0.02, "NATSU"),
    (10.0, 15.0, "S0~natsu", 0.02, "NATSU"),
    (20.0, 25.0, "S0~natsu", 0.02, "NATSU"),
    (30.0, 35.0, "S0~wendy", 0.001, "WENDY"),
    (40.0, 45.0, "S0~wendy", 0.001, "WENDY"),
]
# the held-out episode: one merged cluster, as diarization gives it without names
HOLD = [
    (0.0, 5.0, "S0", 0.02, "NATSU"),
    (10.0, 15.0, "S0", 0.02, "NATSU"),
    (20.0, 25.0, "S0", 0.02, "NATSU"),
    (30.0, 35.0, "S0", 0.001, "WENDY"),
]


def _gt(gt_dir: Path, eid: str, rows) -> None:
    gt_dir.mkdir(parents=True, exist_ok=True)
    events = [
        f"Dialogue: 0,{_stamp(a)},{_stamp(b)},Default,{name},0,0,0,,Line number {i} here."
        for i, (a, b, _, _, name) in enumerate(rows)
    ]
    (gt_dir / f"{eid}.ass").write_text("\n".join([*HEAD, *events, ""]), encoding="utf-8")


def _ws(tmp_path: Path, *, names_on_tune: bool = True, hold_verdict: str = "holdout"):
    """S01E12 (tune) and S01E14 (held out). Characters: 1 Natsu, 2 Wendy."""
    root = tmp_path / "ws"
    root.mkdir(parents=True)
    (root / "fukikae.toml").write_text(
        'target_language = "en"\nsource_language = "ja"\n[names]\nholdout_episodes = ["S01E14"]\n'
    )
    reg = Registry(Workspace(root).registry_path)
    one, two = (reg.create_actor_with_character(np.ones(4, np.float32))[1] for _ in range(2))
    reg.name_character(one, "Natsu", "natsu", "en")
    reg.name_character(two, "Wendy", "wendy", "en")
    reg.close()
    turns = [(a, b, c, n) for a, b, c, n, _ in TUNE]
    if names_on_tune:
        clusters = [
            {"cluster": "S0~natsu", "character_id": one, "name": "natsu", "source": "name"},
            {"cluster": "S0~wendy", "character_id": two, "name": "wendy", "source": "name"},
        ]
        names = [(a, b, g.lower(), False) for a, b, _, _, g in TUNE]
        _named_episode(root, "S01E12", turns, clusters, names=names)
    else:
        merged = [(a, b, "S0", n) for a, b, _, n in turns]
        _named_episode(root, "S01E12", merged, [{"cluster": "S0", "character_id": one}])
    _named_episode(root, "S01E14", [(a, b, c, n) for a, b, c, n, _ in HOLD],
                   [{"cluster": "S0", "character_id": one}])  # fmt: skip
    hold = NamesResult(
        episode="S01E14", used=False, verdict=hold_verdict, reason="held out", source=None,  # type: ignore[arg-type]
        offset_s=0.0, coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[],
        excluded=[], dropped_row_seconds=0.0, suggestions=[],
    )  # fmt: skip
    names_path(Workspace(root).work_dir("S01E14")).parent.mkdir(parents=True, exist_ok=True)
    write_json(names_path(Workspace(root).work_dir("S01E14")), hold)
    gt = tmp_path / "gt"
    _gt(gt, "S01E12", TUNE)
    _gt(gt, "S01E14", HOLD)
    return root, gt


def test_holdout_cue_accuracy_never_reads_holdout_names(tmp_path: Path):
    """Protocol A: the held-out episode's names stage read nothing, so its cues go by turn
    overlap alone: the merged cluster is character 1 (Natsu), right for Natsu's 15 s and
    wrong for Wendy's 5 s. The tuning episode, whose own names attributed its cues, is right
    throughout (graded on its own input)."""
    root, gt = _ws(tmp_path)
    s = names_eval(root, ["S01E12"], "S01E14", gt_file=gt)
    assert s.modal == {1: "natsu", 2: "wendy"}
    assert s.tune.cue_accuracy == pytest.approx(1.0) and s.tune.cues == 5
    assert s.holdout is not None
    assert s.holdout.cue_accuracy == pytest.approx(15.0 / 20.0)
    assert s.holdout.per_character == {"natsu": 1.0, "wendy": 0.0}
    assert s.hidden is None and s.tune.hidden_accuracy is None


@pytest.mark.parametrize("verdict", ["full", "partial", "none"])
def test_a_held_out_episode_that_read_names_is_refused(tmp_path: Path, verdict: str):
    from fukikae.names.inputs import HeldOutNamesError

    root, gt = _ws(tmp_path, hold_verdict=verdict)
    with pytest.raises(HeldOutNamesError, match="held-out episode S01E14"):
        names_eval(root, ["S01E12"], "S01E14", gt_file=gt)
    r = runner.invoke(app, ["eval", "names", str(root), "--tune", "S01E12", "--holdout",
                            "S01E14", "--gt-file", str(gt), "--no-ledger"])  # fmt: skip
    assert r.exit_code == 1 and "held-out episode S01E14" in r.output


def test_every_configured_held_out_episode_is_checked_even_as_tune(tmp_path: Path):
    """S01E14 is in [names] holdout_episodes: scored as a tuning episode it is still checked."""
    from fukikae.names.inputs import HeldOutNamesError

    root, gt = _ws(tmp_path, hold_verdict="full")
    with pytest.raises(HeldOutNamesError):
        names_eval(root, ["S01E12", "S01E14"], gt_file=gt)


def test_reference_purity_counts_wrong_speaker_seconds(tmp_path: Path):
    """Without names the merged cluster is all Natsu's (20 s of his speech against Wendy's
    10 s), and its loudest turns, the ones a clone picks first, are Wendy's: two of Natsu's
    three 5 s clips are Wendy's, a purity of a third. With names, his references are his
    own."""
    root, gt = _ws(tmp_path / "merged", names_on_tune=False)
    merged = names_eval(root, ["S01E12"], gt_file=gt)
    assert merged.modal == {1: "natsu"}
    assert merged.tune.purity_per_character["Natsu"] == pytest.approx(1 / 3, abs=1e-5)
    assert merged.tune.reference_seconds == pytest.approx(15.0)
    root, gt = _ws(tmp_path / "named")
    named = names_eval(root, ["S01E12"], gt_file=gt)
    assert named.tune.purity_per_character == {"Natsu": 1.0, "Wendy": 1.0}
    assert named.tune.reference_purity == pytest.approx(1.0)


def test_hidden_accuracy_scores_only_hidden_lines(tmp_path: Path):
    """Protocol B: Wendy's 40-45 line was withheld, so the names stage left its turn in the
    remainder (character 1) and its cue falls back to turn overlap: wrong. Only that cue is in
    the hidden split; the tune split holds every cue."""
    root, gt = _ws(tmp_path)
    work = Workspace(root).work_dir("S01E12")
    from fukikae.models import Assignments

    asg = read_json(work / "05-characters" / "assignments.json", Assignments)
    asg.turns[-1] = asg.turns[-1].model_copy(update={"speaker": "S0"})  # type: ignore[index]
    asg.clusters.append(asg.clusters[0].model_copy(
        update={"cluster": "S0", "name": None, "source": "voice"}))  # fmt: skip
    write_json(work / "05-characters" / "assignments.json", asg)
    n = read_json(names_path(work), NamesResult)
    write_json(names_path(work), n.model_copy(update={"hidden": [4]}))
    s = names_eval(root, ["S01E12"], gt_file=gt)
    assert s.hidden is not None and s.hidden.cues == 1
    assert s.hidden.cue_accuracy == pytest.approx(0.0)
    assert s.tune.hidden_accuracy == pytest.approx(0.0)
    assert s.tune.cue_accuracy == pytest.approx(20.0 / 25.0) and s.tune.cues == 5


def test_hidden_lines_are_drawn_again_under_use_off(tmp_path: Path, monkeypatch):
    """The baseline arm of protocol B: names off, the same lines drawn from the ground-truth
    track with the same seed, so both arms score the same cues."""
    import fukikae.stages.names as stage

    root, gt = _ws(tmp_path, names_on_tune=False)
    toml = root / "fukikae.toml"
    toml.write_text(toml.read_text() + 'use = "off"\nhide_fraction = 0.2\nhide_seed = 0\n')
    seen = {}

    def hide(lines, fraction, seed, episode):
        seen[episode] = (fraction, seed)
        return [ln.id for ln in lines if ln.start == 40.0]

    monkeypatch.setattr(stage, "hide_lines", hide)
    s = names_eval(root, ["S01E12"], gt_file=gt)
    assert seen == {"S01E12": (0.2, 0)}
    assert s.hidden is not None and s.hidden.cues == 1


def test_ledger_rows_have_stage_names_and_one_row_per_split(tmp_path: Path):
    root, gt = _ws(tmp_path)
    ledger = tmp_path / "results.jsonl"
    out = tmp_path / "run.json"
    r = runner.invoke(app, ["eval", "names", str(root), "--tune", "S01E12", "--holdout",
                            "S01E14", "--gt-file", str(gt), "--test-set", "ft", "--tag",
                            "names-arm", "--ledger", str(ledger), "--json", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "holdout: cue accuracy 75.0% over 4 cues" in r.output
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [(x["stage"], x["split"], x["episodes"]) for x in rows] == [
        ("names", "tune", ["S01E12"]), ("names", "holdout", ["S01E14"])
    ]  # fmt: skip
    tune = rows[0]
    assert tune["backend"] == "subtitle-names" and tune["test_set"] == "ft"
    assert tune["tag"] == "names-arm" and tune["options"]["holdout_episodes"] == ["S01E14"]
    assert tune["options"]["use"] == "auto" and tune["options"]["turn_share"] == 0.6
    assert tune["metrics"]["cue_accuracy"] == 1.0 and tune["metrics"]["reference_purity"] == 1.0
    assert rows[1]["metrics"]["cue_accuracy"] == 0.75
    assert "reference_purity" not in rows[1]["metrics"]
    assert tune["detail"]["modal"] == {"1": "natsu", "2": "wendy"}
    assert json.loads(out.read_text())["holdout"]["cue_accuracy"] == 0.75


def test_the_ledger_takes_names_rows_and_the_hidden_split():
    from fukikae.eval.ledger import SPLITS, STAGES

    assert "names" in STAGES and "hidden" in SPLITS
    assert STAGES[:4] == ("asr", "identity", "embedder", "tts")  # GT2: added, none changed


def test_a_held_out_episode_with_names_built_assignments_is_refused(tmp_path: Path):
    """The reviewer's reproduction: S01E14 ran with names, then its names.json was rewritten
    to verdict holdout; the assignments still hold its name decisions."""
    from fukikae.names.inputs import HeldOutNamesError

    root, gt = _ws(tmp_path)
    work = Workspace(root).work_dir("S01E14")
    from fukikae.models import Assignments

    path = work / "05-characters" / "assignments.json"
    asg = read_json(path, Assignments)
    asg.clusters[0] = asg.clusters[0].model_copy(update={"name": "natsu", "source": "name"})
    write_json(path, asg)
    with pytest.raises(HeldOutNamesError, match="S01E14's assignments hold name decisions"):
        names_eval(root, ["S01E12"], "S01E14", gt_file=gt)
