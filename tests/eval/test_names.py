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
    from fukikae.models import Diarization

    # the names stage's refined pair says the same (the turn stayed in the remainder)
    write_json(work / "04c-names" / "turns.json", Diarization(backend="x", turns=asg.turns or []))
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
    assert "holdout S01E14: cue accuracy 75.0% over 4 cues" in r.output
    # F9: no media source here, so the translate settings can't be read: said, not hidden
    assert "warning: S01E12: the translate settings can't be read" in r.output
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
    # purity per split, over the split's own picks: the held-out merged cluster's loudest
    # turn is Wendy's
    assert rows[1]["metrics"]["reference_purity"] == pytest.approx(2 / 3, abs=1e-5)
    assert rows[1]["metrics"]["foreign_seconds"] == 5.0
    assert tune["detail"]["ceiling"] is True and rows[1]["detail"]["ceiling"] is False
    assert "tune (ceiling: graded on the names its run read) S01E12" in r.output
    assert tune["metrics"]["max_characters_per_name"] == 1
    assert tune["detail"]["characters_per_name"] == {"natsu": 1, "wendy": 1}
    assert tune["detail"]["modal"] == {"1": "natsu", "2": "wendy"}
    assert json.loads(out.read_text())["holdout"]["cue_accuracy"] == 0.75


def test_the_ledger_takes_names_rows_and_the_hidden_split():
    from fukikae.eval.ledger import SPLITS, STAGES

    assert "names" in STAGES and "hidden" in SPLITS and "ceiling" in SPLITS
    assert SPLITS[:3] == ("tune", "holdout", "hidden")  # GT2: added, none changed
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


def test_an_episode_whose_assignments_predate_its_names_is_refused(tmp_path: Path):
    """S01E12's names stage refined its clusters after its characters stage ran: the
    assignments answer other turns, so its cues and references would be graded on clusters
    the names never saw."""
    root, gt = _ws(tmp_path)
    work = Workspace(root).work_dir("S01E12")
    from fukikae.models import Assignments

    path = work / "05-characters" / "assignments.json"
    asg = read_json(path, Assignments)
    write_json(path, asg.model_copy(update={"turns": asg.turns[:-1]}))  # type: ignore[index]
    with pytest.raises(
        ValueError, match="S01E12: its assignments.json is older.*characters stage first"
    ):
        names_eval(root, ["S01E12"], gt_file=gt)
    r = runner.invoke(app, ["eval", "names", str(root), "--tune", "S01E12", "--gt-file",
                            str(gt), "--no-ledger"])  # fmt: skip
    assert r.exit_code == 1 and "run the characters stage first" in r.output


# -- the ceiling arm (B2) ------------------------------------------------------------------


def _ceiling_ws(tmp_path: Path):
    """_ws, with S01E14 run with its names (the ceiling arm): its Wendy turn is the sub-cluster
    S0~wendy on character 2, and a third character (Charle, 3) speaks only there."""
    root, gt = _ws(tmp_path)
    (root / "fukikae.toml").write_text('target_language = "en"\nsource_language = "ja"\n')
    with Registry(Workspace(root).registry_path) as reg:
        three = reg.create_actor_with_character(np.ones(4, np.float32))[1]
        reg.name_character(three, "Charle", "charle", "en")
    rows = [*HOLD[:3], (30.0, 35.0, "S0~wendy", 0.001, "WENDY"), (40.0, 45.0, "S1", 0.01, "CHARLE")]
    clusters = [
        {"cluster": "S0", "character_id": 1},
        {"cluster": "S0~wendy", "character_id": 2, "name": "wendy", "source": "name"},
        {"cluster": "S1", "character_id": three},
    ]
    turns = [(a, b, c, n) for a, b, c, n, _ in rows]
    names = [(a, b, g.lower(), False) for a, b, _, _, g in rows]
    _named_episode(root, "S01E14", turns, clusters, names=names)
    _gt(gt, "S01E14", rows)
    return root, gt, three


def test_ceiling_scores_a_names_read_episode_without_the_held_out_check(tmp_path: Path):
    """--ceiling S01E14: its names were read, so --holdout refuses it; as a ceiling it is
    scored, labelled so, and the modal names still come from S01E12 alone (Charle, who
    speaks only in S01E14, has none, so its cue is wrong)."""
    from fukikae.names.inputs import HeldOutNamesError

    root, gt, three = _ceiling_ws(tmp_path)
    with pytest.raises(HeldOutNamesError):
        names_eval(root, ["S01E12"], "S01E14", gt_file=gt)
    s = names_eval(root, ["S01E12"], ceiling="S01E14", gt_file=gt)
    assert s.modal == {1: "natsu", 2: "wendy"} and three not in s.modal
    assert s.holdout is None and s.ceiling is not None and s.ceiling.ceiling
    assert s.ceiling.cue_accuracy == pytest.approx(20.0 / 25.0)
    assert s.ceiling.per_character == {"charle": 0.0, "natsu": 1.0, "wendy": 1.0}
    ledger = tmp_path / "results.jsonl"
    r = runner.invoke(app, ["eval", "names", str(root), "--tune", "S01E12", "--ceiling",
                            "S01E14", "--gt-file", str(gt), "--ledger", str(ledger)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "ceiling (graded on its own names) S01E14: cue accuracy 80.0%" in r.output
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [(x["split"], x["episodes"], x["detail"]["ceiling"]) for x in rows] == [
        ("tune", ["S01E12"], True), ("ceiling", ["S01E14"], True)
    ]  # fmt: skip


def test_an_episode_is_tune_holdout_or_ceiling_never_two(tmp_path: Path):
    root, gt, _ = _ceiling_ws(tmp_path)
    with pytest.raises(ValueError, match="named twice"):
        names_eval(root, ["S01E12"], ceiling=["S01E12"], gt_file=gt)


# -- reference purity in ground-truth seconds (B4) ----------------------------------------


def _line(start: float, end: float, speaker: str):
    from fukikae.eval.groundtruth.base import GroundTruthLine

    return GroundTruthLine(start=start, end=end, speaker=speaker)


def test_reference_split_counts_pure_foreign_and_unverifiable_seconds():
    from fukikae.eval.names import reference_split

    gt = [_line(0.0, 5.0, "NATSU"), _line(5.0, 10.0, "CHARLE"), _line(12.0, 14.0, "NATSU")]
    assert reference_split(gt, 0.0, 10.0, "natsu") == pytest.approx((5.0, 5.0, 0.0))
    assert reference_split(gt, 9.0, 15.0, "natsu") == pytest.approx((2.0, 1.0, 3.0))
    both = [_line(0.0, 4.0, "NATSU"), _line(3.0, 6.0, "WENDY")]  # an overlap is foreign
    assert reference_split(both, 0.0, 6.0, "natsu") == pytest.approx((3.0, 3.0, 0.0))


def test_a_pick_half_natsu_half_charle_is_half_foreign(tmp_path: Path, monkeypatch):
    """B4: Natsu's one reference covers a Natsu line and then a Charle line; the old 60%-cover
    label called it unlabelled (neither name covers 60%) and so impure without saying why.
    Now it is 5 s pure and 5 s foreign, purity 50%; a second pick in a gap between lines is
    2 s unverifiable, reported beside it and left out of the purity."""
    from fukikae.cards import auto
    from fukikae.cards.select import RefPick

    root, gt = _ws(tmp_path)
    rows = [(0.0, 5.0, "S0~natsu", 0.02, "NATSU"), (5.0, 10.0, "S0~natsu", 0.02, "CHARLE"),
            *TUNE[1:]]  # fmt: skip
    _gt(gt, "S01E12", rows)
    real = auto.character_references

    def refs(ws, order, cid, *a, **k):
        if cid == 1:
            return order[0], [RefPick(0.0, 10.0, "x", 30.0), RefPick(25.0, 27.0, "x", 30.0)], True
        return real(ws, order, cid, *a, **k)

    monkeypatch.setattr(auto, "character_references", refs)
    s = names_eval(root, ["S01E12"], gt_file=gt)
    assert s.modal[1] == "natsu"
    natsu = s.tune.purity_seconds_per_character["Natsu"]
    assert natsu == {"pure_seconds": 5.0, "foreign_seconds": 5.0, "unverifiable_seconds": 2.0,
                     "unverifiable_share": pytest.approx(2 / 12, abs=1e-5)}  # fmt: skip
    assert s.tune.purity_per_character["Natsu"] == pytest.approx(0.5)
    assert s.tune.foreign_seconds == pytest.approx(5.0)
    out = runner.invoke(app, ["eval", "names", str(root), "--tune", "S01E12", "--gt-file",
                              str(gt), "--no-ledger"]).output  # fmt: skip
    assert (
        "purity Natsu                50.0% (pure 5.0 s, foreign 5.0 s, unverifiable 2.0 s)" in out
    )


# -- the modal map leaves hidden turns out (B5) -------------------------------------------


def test_modal_names_skip_turns_under_hidden_lines():
    """Character 1 speaks 5 s as Natsu and, in two hidden lines, 8 s as Wendy: scored on the
    hidden lines, the Wendy turns must not make Wendy its name."""
    from fukikae.eval.names import modal_names
    from fukikae.models import Assignments, ClusterAssignment, Turn

    asg = Assignments(
        episode="S01E12", threshold=0.6,
        clusters=[ClusterAssignment(cluster="S0", actor_id=1, character_id=1, score=None,
                                    clean_seconds=13.0, status="matched")],
        turns=[Turn(start=0.0, end=5.0, speaker="S0"), Turn(start=10.0, end=14.0, speaker="S0"),
               Turn(start=20.0, end=24.0, speaker="S0")],
    )  # fmt: skip
    gt = [_line(0.0, 5.0, "NATSU"), _line(10.0, 14.0, "WENDY"), _line(20.0, 24.0, "WENDY")]
    assert modal_names([(asg, gt)]) == {1: "wendy"}
    assert modal_names([(asg, gt)], [[(10.0, 14.0), (20.0, 24.0)]]) == {1: "natsu"}


# -- the subtitle offset (F2) -------------------------------------------------------------


def test_ground_truth_lines_are_aligned_before_labelling_turns(tmp_path: Path):
    """The ground-truth track runs 3 s late against the speech: unaligned, no 5 s turn has a
    name over 60% of it, so no character would get a modal name. eval names aligns the lines
    as eval identity does."""
    root, gt = _ws(tmp_path, names_on_tune=False)
    late = [(a + 3.0, b + 3.0, c, n, g) for a, b, c, n, g in TUNE]
    _gt(gt, "S01E12", late)
    s = names_eval(root, ["S01E12"], gt_file=gt)
    assert s.modal == {1: "natsu"}
    assert s.tune.reference_seconds > 0


def test_a_track_that_does_not_line_up_is_refused_unless_forced(tmp_path: Path):
    root, gt = _ws(tmp_path, names_on_tune=False)
    _gt(gt, "S01E12", [(a + 100.0, b + 100.0, c, n, g) for a, b, c, n, g in TUNE])
    with pytest.raises(ValueError, match="do not line up"):
        names_eval(root, ["S01E12"], gt_file=gt)
    assert names_eval(root, ["S01E12"], gt_file=gt, force=True).modal == {}


# -- the hidden lines are the same track's in both arms (F4) ------------------------------


def test_hidden_lines_match_between_the_names_and_off_arms(tmp_path: Path, monkeypatch):
    """The real names stage hides lines with the real hide_lines; the off arm draws them again
    from the ground-truth track. Both give the same spans, and a names.json whose hidden lines
    aren't the ground-truth track's draw is refused."""
    from fukikae import media
    from fukikae.config import load_config
    from fukikae.eval.names import _hidden_spans
    from fukikae.stages.names import NamesStage
    from tests.names.test_cli import E01, _episode

    root = tmp_path / "ws"
    media_dir = tmp_path / "media"
    root.mkdir()
    media_dir.mkdir()
    base = f'target_language = "en"\nsource_language = "ja"\nsource = "{media_dir}"\n'
    hide = "[names]\nhide_fraction = 0.2\nhide_seed = 0\n"
    (root / "fukikae.toml").write_text(base + hide)
    monkeypatch.setattr(media, "probe", lambda *_a, **_k: media.ProbeResult())
    _episode(root, media_dir, "S01E01", E01)
    track = media_dir / "S01E01.en.ass"
    cfg = load_config(root, user_config=None)
    ep = Workspace(root).find_episode("S01E01")
    NamesStage(cfg).run(ep, cfg)
    names = read_json(names_path(ep.work), NamesResult)
    assert names.used and names.hidden
    named_arm = _hidden_spans(cfg, "S01E01", track, names)
    (root / "fukikae.toml").write_text(base + hide + 'use = "off"\n')
    off = load_config(root, user_config=None)
    NamesStage(off).run(ep, off)
    off_names = read_json(names_path(ep.work), NamesResult)
    assert off_names.verdict == "off"
    assert _hidden_spans(off, "S01E01", track, off_names) == named_arm
    assert len(named_arm) == len(names.hidden)
    other = names.model_copy(update={"hidden": names.hidden[1:]})
    with pytest.raises(ValueError, match="other cues"):
        _hidden_spans(cfg, "S01E01", track, other)
