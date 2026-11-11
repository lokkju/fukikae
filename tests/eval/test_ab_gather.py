"""A/B packs end to end through `gather_set` with the eval tts fakes (Task 7 review): stock
baselines stay out of the workspace's card store, workspace clone cards and their sidecars
still render, and the line gate (A3) routes lines as the tts stage does."""

from pathlib import Path

import pytest

from fukikae.cards.store import CardStore
from fukikae.eval import ab
from fukikae.pitch import GenderEstimate
from fukikae.stages import tts as tts_stage
from fukikae.workspace import Workspace
from tests.eval.test_tts import (  # noqa: F401  (fakes is an autouse fixture)
    FakeClone,
    _clone_backend,
    _long_turns,
    _workspace,
    fakes,
    tone_wav,
)


def _cards(root: Path) -> dict[str, str]:
    """Card id -> voice source, for every card file in a store's folder."""
    store = CardStore(root)
    return {p.stem: store.get(p.stem).voice.source for p in store.root.glob("*.toml")}


@pytest.fixture(autouse=True)
def _no_stretch(monkeypatch):
    """Fitting copies the render (ffmpeg's atempo is not what these tests are about)."""
    import shutil

    from fukikae import timefit

    def copy(src: Path, _plan: object, out: Path) -> float:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return 0.0

    monkeypatch.setattr(timefit, "apply_fit", copy)


def _setup(tmp_path: Path) -> tuple[Path, ab.SetSpec]:
    root = _workspace(tmp_path, emotions=False)
    _long_turns(root)
    work = Workspace(root).work_dir("S01E11")
    (work / "00-extract").mkdir(parents=True, exist_ok=True)
    tone_wav(work / "00-extract" / "audio.wav", 40.0, 16000)
    spec = ab.SetSpec(
        name="set1",
        workspace=root,
        episode="S01E11",
        candidates=(ab.Candidate("fakeclone"), ab.Candidate("fake2")),
        cues=3,
        characters=2,
    )
    return root, spec


@pytest.mark.usefixtures("_clone_backend")
def test_gather_set_keeps_stock_baselines_out_of_the_workspace(tmp_path):
    root, spec = _setup(tmp_path)
    before = _cards(root)
    tmp = tmp_path / "pack-tmp"
    cues, _scenes, notes = ab.gather_set(spec, seed=1, ledger=None, tmp=tmp, log=lambda *_: None)
    assert cues, notes
    added = {k: v for k, v in _cards(root).items() if k not in before}
    assert added and set(added.values()) == {"clone"}  # the fakeclone candidate's cards
    scratch = _cards(tmp / "set1" / "stock-cards")
    assert scratch and set(scratch.values()) == {"stock"}  # the fake2 baselines, apart
    store = CardStore(root)
    for cid in added:
        assert store.missing_sidecars(store.get(cid)) == []
    for c in cues:  # every render of both candidates was made and fitted
        assert set(c.renders) == {"fakeclone", "fake2"}
        assert all(r.path.exists() for r in c.renders.values())
        assert c.info["fakeclone"]["card"] in added
        assert c.info["fake2"]["card"] in scratch


@pytest.mark.usefixtures("_clone_backend")
def test_gather_set_gates_lines_like_the_stage(tmp_path, monkeypatch):
    """A line whose source pitch is the other band from its speaker's renders without the
    emotion audio, as in the tts stage; every other clone render keeps it."""
    _root, spec = _setup(tmp_path)
    spec = ab.SetSpec(**{**spec.__dict__, "candidates": (ab.Candidate("fakeclone"),)})
    built: list[list[str]] = []

    def gate(_work, cues, _root):
        built.append([c.id for c in cues])
        f = GenderEstimate("f", 1.0, 260.0, 1.0)
        return tts_stage.GenderGate({"c0001": f}, {"c0001": "m"})

    monkeypatch.setattr(tts_stage, "gender_gate", gate)
    ab.gather_set(spec, seed=1, ledger=None, tmp=tmp_path / "t", log=lambda *_: None)
    assert built  # the gate was measured over the episode's cues
    emo = {k["text"]: k["emo"] for k in FakeClone.KW}
    assert emo["natsu one"] is False
    assert emo and all(v for t, v in emo.items() if t != "natsu one")


@pytest.mark.usefixtures("_clone_backend")
def test_scene_gates_lines_like_the_stage(tmp_path, monkeypatch):
    root, spec = _setup(tmp_path)
    work = Workspace(root).work_dir("S01E11")
    tone_wav(work / "01-separate" / "background.wav", 40.0, 16000, amp=0.001)  # a quiet bed
    spec = ab.SetSpec(
        **{
            **spec.__dict__,
            "candidates": (ab.Candidate("fakeclone"),),
            "scene": ab.SceneSpec(start=0.0, seconds=20.0),
        }
    )
    built: list[int] = []

    def gate(_work, cues, _root):
        built.append(len(cues))
        f = GenderEstimate("f", 1.0, 260.0, 1.0)
        return tts_stage.GenderGate({"c0003": f}, {"c0003": "m"})

    monkeypatch.setattr(tts_stage, "gender_gate", gate)
    _cues, scenes, _notes = ab.gather_set(
        spec, seed=1, ledger=None, tmp=tmp_path / "t", log=lambda *_: None
    )
    assert scenes and len(built) == 1  # measured once for the set and its scene
    emo = [(k["text"], k["emo"]) for k in FakeClone.KW]
    assert ("natsu two is longer than its slot", False) in emo
    assert all(v for t, v in emo if t != "natsu two is longer than its slot")
