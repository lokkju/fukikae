"""A/B packs end to end through `gather_set` with the eval tts fakes (Task 7 review): stock
baselines stay out of the workspace's card store, and workspace clone cards and their
sidecars still render."""

from pathlib import Path

import pytest

from fukikae.cards.store import CardStore
from fukikae.eval import ab
from fukikae.workspace import Workspace
from tests.eval.test_tts import (  # noqa: F401  (fakes is an autouse fixture)
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
