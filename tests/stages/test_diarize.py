from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from doppio.artifacts import read_json
from doppio.config import Config
from doppio.models import Diarization, Turn
from doppio.stages import diarize
from doppio.stages.diarize import MAX_SPANS_PER_CLUSTER, DiarizeStage, clean_spans
from doppio.workspace import Episode

LOG: list[Any] = []


def test_clean_spans_drop_overlap_and_short():
    turns = [
        Turn(start=0, end=5, speaker="A"),
        Turn(start=4, end=8, speaker="B"),
        Turn(start=10, end=11.5, speaker="A"),
        Turn(start=12, end=20, speaker="A"),
    ]
    assert clean_spans(turns, "A") == [(12.1, 19.9)]


def test_clean_spans_ignores_touching_turns_and_same_speaker_overlap():
    turns = [
        Turn(start=0, end=4, speaker="A"),
        Turn(start=3, end=7, speaker="A"),
        Turn(start=7, end=12, speaker="B"),
    ]
    got = clean_spans(turns, "A")
    assert got == [(0.1, 3.9), (3.1, 6.9)]


class FakeDiarizer:
    name: ClassVar[str] = "fake-diar"
    kind: ClassVar[str] = "diarize"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 10

    def __init__(self, options: dict[str, Any], _device: str) -> None:
        LOG.append(("diar-init", options))

    def load(self) -> None:
        LOG.append("diar-load")

    def unload(self) -> None:
        LOG.append("diar-unload")

    def diarize(self, _audio: Path) -> Diarization:
        LOG.append("diarize")
        return Diarization(
            backend="fake",
            turns=[
                Turn(start=0, end=6, speaker="S0"),
                Turn(start=8, end=9, speaker="S1"),  # too short: no clean span
                Turn(start=10, end=16, speaker="S0"),
            ],
        )


class FakeEmbedder:
    name: ClassVar[str] = "fake-emb"
    kind: ClassVar[str] = "embed"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 10
    dim: ClassVar[int] = 4

    def __init__(self, options: dict[str, Any], _device: str) -> None:
        LOG.append(("emb-init", options))

    def load(self) -> None:
        LOG.append("emb-load")

    def unload(self) -> None:
        LOG.append("emb-unload")

    def embed(self, _audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        LOG.append(("embed", list(spans)))
        out = np.zeros((len(spans), 4), dtype=np.float32)
        out[:, 0] = 1.0
        return out


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "01-separate").mkdir(parents=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


@pytest.fixture
def fakes(monkeypatch):
    LOG.clear()
    classes = {"diarize": FakeDiarizer, "embed": FakeEmbedder}
    monkeypatch.setattr(diarize, "create", lambda kind, bc, dev: classes[kind](bc.options, dev))
    monkeypatch.setattr(diarize, "device_string", lambda _cfg: "cpu")


@pytest.mark.usefixtures("fakes")
def test_stage_writes_turns_and_clusters_with_empty_cluster(ep):
    DiarizeStage().run(ep, Config(raw={"diarize": {"backend": "x", "min_speakers": 2}}))
    d = read_json(ep.work / "04-diarize" / "turns.json", Diarization)
    assert [t.speaker for t in d.turns] == ["S0", "S1", "S0"]
    z = np.load(ep.work / "04-diarize" / "clusters.npz")
    assert list(z["labels"]) == ["S0", "S1"]
    assert z["S0/embeddings"].shape == (2, 4)
    assert z["S0/spans"].shape == (2, 2)
    assert float(z["S0/clean_seconds"]) == pytest.approx(5.8 + 5.8)
    assert z["S1/embeddings"].shape == (0, 4)
    assert z["S1/spans"].shape == (0, 2)
    assert float(z["S1/clean_seconds"]) == 0.0


@pytest.mark.usefixtures("fakes")
def test_models_never_resident_together(ep):
    DiarizeStage().run(ep, Config(raw={}))
    assert [e for e in LOG if isinstance(e, str)] == [
        "diar-load",
        "diarize",
        "diar-unload",
        "emb-load",
        "emb-unload",
    ]
    assert LOG.index("diar-unload") < LOG.index("emb-load")


@pytest.mark.usefixtures("fakes")
def test_span_cap_keeps_longest(ep, monkeypatch):
    class Many(FakeDiarizer):
        def diarize(self, _audio: Path) -> Diarization:
            turns = [
                Turn(start=i * 10, end=i * 10 + 2 + (i % 7) * 0.1, speaker="S0")
                for i in range(MAX_SPANS_PER_CLUSTER + 20)
            ]
            return Diarization(backend="fake", turns=turns)

    classes = {"diarize": Many, "embed": FakeEmbedder}
    monkeypatch.setattr(diarize, "create", lambda kind, bc, dev: classes[kind](bc.options, dev))
    DiarizeStage().run(ep, Config(raw={}))
    z = np.load(ep.work / "04-diarize" / "clusters.npz")
    spans = z["S0/spans"]
    assert len(spans) == MAX_SPANS_PER_CLUSTER
    lengths = spans[:, 1] - spans[:, 0]
    # 70 clean spans of 1.8 + 0.1 * (i % 7) s, ten per length; the 20 dropped are the two
    # shortest lengths (1.8, 1.9), so every kept span is at least 2.0 s.
    assert lengths.min() >= 2.0 - 1e-6
    assert list(lengths) == sorted(lengths, reverse=True)
    # clean_seconds totals all 70 spans, not only the 50 embedded.
    assert float(z["S0/clean_seconds"]) == pytest.approx(10 * sum(1.8 + 0.1 * k for k in range(7)))


@pytest.mark.usefixtures("fakes")
def test_diarize_passes_table_options(ep):
    DiarizeStage().run(ep, Config(raw={"diarize": {"min_speakers": 2}}))
    assert ("diar-init", {"min_speakers": 2}) in LOG


@pytest.mark.gpu
def test_chimera_clip_has_two_clusters(clip_ep):
    DiarizeStage().run(clip_ep, Config(source_language="ko"))
    z = np.load(clip_ep.work / "04-diarize" / "clusters.npz")
    labels = list(z["labels"])
    assert len(labels) >= 2
    for lab in labels:
        emb = z[f"{lab}/embeddings"]
        if len(emb):
            assert emb.shape[1] == 192
            np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
