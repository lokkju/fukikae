import numpy as np
import pytest

from fukikae.backends import loader
from fukikae.backends.diarize.pyannote import PyannoteDiarizer
from fukikae.backends.embed.ecapa import EcapaEmbedder


def test_declarations_and_registration():
    d, e = PyannoteDiarizer, EcapaEmbedder
    assert (d.name, d.kind) == ("pyannote", "diarize")
    assert (e.name, e.kind, e.dim) == ("ecapa", "embed", 192)
    assert loader.EXTRAS["pyannote"] == loader.EXTRAS["ecapa"] == "pyannote"
    assert "pyannote" in loader.names("diarize")
    assert "ecapa" in loader.names("embed")


def test_default_models():
    assert PyannoteDiarizer({}, "cpu").model == "pyannote/speaker-diarization-community-1"
    assert EcapaEmbedder({}, "cpu").model == "speechbrain/spkrec-ecapa-voxceleb"
    assert EcapaEmbedder({"model": "x/y"}, "cpu").model == "x/y"


def test_missing_token_names_model_page(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(
        RuntimeError, match="huggingface.co/pyannote/speaker-diarization-community-1"
    ):
        PyannoteDiarizer({}, "cpu").load()


def test_requires_load(tmp_path):
    with pytest.raises(RuntimeError, match="load"):
        PyannoteDiarizer({}, "cpu").diarize(tmp_path / "a.wav")
    with pytest.raises(RuntimeError, match="load"):
        EcapaEmbedder({}, "cpu").embed(tmp_path / "a.wav", [(0.0, 1.0)])


def test_embed_no_spans_needs_no_model():
    e = EcapaEmbedder({}, "cpu")
    e._loaded = True  # noqa: SLF001
    assert e.embed(None, []).shape == (0, 192)  # pyright: ignore[reportArgumentType]


@pytest.mark.gpu
def test_real_models_on_clip(clip_ep):
    audio = clip_ep.work / "01-separate" / "vocals_16k.wav"
    d = PyannoteDiarizer({}, "cuda:0")
    d.load()
    try:
        out = d.diarize(audio)
    finally:
        d.unload()
    assert out.turns
    spans = [(t.start, t.end) for t in out.turns if t.end - t.start > 1.0][:4]
    e = EcapaEmbedder({}, "cuda:0")
    e.load()
    try:
        emb = e.embed(audio, spans)
    finally:
        e.unload()
    assert emb.shape == (len(spans), 192)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


class FakePipeline:
    def __init__(self, clustering: dict) -> None:
        self.params = {"segmentation": {"min_duration_off": 0.0}, "clustering": dict(clustering)}
        self.embedding_exclude_overlap = True
        self.instantiated: list[dict] = []

    def parameters(self, instantiated: bool = False) -> dict:  # noqa: ARG002
        return {k: dict(v) for k, v in self.params.items()}

    def instantiate(self, params: dict) -> "FakePipeline":
        self.instantiated.append(params)
        self.params = params
        return self


VBX = {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}
AHC = {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7}


def test_tuning_options_reach_the_pipeline():
    d = PyannoteDiarizer({"clustering_threshold": 0.7, "exclude_overlap": False}, "cpu")
    p = FakePipeline(VBX)
    d.tune(p)
    assert p.params["clustering"] == {"threshold": 0.7, "Fa": 0.07, "Fb": 0.8}
    assert p.embedding_exclude_overlap is False
    a = FakePipeline(AHC)
    PyannoteDiarizer({"min_cluster_size": 5}, "cpu").tune(a)
    assert a.params["clustering"]["min_cluster_size"] == 5


def test_no_tuning_options_leave_the_pipeline_alone():
    p = FakePipeline(VBX)
    PyannoteDiarizer({}, "cpu").tune(p)
    assert p.instantiated == [] and p.embedding_exclude_overlap is True


def test_min_cluster_size_on_a_vbx_pipeline_is_an_error():
    with pytest.raises(ValueError, match="min_cluster_size"):
        PyannoteDiarizer({"min_cluster_size": 5}, "cpu").tune(FakePipeline(VBX))


@pytest.mark.parametrize(
    "opts",
    [
        {"clustering_threshold": "high"},
        {"clustering_threshold": True},
        {"min_cluster_size": 0},
        {"min_cluster_size": 2.5},
        {"exclude_overlap": "yes"},
    ],
)
def test_bad_tuning_options(opts):
    with pytest.raises(ValueError):
        PyannoteDiarizer(opts, "cpu")
