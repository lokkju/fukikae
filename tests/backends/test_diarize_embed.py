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
