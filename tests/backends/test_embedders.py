from pathlib import Path

import numpy as np
import pytest

from fukikae.backends import loader
from fukikae.backends.embed import _common
from fukikae.config import BackendConfig

NEW = {
    "wespeaker": ("fukikae.backends.embed.wespeaker", "WespeakerEmbedder", 256, "pyannote"),
    "wavlm_sv": ("fukikae.backends.embed.wavlm_sv", "WavlmSvEmbedder", 512, "wavlm"),
    "campplus": ("fukikae.backends.embed.sherpa", "CampplusEmbedder", 512, "sherpa"),
    "eres2netv2": ("fukikae.backends.embed.sherpa", "Eres2netv2Embedder", 192, "sherpa"),
    "titanet": ("fukikae.backends.embed.sherpa", "TitanetEmbedder", 192, "sherpa"),
}


@pytest.mark.parametrize("name", sorted(NEW))
def test_declarations_and_registration(name):
    import importlib

    mod, cls, dim, extra = NEW[name]
    c = getattr(importlib.import_module(mod), cls)
    assert (c.name, c.kind, c.dim) == (name, "embed", dim)
    assert loader.EXTRAS[name] == extra
    assert name in loader.names("embed")


@pytest.mark.parametrize("name", sorted(NEW))
def test_requires_load_and_empty_spans(name, tmp_path):
    import importlib

    mod, cls, dim, _ = NEW[name]
    e = getattr(importlib.import_module(mod), cls)({}, "cpu")
    with pytest.raises(RuntimeError, match="load"):
        e.embed(tmp_path / "a.wav", [(0.0, 1.0)])
    e._loaded = True  # noqa: SLF001
    assert e.embed(None, []).shape == (0, dim)  # pyright: ignore[reportArgumentType]


def test_cut_pads_short_spans_and_clamps():
    wav = np.arange(16000, dtype=np.float32)
    assert len(_common.cut(wav, 0.0, 0.1)) == _common.MIN_SAMPLES
    assert len(_common.cut(wav, 0.25, 0.75)) == 8000
    assert len(_common.cut(wav, 0.2, 1.0)) == 12800
    assert len(_common.cut(wav, 5.0, 6.0)) == _common.MIN_SAMPLES  # past the end: silence


def test_l2_rows_unit_norm_and_zero_row():
    out = _common.l2_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)
    assert np.isfinite(out).all()


def test_sherpa_model_files_are_cached(tmp_path, monkeypatch):
    import io

    from fukikae.backends.embed import sherpa

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls: list[str] = []

    def fake_urlopen(url, timeout=0):  # noqa: ARG001
        calls.append(url)
        return io.BytesIO(b"onnx")

    monkeypatch.setattr(sherpa.urllib.request, "urlopen", fake_urlopen)
    p = sherpa.fetch_model("m.onnx")
    assert p == tmp_path / "fukikae" / "models" / "m.onnx" and p.read_bytes() == b"onnx"
    assert sherpa.fetch_model("m.onnx") == p
    assert calls == [f"{sherpa.RELEASE}/m.onnx"]


class _Member:
    kind = "embed"
    loaded: list[str] = []
    unloaded: list[str] = []

    def __init__(self, name: str, dim: int, value: float) -> None:
        self.name, self.dim, self.value = name, dim, value

    def load(self) -> None:
        self.loaded.append(self.name)

    def unload(self) -> None:
        self.unloaded.append(self.name)

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:  # noqa: ARG002
        out = np.zeros((len(spans), self.dim), dtype=np.float32)
        out[:, 0] = self.value
        out[:, 1] = 1.0
        return out


def test_fusion_concatenates_normalized_members_and_renormalizes(monkeypatch):
    from fukikae.backends.embed import fusion

    made = {"a": _Member("a", 3, 5.0), "b": _Member("b", 2, 0.5)}
    seen: list[BackendConfig] = []

    def fake_create(kind, cfg, device):  # noqa: ARG001
        seen.append(cfg)
        return made[cfg.backend]

    monkeypatch.setattr(fusion, "create", fake_create)
    f = fusion.FusionEmbedder({"members": ["a", "b"], "member_options": {"b": {"x": 1}}}, "cpu")
    f.load()
    out = f.embed(Path("x.wav"), [(0, 1), (1, 2), (2, 3)])
    assert out.shape == (3, 5) and f.dim == 5
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    # each member counts equally: both halves have squared norm 1/2 after renormalizing
    np.testing.assert_allclose((out[:, :3] ** 2).sum(axis=1), 0.5, atol=1e-6)
    assert [c.options for c in seen] == [{}, {"x": 1}]
    assert _Member.unloaded == ["a", "b"]
    assert f.embed(Path("x.wav"), []).shape == (0, 5)


@pytest.mark.parametrize(
    "opts", [{}, {"members": ["a"]}, {"members": "a,b"}, {"members": ["a", "fusion"]}]
)
def test_fusion_rejects_bad_members(opts):
    from fukikae.backends.embed.fusion import FusionEmbedder

    with pytest.raises(ValueError, match="members"):
        FusionEmbedder(opts, "cpu")


def test_fusion_requires_load(tmp_path):
    from fukikae.backends.embed.fusion import FusionEmbedder

    with pytest.raises(RuntimeError, match="load"):
        FusionEmbedder({"members": ["a", "b"]}, "cpu").embed(tmp_path / "a.wav", [(0.0, 1.0)])


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(NEW))
def test_real_model_separates_speakers_on_fairytail_e12(name):
    import importlib

    from fukikae.eval.identity.cli import _ground_truth
    from fukikae.eval.identity.segments import segments
    from fukikae.workspace import Workspace

    root = Path("~/fukikae-ws/fairytail").expanduser()
    if not (root / "fukikae.toml").is_file():
        pytest.skip("fairytail workspace not available")
    ws = Workspace(root)
    eid = "S01E12"
    _, lines = _ground_truth(ws, eid, eid, "ass", None, 1, None, {"NABARU": "NEBARU"})
    segs = [s for s in segments(ws, eid, "diarization", lines) if s.label][:60]
    assert len({s.label for s in segs}) >= 2
    mod, cls, dim, _ = NEW[name]
    e = getattr(importlib.import_module(mod), cls)({}, "cuda:0")
    e.load()
    try:
        audio = ws.work_dir(eid) / "01-separate" / "vocals_16k.wav"
        emb = e.embed(audio, [(s.start, s.end) for s in segs])
    finally:
        e.unload()
    assert emb.shape == (len(segs), dim)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    sim = emb @ emb.T
    same, diff = [], []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (same if segs[i].label == segs[j].label else diff).append(sim[i, j])
    assert np.mean(same) > np.mean(diff)
