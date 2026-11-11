from pathlib import Path

import numpy as np
import pytest

from fukikae.backends import loader
from fukikae.backends.embed import _common
from fukikae.config import BackendConfig

NEW = {
    "wespeaker": ("fukikae.backends.embed.wespeaker", "WespeakerEmbedder", 256, "pyannote"),
    "wavlm_sv": ("fukikae.backends.embed.wavlm_sv", "WavlmSvEmbedder", 512, "wavlm"),
    "campplus": ("fukikae.backends.embed.sherpa", "CampplusEmbedder", 192, "sherpa"),
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


def _write_wav(path: Path, seconds: float = 4.0) -> Path:
    import wave

    t = np.arange(int(seconds * 16000)) / 16000
    pcm = (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return path


SPANS = [(0.0, 1.0), (1.0, 1.2), (2.0, 3.5)]  # the 0.2 s span is padded to 0.5 s


def test_wespeaker_embed_body_with_stub_net(tmp_path):
    import torch

    from fukikae.backends.embed.wespeaker import WespeakerEmbedder

    seen: list[tuple[int, ...]] = []

    def net(x):
        seen.append(tuple(x.shape))
        return torch.full((x.shape[0], 256), 3.0) + torch.arange(256) / 100

    e = WespeakerEmbedder({}, "cpu")
    e._net, e._loaded = net, True  # noqa: SLF001
    out = e.embed(_write_wav(tmp_path / "a.wav"), SPANS)
    assert out.shape == (3, 256) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert seen == [(1, 1, 16000), (1, 1, 8000), (1, 1, 24000)]  # (batch, channel, samples)


def test_wavlm_embed_body_with_stub_model(tmp_path):
    import torch

    from fukikae.backends.embed.wavlm_sv import WavlmSvEmbedder

    calls: list[tuple[int, int]] = []

    class Inputs(dict):
        def to(self, _dev):
            return self

    def fe(wavs, sampling_rate, return_tensors, padding):  # noqa: ARG001
        calls.append((len(wavs), sampling_rate))
        assert return_tensors == "pt" and len(wavs[0]) >= 8000
        return Inputs(input_values=torch.zeros(1, len(wavs[0])))

    class Out:
        embeddings = torch.arange(1.0, 513.0).reshape(1, 512)

    e = WavlmSvEmbedder({}, "cpu")
    e._fe, e._net, e._loaded = fe, lambda **_kw: Out(), True  # noqa: SLF001
    out = e.embed(_write_wav(tmp_path / "a.wav"), SPANS)
    assert out.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert calls == [(1, 16000)] * 3


@pytest.mark.parametrize("name", ["campplus", "eres2netv2", "titanet"])
def test_sherpa_embed_body_with_stub_extractor(name, tmp_path):
    from fukikae.backends.embed import sherpa

    cls = {
        "campplus": sherpa.CampplusEmbedder,
        "eres2netv2": sherpa.Eres2netv2Embedder,
        "titanet": sherpa.TitanetEmbedder,
    }[name]
    fed: list[tuple[int, int, bool]] = []

    class Stream:
        def accept_waveform(self, sample_rate, waveform):
            fed.append((sample_rate, len(waveform), waveform.dtype == np.float32))

        def input_finished(self):
            fed.append((0, 0, True))

    class Extractor:
        def create_stream(self):
            return Stream()

        def compute(self, _stream):
            return [float(i + 1) for i in range(cls.dim)]  # a plain list, as sherpa returns

    e = cls({}, "cpu")
    e._ex, e._loaded = Extractor(), True  # noqa: SLF001
    out = e.embed(_write_wav(tmp_path / "a.wav"), SPANS)
    assert out.shape == (3, cls.dim) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    waves = [f for f in fed if f[0]]
    assert waves == [(16000, 16000, True), (16000, 8000, True), (16000, 24000, True)]


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

    class A:
        dim = 3

    class B:
        dim = 2

    monkeypatch.setattr(fusion, "create", fake_create)
    monkeypatch.setattr(fusion, "backend_class", lambda _k, n: {"a": A, "b": B}[n])
    f = fusion.FusionEmbedder({"members": ["a", "b"], "member_options": {"b": {"x": 1}}}, "cpu")
    assert f.dim == 0
    f.load()
    assert f.dim == 5  # known before the first embed
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


def _real_params():
    return [
        pytest.param(n, marks=pytest.mark.slow if NEW[n][3] == "sherpa" else pytest.mark.gpu)
        for n in sorted(NEW)
    ]


@pytest.mark.parametrize("name", _real_params())
def test_real_model_separates_speakers_on_fairytail_e12(name, fukikae_ws: Path):
    import importlib

    from fukikae.eval.identity.cli import _ground_truth
    from fukikae.eval.identity.segments import segments
    from fukikae.workspace import Workspace

    root = fukikae_ws / "fairytail"
    if not (root / "fukikae.toml").is_file():
        pytest.skip("fairytail workspace not available")
    ws = Workspace(root)
    eid = "S01E12"
    _, lines = _ground_truth(ws, eid, eid, "ass", None, 1, None, {"NABARU": "NEBARU"})
    segs = [s for s in segments(ws, eid, "diarization", lines) if s.label][:60]
    assert len({s.label for s in segs}) >= 2
    mod, cls, _, _ = NEW[name]
    device = "cpu" if NEW[name][3] == "sherpa" else "cuda:0"
    e = getattr(importlib.import_module(mod), cls)({}, device)
    dim = e.dim  # B16: what the embedder declares, which the real model must give
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
    # absolute floor: a broken embedding scores near 0; wespeaker measures 0.49, campplus 0.44
    assert np.mean(same) >= 0.3
    assert np.mean(same) > np.mean(diff) + 0.03
