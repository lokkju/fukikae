from pathlib import Path

import numpy as np
import pytest

from fukikae.config import BackendConfig
from fukikae.eval.identity.cache import EmbeddingCache, cache_root
from fukikae.eval.identity.segments import Segment


class CountingEmbed:
    """A fake embed function: deterministic rows from the spans, counting calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, cfg: BackendConfig, audio: Path, spans: list[tuple[float, float]]):
        self.calls += 1
        bias = float(cfg.options.get("bias", 0.0)) + len(audio.read_bytes())
        return np.array([[s, e, bias] for s, e in spans], dtype=np.float32)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    p = tmp_path / "vocals_16k.wav"
    p.write_bytes(b"RIFF....fake audio")
    return p


SEGS = [Segment("S01E01", 1.0, 3.5, "A"), Segment("S01E01", 4.0, 9.0, None, "SPEAKER_01")]
ECAPA = BackendConfig("ecapa", {})


def test_cache_root_is_in_the_workspace(tmp_path: Path):
    assert cache_root(tmp_path) == tmp_path / ".fukikae" / "cache" / "embeddings"


def test_same_inputs_hit(tmp_path: Path, audio: Path):
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    first = cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    again = cache.get_or_compute("S01E01", list(SEGS), BackendConfig("ecapa", {}), audio)
    assert embed.calls == 1
    np.testing.assert_array_equal(first, again)
    assert first.shape == (2, 3)
    # A fresh cache object over the same directory hits too: it's on disk.
    fresh = EmbeddingCache(tmp_path / "cache", embed=embed)
    fresh.get_or_compute("S01E01", SEGS, ECAPA, audio)
    assert embed.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_changed_audio_bytes_miss(tmp_path: Path, audio: Path):
    """Review Focus 4: a re-separation rewrites the audio; the old embeddings must not return."""
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    before = cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    # Same size, same name, different content.
    audio.write_bytes(b"RIFF....fake AUDIO")
    after = cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    assert embed.calls == 2
    assert before.shape == after.shape


def test_changed_embedder_option_misses(tmp_path: Path, audio: Path):
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    a = cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    b = cache.get_or_compute("S01E01", SEGS, BackendConfig("ecapa", {"bias": 1}), audio)
    c = cache.get_or_compute("S01E01", SEGS, BackendConfig("other", {}), audio)
    assert embed.calls == 3
    assert not np.array_equal(a, b)
    assert c.shape == a.shape


def test_changed_segments_miss(tmp_path: Path, audio: Path):
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    moved = [SEGS[0], Segment("S01E01", 4.0, 9.5, None, "SPEAKER_01")]
    cache.get_or_compute("S01E01", moved, ECAPA, audio)
    assert embed.calls == 2
    # Labels are not part of the key: relabeling the same spans hits.
    relabeled = [Segment("S01E01", 1.0, 3.5, "B"), Segment("S01E01", 4.0, 9.0, "C", "X")]
    cache.get_or_compute("S01E01", relabeled, ECAPA, audio)
    assert embed.calls == 2


def test_wrong_shape_on_disk_is_recomputed(tmp_path: Path, audio: Path):
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    (stored,) = list((tmp_path / "cache").rglob("*.npy"))
    np.save(stored, np.zeros((1, 3), dtype=np.float32))
    got = cache.get_or_compute("S01E01", SEGS, ECAPA, audio)
    assert embed.calls == 2 and got.shape == (2, 3)


def test_no_segments_never_embeds(tmp_path: Path, audio: Path):
    embed = CountingEmbed()
    cache = EmbeddingCache(tmp_path / "cache", embed=embed)
    got = cache.get_or_compute("S01E01", [], ECAPA, audio)
    assert embed.calls == 0 and len(got) == 0


class FakeBackend:
    vram_mb = 1234
    log: list[str] = []

    def __init__(self, options, device):
        self.options, self.device = options, device
        self.name = options.get("name", "x")

    def load(self):
        if self.options.get("fail"):
            raise RuntimeError("load failed")
        FakeBackend.log.append(f"load {self.name}")

    def unload(self):
        FakeBackend.log.append(f"unload {self.name}")

    def embed(self, _audio, spans):
        return np.ones((len(spans), 2), dtype=np.float32)


def test_resident_embedder_gpu_pattern(monkeypatch, audio: Path):
    from fukikae import runctx
    from fukikae.config import Config
    from fukikae.eval.identity import cache as mod

    FakeBackend.log = []
    waits = []
    monkeypatch.setattr(mod, "device_string", lambda _cfg: "cuda:0")
    monkeypatch.setattr(mod, "create", lambda _kind, cfg, device: FakeBackend(cfg.options, device))
    monkeypatch.setattr(
        mod, "wait_for_vram", lambda dev, need, timeout, **_kw: waits.append((dev, need, timeout))
    )
    monkeypatch.setattr(runctx, "vram_timeout_s", 7.0)
    a, b = BackendConfig("e", {"name": "a"}), BackendConfig("e", {"name": "b"})
    with mod.ResidentEmbedder(Config(gpu_device=0)) as embed:
        embed(a, audio, [(0.0, 1.0)])
        embed(a, audio, [(1.0, 2.0)])
        embed(b, audio, [(0.0, 1.0)])
    # One model resident at a time: a stays loaded across calls, is unloaded before b loads.
    assert FakeBackend.log == ["load a", "unload a", "load b", "unload b"]
    assert waits == [(0, 1234, 7.0), (0, 1234, 7.0)]


def test_resident_embedder_unloads_after_a_failed_load(monkeypatch, audio: Path):
    from fukikae.config import Config
    from fukikae.eval.identity import cache as mod

    FakeBackend.log = []
    monkeypatch.setattr(mod, "device_string", lambda _cfg: "cpu")
    monkeypatch.setattr(mod, "create", lambda _kind, cfg, device: FakeBackend(cfg.options, device))
    with mod.ResidentEmbedder(Config()) as embed, pytest.raises(RuntimeError):
        embed(BackendConfig("e", {"name": "a", "fail": True}), audio, [(0.0, 1.0)])
    assert FakeBackend.log == ["unload a"]
