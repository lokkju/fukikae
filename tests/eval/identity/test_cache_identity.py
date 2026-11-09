from pathlib import Path

import numpy as np

from fukikae.backends import identity
from fukikae.config import BackendConfig
from fukikae.eval.identity.cache import EmbeddingCache
from fukikae.eval.identity.segments import Segment


def test_embedding_cache_misses_when_the_model_identity_changes(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    segs = [Segment("e", 0.0, 1.0, "x")]
    cfg = BackendConfig("wespeaker", {})
    calls: list[int] = []

    def embed(_cfg, _audio, spans):
        calls.append(1)
        return np.ones((len(spans), 3), dtype=np.float32)

    cache = EmbeddingCache(tmp_path / "c", embed)
    monkeypatch.setattr(identity, "model_identity", lambda _k, _c: ("m/one", "r1"))
    cache.get_or_compute("e", segs, cfg, Path(audio))
    cache.get_or_compute("e", segs, cfg, Path(audio))
    assert len(calls) == 1
    monkeypatch.setattr(identity, "model_identity", lambda _k, _c: ("m/two", "r1"))
    cache.get_or_compute("e", segs, cfg, Path(audio))
    assert len(calls) == 2
    monkeypatch.setattr(identity, "model_identity", lambda _k, _c: ("m/two", "r2"))
    cache.get_or_compute("e", segs, cfg, Path(audio))
    assert len(calls) == 3
