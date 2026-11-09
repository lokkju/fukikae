from pathlib import Path

import numpy as np
import pytest

from fukikae.backends.tag.emotion2vec import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Emotion2vecTagger,
    best_emotion,
)
from fukikae.models import Turn

LABELS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


def scores(**kw: float) -> dict[str, list]:
    vals = [kw.get(lab.split("/")[-1], 0.0) for lab in LABELS]
    return {"labels": LABELS, "scores": vals}


def test_best_emotion_maps_label_and_score():
    assert best_emotion(scores(angry=0.8, neutral=0.1)) == ("angry", 0.8)
    assert best_emotion(scores(happy=0.6, surprised=0.3)) == ("happy", 0.6)


def test_best_emotion_drops_other_and_unknown_winners():
    assert best_emotion(scores(other=0.9, neutral=0.05)) is None
    assert best_emotion({"labels": LABELS, "scores": [0.0] * 8 + [0.7]}) is None
    assert best_emotion({"labels": [], "scores": []}) is None


def test_defaults_are_cpu_with_a_pinned_revision():
    t = Emotion2vecTagger({}, "cpu")
    assert (t.model, t.revision) == (DEFAULT_MODEL, DEFAULT_REVISION)
    assert len(DEFAULT_REVISION) == 40
    assert Emotion2vecTagger.vram_mb == 0
    assert Emotion2vecTagger.devices == {"cpu", "cuda"}


def test_tag_writes_one_scored_span_per_turn_and_no_events(tmp_path: Path, monkeypatch):
    from fukikae.backends.tag import emotion2vec

    wav = tmp_path / "v.wav"
    wav.write_bytes(b"x")
    monkeypatch.setattr(emotion2vec, "read_wav16k", lambda _p: np.zeros(16000 * 70, np.float32))
    t = Emotion2vecTagger({}, "cpu")
    t._model = object()
    calls = []

    def fake(samples):
        calls.append(len(samples))
        return scores(angry=0.9) if len(calls) == 1 else scores(other=0.9)

    monkeypatch.setattr(t, "_scores", fake)
    turns = [
        Turn(start=0.0, end=3.0, speaker="S0"),
        Turn(start=5.0, end=5.1, speaker="S1"),  # under MIN_TURN_S: skipped
        Turn(start=6.0, end=9.0, speaker="S1"),  # other wins: no span
    ]
    tags = t.tag(wav, turns)
    assert tags.backend == "emotion2vec"
    assert tags.events == []
    assert [(s.start, s.end, s.label, s.confidence) for s in tags.emotions] == [
        (0.0, 3.0, "angry", 0.9)
    ]
    assert len(calls) == 2


def test_tag_before_load_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="before load"):
        Emotion2vecTagger({}, "cpu").tag(tmp_path / "x.wav", [])


class FakeHub:
    """A hub with a cache: a download fills it, local_files_only reads only it."""

    def __init__(self) -> None:
        self.cached = False
        self.calls: list[bool] = []

    def snapshot_download(self, repo, revision=None, local_files_only=False, **_kw):
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.calls.append(local_files_only)
        if local_files_only:
            if not self.cached:
                raise LocalEntryNotFoundError("not cached")
        else:
            self.cached = True
        return f"/cache/{repo}/{revision}"


@pytest.fixture
def hub(monkeypatch):
    import sys
    import types

    import huggingface_hub

    h = FakeHub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
    fake = types.ModuleType("funasr")
    fake.AutoModel = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", fake)
    return h


def test_runtime_loads_never_download(hub):
    # ruling CD24: a cache miss names the setup command; only fetch downloads
    from fukikae.backends.loader import BackendUnavailable

    with pytest.raises(BackendUnavailable, match="fukikae backends setup emotion2vec"):
        Emotion2vecTagger({}, "cpu").load()
    assert hub.calls == [True]
    Emotion2vecTagger.fetch()
    assert hub.calls == [True, False]
    hub.calls.clear()
    t = Emotion2vecTagger({}, "cpu")
    t.load()
    assert hub.calls == [True]
    assert t._model["model"] == f"/cache/{DEFAULT_MODEL}/{DEFAULT_REVISION}"


@pytest.mark.slow
@pytest.mark.media
def test_emotion2vec_on_fixture_clip(tmp_path: Path):
    import wave

    clip = Path(
        "~/fukikae-ws/fairytail/.fukikae/episodes/S01E11/01-separate/vocals_16k.wav"
    )
    if not clip.is_file():
        pytest.skip("no Fairy Tail E11 vocals on this machine")
    with wave.open(str(clip)) as w:
        w.setpos(int(60 * 16000))
        raw = w.readframes(40 * 16000)
    short = tmp_path / "clip.wav"
    with wave.open(str(short), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(16000)
        o.writeframes(raw)
    t = Emotion2vecTagger({}, "cpu")
    t.load()
    from huggingface_hub import snapshot_download

    pinned = snapshot_download(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    assert Path(t._model.kwargs["model_path"]) == Path(pinned)
    assert Path(pinned).name == DEFAULT_REVISION
    try:
        tags = t.tag(short, [Turn(start=19.07, end=22.75, speaker="S0")])
    finally:
        t.unload()
    allowed = {"neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"}
    assert len(tags.emotions) == 1
    span = tags.emotions[0]
    assert span.label in allowed
    assert span.confidence is not None and 0.0 < span.confidence <= 1.0
    assert tags.events == []
