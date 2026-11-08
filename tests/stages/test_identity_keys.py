"""Ruling GT6: a default-model change with no option change changes every stage's key."""

import logging

import pytest

from fukikae.backends import identity
from fukikae.backends.identity import identity_key
from fukikae.config import BackendConfig, Config
from fukikae.stages.asr import AsrStage
from fukikae.stages.diarize import DiarizeStage
from fukikae.stages.language import LanguageStage
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Episode


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "02-language").mkdir(parents=True)
    (work / "02-language" / "language.json").write_text(
        '{"language": "ko", "confidence": 0.9, "source": "detected"}'
    )
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


def keys(monkeypatch, make_key, _unused=None):
    """The key with every backend resolving to model v1, then to v2, options untouched."""
    out = []
    for version in ("v1", "v2"):
        monkeypatch.setattr(
            identity, "model_identity", lambda kind, _cfg, v=version: (f"{kind}-{v}", None)
        )
        out.append(make_key())
    return out


def test_diarize_key_tracks_both_default_models(monkeypatch, ep):
    a, b = keys(
        monkeypatch,
        lambda: DiarizeStage().config_key(Config(raw={"embed": {"backend": "wespeaker"}}), ep),
        None,
    )
    assert a != b
    assert a["diarize"]["identity"] == {"model": "diarize-v1", "revision": None}
    assert a["embed"]["identity"] == {"model": "embed-v1", "revision": None}


def test_separate_key_tracks_the_default_model(monkeypatch, ep):
    a, b = keys(monkeypatch, lambda: SeparateStage().config_key(Config(), ep), None)
    assert a != b


def test_asr_key_tracks_asr_and_align_default_models(monkeypatch, ep):
    a, b = keys(monkeypatch, lambda: AsrStage().config_key(Config(), ep), None)
    assert a != b
    assert a["asr"]["identity"]["model"] == "asr-v1"
    assert a["align"]["identity"]["model"] == "align-v1"


def test_language_key_tracks_the_default_lid_model(monkeypatch, ep):
    a, b = keys(monkeypatch, lambda: LanguageStage().config_key(Config(), ep), None)
    assert a != b
    # A pinned source language runs no backend, so no model matters.
    pinned = Config(source_language="ko")
    c, d = keys(monkeypatch, lambda: LanguageStage().config_key(pinned, ep), None)
    assert c == d


def test_identity_falls_back_to_the_class_default_and_is_deterministic(monkeypatch):
    monkeypatch.setattr(identity, "model_identity", lambda _k, _c: (None, None))
    cfg = BackendConfig("wespeaker", {})
    first = identity_key("embed", cfg)
    assert first == identity_key("embed", cfg)
    assert first["model"] == "pyannote/wespeaker-voxceleb-resnet34-LM"
    sherpa = identity_key("embed", BackendConfig("campplus", {}))
    assert sherpa["model"].endswith(".onnx")


def test_unknown_identity_is_distinct_stable_and_warned(monkeypatch, caplog):
    monkeypatch.setattr(identity, "_warned", set())
    cfg = BackendConfig("no-such-backend", {})
    with caplog.at_level(logging.WARNING):
        first = identity_key("embed", cfg)
    assert first == identity_key("embed", cfg)
    assert first["model"] == "unresolved:embed:no-such-backend"
    assert "no-such-backend" in caplog.text
    assert first != identity_key("embed", BackendConfig("other", {}))


def test_identity_uses_the_backend_instances_pinned_revision(monkeypatch):
    class Pinned:
        model = "org/m"
        revision = "abc123"

        def __init__(self, options, device):
            pass

    from fukikae.backends import loader

    monkeypatch.setattr(loader, "backend_class", lambda _k, _n: Pinned)
    got = identity_key("tag", BackendConfig("x", {}))
    assert got == {"model": "org/m", "revision": "abc123"}
