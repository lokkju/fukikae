"""IndexTTS2's declarations, offline (the model runs in its worker; see test_tts_worker)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from fukikae.backends import loader
from fukikae.backends.tts import impl_indextts2 as impl
from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
from fukikae.cards.card import Card, CardError, Reference, Voice, validate_for_backend

ROOT = Path(__file__).resolve().parents[2]


def _card(**voice: object) -> Card:
    ref = Reference(clip="sha256:" + "c" * 64, transcript="なんだよ", language="ja")
    v = Voice(
        backend="indextts2",
        model_revision=IndexTts2Synthesizer.model_revision,
        source="clone",
        seed=1234,
        references=(ref,),
    )
    return Card(target_language="en", voice=replace(v, **voice))  # type: ignore[arg-type]


def test_declares_clone_with_a_separate_emotion_reference():
    c = IndexTts2Synthesizer
    assert c.name == "indextts2" and c.extra == "indextts2"
    assert c.sources == frozenset({"clone"})
    assert c.capabilities == frozenset({"emotion_audio"})
    assert c.voice_keys == frozenset({"seed", "temperature", "top_p", "emotion_strength"})
    assert c.sample_rate == 22050 and "en" in c.languages
    assert impl.REVISION in c.model_revision and impl.CODE_REVISION[:12] in c.model_revision
    assert c.speech_cps["en"] > 0


def test_a_clone_card_validates_and_unsupported_keys_are_refused():
    validate_for_backend(_card(), IndexTts2Synthesizer)
    validate_for_backend(_card(emotion_strength=0.5, temperature=0.7), IndexTts2Synthesizer)
    with pytest.raises(CardError, match="rate"):
        validate_for_backend(_card(rate=1.2), IndexTts2Synthesizer)
    with pytest.raises(CardError, match="voice.options"):
        validate_for_backend(_card(options={"steps": 10}), IndexTts2Synthesizer)


def test_registered_as_a_tts_backend():
    assert loader.backend_class("tts", "indextts2") is IndexTts2Synthesizer


def test_unknown_stage_option_is_refused():
    from fukikae.config import ConfigError

    IndexTts2Synthesizer({"fp16": True}, "cpu")
    with pytest.raises(ConfigError, match="fp16"):
        IndexTts2Synthesizer({"steps": 3}, "cpu")


def test_impl_module_level_is_stdlib_only():
    """fukikae's environment imports it for the pins; the worker's runs it."""
    code = (
        "import sys; import fukikae.backends.tts.impl_indextts2; "
        "print(sorted(m for m in ('numpy', 'torch', 'indextts', 'librosa') if m in sys.modules))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip
    assert r.stdout.strip() == "[]"


def test_worker_project_pins_the_code_revision():
    project = tomllib.loads((ROOT / "workers" / "indextts2" / "pyproject.toml").read_text())
    source = project["tool"]["uv"]["sources"]["indextts"]
    assert source["rev"] == impl.CODE_REVISION
    assert (ROOT / "workers" / "indextts2" / "uv.lock").is_file()


# ---- fast inference (D26) -----------------------------------------------------------------


def test_fast_inference_options_are_stage_options():
    assert IndexTts2Synthesizer.option_keys == frozenset({"fp16", "tf32"})
    IndexTts2Synthesizer({"fp16": True, "tf32": True}, "cpu")


class _FakeIndexTTS2:
    seen: dict = {}

    def __init__(self, **kw):
        _FakeIndexTTS2.seen = kw


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    import types

    mod = types.ModuleType("indextts.infer_v2")
    mod.IndexTTS2 = _FakeIndexTTS2  # type: ignore[attr-defined]
    pkg = types.ModuleType("indextts")
    monkeypatch.setitem(sys.modules, "indextts", pkg)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", mod)
    monkeypatch.setattr(impl, "installed_code_revision", lambda: impl.CODE_REVISION)
    monkeypatch.setattr(impl, "_snapshot", lambda *_a: str(tmp_path))
    tf32: list[bool] = []
    monkeypatch.setattr(impl, "_set_tf32", tf32.append)
    _FakeIndexTTS2.seen = {}
    return tf32


def test_load_passes_the_fast_inference_options(fake_model):
    i = impl.Impl()
    i.load({"fp16": True, "tf32": True}, "cuda:0")
    seen = _FakeIndexTTS2.seen
    assert seen["use_fp16"] is True and fake_model == [True]
    # the engines that failed the gate or the env stay off (impl_indextts2's docstring)
    assert seen["use_accel"] is False and seen["use_torch_compile"] is False
    assert seen["use_deepspeed"] is False
    i.unload()


def test_load_defaults_keep_the_reference_inference(fake_model):
    i = impl.Impl()
    i.load({}, "cuda:0")
    assert _FakeIndexTTS2.seen["use_fp16"] is False
    assert fake_model == []  # torch's defaults untouched
    i.unload()


def test_fast_inference_is_off_on_the_cpu(fake_model):
    i = impl.Impl()
    i.load({"fp16": True, "tf32": True}, "cpu")
    assert _FakeIndexTTS2.seen["use_fp16"] is False and fake_model == []
    i.unload()


@pytest.mark.parametrize("profile", ["default", "anime", "live-action"])
def test_shipped_profiles_turn_tf32_on(tmp_path, profile):
    from fukikae.config import load_config
    from fukikae.stages.tts import backend_config

    (tmp_path / "fukikae.toml").write_text(f'profile = "{profile}"\n')
    cfg = load_config(tmp_path, user_config=None)
    assert backend_config(cfg, "indextts2").options.get("tf32") is True
