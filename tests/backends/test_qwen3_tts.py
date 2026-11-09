"""Qwen3-TTS: the declarations fukikae reads, the worker impl's pure helpers, delivery through
the stage, and (gpu) a real render through the worker."""

from __future__ import annotations

import dataclasses

import pytest

from fukikae.backends import loader
from fukikae.backends.tts import impl_qwen3_tts as impl
from fukikae.cards.card import Card, CardError, Reference, Voice, validate_for_backend
from fukikae.config import ConfigError
from fukikae.models import Delivery, DubCue
from fukikae.stages import tts


def _cls():
    from fukikae.backends.tts.qwen3_tts import Qwen3TtsSynthesizer

    return Qwen3TtsSynthesizer


def test_qwen3_tts_declares_only_what_was_verified():
    cls = _cls()
    assert loader.backend_class("tts", "qwen3_tts") is cls
    assert (cls.name, cls.extra) == ("qwen3_tts", "qwen3tts")
    assert cls.sources == frozenset({"clone"})
    # ruling CD20: no delivery capability without an A/B showing it works (report, Task 9c)
    assert cls.capabilities == frozenset()
    assert cls.capabilities_for("clone") == frozenset()
    assert cls.voice_keys == frozenset({"seed", "temperature", "top_p", "qwen3_tts_mode"})
    assert cls.sample_rate == 24000
    assert {"en", "ja", "de", "zh"} <= cls.languages
    assert cls.default_device == "cuda"
    assert cls.option_keys == frozenset({"mode"})
    assert cls.model == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    # the model snapshot and the qwen-tts code: either can change a voice
    assert impl.REVISION in cls.model_revision
    assert f"qwen-tts=={impl.CODE_VERSION}" in cls.model_revision
    assert set(cls.speech_cps) == {"en"} and 8.0 < cls.speech_cps["en"] < 25.0
    assert 0 < cls.vram_mb < 24000


def test_an_unknown_mode_is_refused_at_construction():
    cls = _cls()
    assert cls({"mode": "xvector"}, "cpu").options == {"mode": "xvector"}
    with pytest.raises(ConfigError, match="mode"):
        cls({"mode": "sideways"}, "cpu")
    with pytest.raises(ConfigError, match="no option"):
        cls({"fp16": True}, "cpu")


def test_auto_mode_uses_the_transcripts_when_every_clip_has_one():
    assert impl.resolve_mode("auto", ["やあ", "うん"]) == "icl"
    assert impl.resolve_mode("auto", ["やあ", ""]) == "xvector"
    assert impl.resolve_mode("xvector", ["やあ"]) == "xvector"
    with pytest.raises(ValueError, match="transcript"):
        impl.resolve_mode("icl", ["", "x"])
    with pytest.raises(ValueError, match="mode"):
        impl.resolve_mode("both", [])


def test_the_card_mode_beats_the_stage_mode_which_is_only_the_fallback():
    # CD21: a card at the default (auto) takes the stage option when there is one
    assert impl.card_mode("icl", "xvector") == "icl"
    assert impl.card_mode("xvector", "icl") == "xvector"
    assert impl.card_mode("auto", "xvector") == "xvector"
    assert impl.card_mode("auto", None) == "auto"
    assert impl.card_mode(None, None) == "auto"
    with pytest.raises(ValueError, match="mode"):
        impl.card_mode("sideways", None)


def test_the_card_mode_reaches_the_worker_request(tmp_path):
    from fukikae.backends.tts.remote import WorkerSynthesizer

    voice = dataclasses.replace(_clone_card().voice, qwen3_tts_mode="xvector")
    ref = tmp_path / "r.wav"
    ref.write_bytes(b"")
    payload = WorkerSynthesizer.voice_payload(voice, lambda _r: ref)
    assert payload["qwen3_tts_mode"] == "xvector"
    from fukikae.backends.tts import worker

    worker._voice({**payload, "references": []})  # the worker accepts the key


def test_language_codes_map_to_the_model_names():
    assert impl.language_name("en") == "english"
    assert impl.language_name("ja") == "japanese"
    with pytest.raises(ValueError, match="speaks"):
        impl.language_name("tlh")


def test_the_prompt_takes_whole_clips_up_to_the_limit():
    refs = [("a.wav", 8.0, "uno"), ("b.wav", 6.0, "dos"), ("c.wav", 5.0, "tres")]
    clips, text = impl.prompt_plan(refs)
    assert clips == ["a.wav", "b.wav"] and text == "uno dos"  # c would pass 15 s
    assert impl.prompt_plan([("long.wav", 40.0, "x")]) == (["long.wav"], "x")


def test_load_refuses_another_qwen_tts_version(monkeypatch):
    monkeypatch.setattr(impl, "installed_code_version", lambda: "0.0.5")
    with pytest.raises(RuntimeError, match=r"0\.0\.5.*fukikae backends setup qwen3_tts"):
        impl.check_code()
    monkeypatch.setattr(impl, "installed_code_version", lambda: impl.CODE_VERSION)
    impl.check_code()


def test_the_tokenizers_other_control_tokens_are_stripped_by_the_worker():
    from fukikae.backends.tts import worker

    text = "Hi <tts_pad>there<think> now</think> <tool_call>x</tool_call> <tts_text_eod>"
    assert worker.strip_control_markup(text, impl.CONTROL_MARKUP) == "Hi there now x"


def _clone_card() -> Card:
    cls = _cls()
    voice = Voice(
        backend="qwen3_tts",
        model_revision=cls.model_revision,
        source="clone",
        seed=1234,
        references=(Reference(clip="sha256:" + "a" * 64, transcript="やあ", language="ja"),),
    )
    return Card(voice=voice, target_language="en")


def test_a_clone_card_checks_and_rate_is_refused():
    cls = _cls()
    validate_for_backend(_clone_card(), cls)
    card = _clone_card()
    bad = Card(voice=dataclasses.replace(card.voice, rate=1.2), target_language="en")
    with pytest.raises(CardError, match="rate"):
        validate_for_backend(bad, cls)


def test_delivery_passes_nothing():
    cue = DubCue(
        id="c0001", start=3.0, end=5.0, character_id=1, extra=None, speaker="x", source_text="s",
        text="Hello there.", budget=40,
        delivery=Delivery(reference=(3.0, 5.0), direction="angry", emotion="angry",
                          emotion_confidence=0.9),
    )  # fmt: skip
    assert tts.delivery_passed(cue, _clone_card(), _cls(), 2.0) == {}


@pytest.mark.gpu
def test_real_qwen3_tts_renders_a_clone_line(tmp_path, monkeypatch):
    """Needs `.venvs/qwen3tts` (README, "Cloning backends") and GPU 0."""
    import numpy as np
    from scipy.io import wavfile

    monkeypatch.delenv("FUKIKAE_WORKER_CHECKOUT", raising=False)
    cls = _cls()
    ref = tmp_path / "ref.wav"
    t = np.arange(24000 * 4) / 24000
    wavfile.write(ref, 24000, (0.2 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2"))
    synth = cls({}, "cuda:0")
    synth.load()
    try:
        card = _clone_card()
        out = tmp_path / "o.wav"
        seconds = synth.synthesize(
            "Hello there, this is a test.", card.voice, "en", out, resolve=lambda _r: ref
        )
        assert 0.5 < seconds < 10.0
        again = tmp_path / "o2.wav"
        synth.synthesize(
            "Hello there, this is a test.", card.voice, "en", again, resolve=lambda _r: ref
        )
        assert out.read_bytes() == again.read_bytes()  # same seed, same render
    finally:
        synth.unload()
