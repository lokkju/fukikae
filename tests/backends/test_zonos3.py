"""ZONOS3-Mini (spike, #43): the declarations fukikae reads and the worker impl's pure helpers."""

from __future__ import annotations

import re

import pytest

from fukikae.backends import loader
from fukikae.backends.tts import impl_zonos3 as impl
from fukikae.config import ConfigError


def _cls():
    from fukikae.backends.tts.zonos3 import Zonos3Synthesizer

    return Zonos3Synthesizer


def test_zonos3_declares_no_capability():
    cls = _cls()
    assert loader.backend_class("tts", "zonos3") is cls
    assert (cls.name, cls.extra) == ("zonos3", "zonos3")
    assert cls.sources == frozenset({"clone"})  # D41: clones only
    assert cls.capabilities == frozenset()  # CD20: nothing verified
    assert cls.sample_rate == 44100
    assert impl.REVISION in cls.model_revision
    assert impl.CODE_REVISION[:12] in cls.model_revision


def test_options_are_checked():
    with pytest.raises(ConfigError):
        _cls()({"mode": "stock"}, "cpu")
    with pytest.raises(ConfigError):
        _cls()({"dtype": "fp16"}, "cpu")


def test_language_codes():
    assert impl.language_code("en") == "en_us"
    assert impl.language_code("ja") == "ja"
    with pytest.raises(ValueError):
        impl.language_code("xx")


def test_speaking_rate_is_utf8_bytes_over_target():
    assert impl.speaking_rate_for(" hello ", 2.0) == 2.5
    assert impl.speaking_rate_for("はい", 1.0) == 6.0
    with pytest.raises(ValueError):
        impl.speaking_rate_for("x", 0)


def test_controls_scale_and_negative_pole():
    assert impl.controls_for(None, 3.0) is None
    got = impl.controls_for({"mood.arousal": 0.5, "delivery.tempo-": 1.0}, 3.0)
    assert got == {"mood.arousal": 1.5, "delivery.tempo": -3.0}
    with pytest.raises(ValueError):
        impl.controls_for({"angry": 1.0}, 3.0)


def test_prompt_plan_keeps_whole_clips_up_to_the_limit():
    refs = [("a", 20.0), ("b", 9.0), ("c", 5.0)]
    assert impl.prompt_plan(refs) == ["a", "b"]
    assert impl.prompt_plan([("long", 90.0), ("b", 1.0)]) == ["long"]


def test_private_use_markup_is_listed():
    pattern = re.compile("|".join(impl.CONTROL_MARKUP))
    assert pattern.sub("", "abx") == "abx"
