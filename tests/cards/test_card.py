from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest

from fukikae.cards.card import (
    CARD_SOURCES,
    Card,
    CardError,
    Voice,
    card_id,
    dump_card,
    parse_card,
    sidecars,
    validate_for_backend,
)

CLIP = "sha256:9e1f" + "0" * 60
VECTOR = "sha256:5d2c" + "1" * 60

# The spec's example card, with full sidecar hashes and the per-language catchphrase key `target`.
SPEC_CARD = f"""\
target_language = "en"

[meta]
title = "Natsu, cloned from the original, a bit brighter"
author = "lokkju"
created = 2026-11-08
derived_from = "3f9a0c1d2e4b5a69"

[voice]
backend = "indextts2"
model_revision = "a1b2c3d"
source = "clone"
seed = 1234
temperature = 0.7
top_p = 0.9
rate = 1.0
pitch_semitones = 0.0
emotion_strength = 0.8

[[voice.references]]
clip = "{CLIP}"
transcript = "なんだよ、それ"
language = "ja"
origin = {{ episode = "S01E11", start = 512.40, end = 516.10 }}

[voice.blend]
with = "card:77c1e0aa12f3b4c5"
weight = 0.3

[voice.options]

[voice.post]
gain_db = 0.0
eq = "none"

[voice.fingerprint]
embedder = "wespeaker:resnet34"
vector = "{VECTOR}"
threshold = 0.85

[style]
register = "casual and rough; first person 'ore'; never polite speech"
address = "calls Lucy by name, Gray 'ice-brain'"
tics = "ends boasts with 'got it?'"
catchphrases = [ {{ source = "燃えてきたぞ", target = "I'm all fired up!" }} ]
notes = ""
"""

STOCK_CARD = """\
target_language = "en"

[voice]
backend = "fake"
model_revision = "1.0"
source = "stock"
voice = "af_heart"
"""


def test_spec_example_round_trips():
    c = parse_card(SPEC_CARD)
    assert c.target_language == "en" and c.voice.source == "clone"
    assert c.voice.references[0].transcript == "なんだよ、それ"
    assert c.voice.references[0].origin == {"episode": "S01E11", "start": 512.4, "end": 516.1}
    assert c.voice.fingerprint is not None and c.voice.fingerprint.threshold == 0.85
    assert c.style is not None
    assert c.style["catchphrases"] == [{"source": "燃えてきたぞ", "target": "I'm all fired up!"}]
    assert parse_card(dump_card(c)) == c


def test_stock_and_blend_round_trip():
    stock = parse_card(STOCK_CARD)
    assert stock.voice.voice == "af_heart" and stock.style is None
    assert parse_card(dump_card(stock)) == stock
    blend = parse_card(
        STOCK_CARD.replace('source = "stock"', 'source = "blend"').replace(
            'voice = "af_heart"',
            'voices = [{ id = "af_heart", weight = 0.7 }, { id = "am_adam", weight = 0.3 }]',
        )
    )
    assert blend.voice.voices == (("af_heart", 0.7), ("am_adam", 0.3))
    assert parse_card(dump_card(blend)) == blend


def test_card_id_ignores_meta_and_fingerprint():
    c = parse_card(SPEC_CARD)
    assert card_id(c) == card_id(replace(c, meta={"title": "other"}))
    assert card_id(c) == card_id(replace(c, voice=replace(c.voice, fingerprint=None)))
    assert len(card_id(c)) == 16
    int(card_id(c), 16)


def test_card_id_changes_with_voice_style_and_language():
    c = parse_card(SPEC_CARD)
    assert card_id(c) != card_id(replace(c, target_language="es"))
    assert card_id(c) != card_id(replace(c, voice=replace(c.voice, seed=1)))
    assert card_id(c) != card_id(replace(c, style={"register": "polite"}))
    other_clip = replace(c.voice.references[0], clip="sha256:" + "a" * 64)
    assert card_id(c) != card_id(replace(c, voice=replace(c.voice, references=(other_clip,))))


def test_card_id_ignores_explicit_defaults_and_int_spelling():
    plain = parse_card(STOCK_CARD)
    spelled = parse_card(STOCK_CARD + "rate = 1\nemotion_strength = 0.8\n")
    assert spelled == plain
    assert card_id(spelled) == card_id(plain)


def test_card_id_is_stable_through_dump():
    c = parse_card(SPEC_CARD)
    assert card_id(parse_card(dump_card(c))) == card_id(c)


def test_unknown_key_names_table_and_key():
    with pytest.raises(CardError, match=r"\[voice\].*colour"):
        parse_card(SPEC_CARD.replace("seed = 1234", "seed = 1234\ncolour = 3"))
    with pytest.raises(CardError, match=r"\[voice\.post\].*reverb"):
        parse_card(SPEC_CARD.replace('eq = "none"', 'eq = "none"\nreverb = 1'))
    with pytest.raises(CardError, match=r"top level.*voices_extra"):
        parse_card("voices_extra = 1\n" + SPEC_CARD)
    with pytest.raises(CardError, match=r"\[\[voice\.references\]\].*speaker"):
        parse_card(SPEC_CARD.replace('language = "ja"', 'language = "ja"\nspeaker = "x"'))
    with pytest.raises(CardError, match=r"catchphrases.*english"):
        parse_card(SPEC_CARD.replace("target = ", "english = "))


def test_missing_required_key_is_named():
    with pytest.raises(CardError, match="target_language"):
        parse_card(SPEC_CARD.replace('target_language = "en"', ""))
    with pytest.raises(CardError, match="model_revision"):
        parse_card(SPEC_CARD.replace('model_revision = "a1b2c3d"', ""))
    with pytest.raises(CardError, match=r"\[voice\]"):
        parse_card('target_language = "en"\n')


def test_source_requires_its_field():
    with pytest.raises(CardError, match="voice"):
        parse_card(STOCK_CARD.replace('voice = "af_heart"', ""))
    with pytest.raises(CardError, match="voices"):
        parse_card(STOCK_CARD.replace('source = "stock"', 'source = "blend"'))
    with pytest.raises(CardError, match="references"):
        parse_card(STOCK_CARD.replace('source = "stock"', 'source = "clone"'))
    with pytest.raises(CardError, match="latent"):
        parse_card(STOCK_CARD.replace('source = "stock"', 'source = "latent"'))
    with pytest.raises(CardError, match="description"):
        parse_card(STOCK_CARD.replace('source = "stock"', 'source = "design"'))
    with pytest.raises(CardError, match=r"'x'.*" + ".*".join(CARD_SOURCES)):
        parse_card(STOCK_CARD.replace('source = "stock"', 'source = "x"'))


def test_value_checks():
    with pytest.raises(CardError, match="emotion_strength"):
        parse_card(SPEC_CARD.replace("emotion_strength = 0.8", "emotion_strength = 1.5"))
    with pytest.raises(CardError, match="seed"):
        parse_card(SPEC_CARD.replace("seed = 1234", 'seed = "1234"'))
    with pytest.raises(CardError, match="clip"):
        parse_card(SPEC_CARD.replace(CLIP, "sha256:9e1f..."))
    with pytest.raises(CardError, match="register"):
        parse_card(SPEC_CARD.replace('register = "casual', 'register = 3\nx = "casual'))


def test_sidecars_lists_every_sha256_ref():
    c = parse_card(SPEC_CARD)
    assert sidecars(c) == [CLIP, VECTOR]
    latent = "sha256:" + "b" * 64
    lc = parse_card(
        STOCK_CARD.replace('source = "stock"', 'source = "latent"').replace(
            'voice = "af_heart"', f'latent = "{latent}"'
        )
    )
    assert sidecars(lc) == [latent]
    assert sidecars(parse_card(STOCK_CARD)) == []


class Fake:
    """A minimal but complete Synthesizer: stock voices only, honors only `rate`."""

    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "tts"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar[frozenset[str]] = frozenset({"stock"})
    voice_keys: ClassVar[frozenset[str]] = frozenset({"rate"})
    capabilities: ClassVar[frozenset[str]] = frozenset()
    sample_rate: ClassVar[int] = 24000
    model_revision: str = "1.0"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @classmethod
    def validate_options(cls, options: dict[str, Any]) -> None:
        if "steps" in options:
            raise CardError("fake: unknown option 'steps'")

    def synthesize(
        self,
        text: str,
        voice: Voice,
        language: str,
        out: Path,
        *,
        resolve: Callable[[str], Path],
        emotion_audio: Path | None = None,
        emotion_vector: dict[str, float] | None = None,
        instruction: str | None = None,
        emotion_strength: float = 1.0,
        target_seconds: float | None = None,
    ) -> float:
        raise NotImplementedError


class FakeBlender(Fake):
    voice_keys: ClassVar[frozenset[str]] = frozenset({"rate", "blend"})


def _stock(**voice: Any) -> Card:
    c = parse_card(STOCK_CARD)
    return replace(c, voice=replace(c.voice, **voice))


def test_validate_for_backend_accepts_supported_card():
    validate_for_backend(_stock(rate=1.2), Fake)


def test_validate_for_backend_rejects_unsupported_source_and_key():
    with pytest.raises(CardError, match="fake does not support source 'clone'"):
        validate_for_backend(parse_card(SPEC_CARD), Fake)
    with pytest.raises(CardError, match="temperature"):
        validate_for_backend(_stock(temperature=0.7), Fake)
    with pytest.raises(CardError, match="steps"):
        validate_for_backend(_stock(options={"steps": 3}), Fake)
    with pytest.raises(CardError, match="eq"):
        validate_for_backend(_stock(post={"eq": "bright"}), Fake)


def test_validate_for_backend_refuses_cross_backend_blend():
    other = replace(parse_card(STOCK_CARD), voice=replace(_stock().voice, backend="other"))
    same = parse_card(STOCK_CARD)
    cards = {card_id(other): other, card_id(same): same}
    with pytest.raises(CardError, match="blend"):
        validate_for_backend(
            _stock(blend={"with": f"card:{card_id(other)}", "weight": 0.3}),
            FakeBlender,
            cards.__getitem__,
        )
    validate_for_backend(
        _stock(blend={"with": f"card:{card_id(same)}", "weight": 0.3}),
        FakeBlender,
        cards.__getitem__,
    )
    validate_for_backend(_stock(blend={"with": "af_bella", "weight": 0.3}), FakeBlender)
    with pytest.raises(CardError, match="blend"):
        validate_for_backend(
            _stock(blend={"with": f"card:{card_id(other)}", "weight": 0.3}), FakeBlender
        )


def test_backend_kinds_and_synthesizer_protocol():
    from fukikae.backends.base import KINDS, Synthesizer

    assert {"tag", "translate", "tts"} <= set(KINDS)
    for attr in ("sources", "voice_keys", "capabilities", "sample_rate", "synthesize"):
        assert attr in Synthesizer.__dict__ or attr in Synthesizer.__annotations__
