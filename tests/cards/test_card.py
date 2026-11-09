import json
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest

from fukikae.cards.card import (
    CARD_SOURCES,
    Card,
    CardError,
    Fingerprint,
    Reference,
    Voice,
    _card_dict,
    _fingerprint_dict,
    _reference_dict,
    _voice_dict,
    canonical_form,
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
    speech_cps: ClassVar[dict[str, float]] = {}
    model_revision: str = "1.0"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        if "steps" in options:
            raise CardError(f"fake: unknown option 'steps' (source {source})")

    @classmethod
    def capabilities_for(cls, source: str) -> frozenset[str]:  # noqa: ARG003
        return cls.capabilities

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
        spec = parse_card(SPEC_CARD)
        validate_for_backend(replace(spec, voice=replace(spec.voice, backend="fake")), Fake)
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


def test_validate_for_backend_refuses_a_card_of_another_backend():
    kokoro = _stock(backend="kokoro")
    with pytest.raises(CardError, match=r"'kokoro'.*'fake'"):
        validate_for_backend(kokoro, Fake)


def test_eq_error_lists_presets_sorted():
    with pytest.raises(CardError, match=r"use one of none$"):
        validate_for_backend(_stock(post={"eq": "bright"}), Fake)


# --- CD1: every field of an id-bearing dataclass reaches the canonical dict ----------------

FULL_REFERENCE = Reference(
    clip=CLIP, transcript="t", language="ja", origin={"episode": "S01E11", "start": 1.5}
)
FULL_FINGERPRINT = Fingerprint(embedder="wespeaker:resnet34", vector=VECTOR, threshold=0.85)
FULL_VOICE = Voice(
    backend="b",
    model_revision="r",
    source="clone",
    seed=7,
    temperature=0.5,
    top_p=0.5,
    rate=1.5,
    pitch_semitones=1.0,
    emotion_strength=0.5,
    voice="v",
    voices=(("v", 1.0),),
    references=(FULL_REFERENCE,),
    latent="sha256:" + "b" * 64,
    description="d",
    blend={"with": "v", "weight": 0.5},
    options={"steps": 3},
    post={"gain_db": 1.0},
    fingerprint=FULL_FINGERPRINT,
)


def test_every_voice_field_is_in_the_canonical_dict():
    """Set every field off its default; each must appear. A new Voice field fails here until
    _voice_dict handles it (and this test sets it), so it can't drop out of the id and file."""
    for f in fields(Voice):
        assert getattr(FULL_VOICE, f.name) != getattr(Voice("", "", ""), f.name), f.name
    assert set(_voice_dict(FULL_VOICE, fingerprint=True)) == {f.name for f in fields(Voice)}
    assert set(_voice_dict(FULL_VOICE, fingerprint=False)) == {f.name for f in fields(Voice)} - {
        "fingerprint"
    }


def test_every_reference_fingerprint_and_card_field_is_dumped():
    assert set(_reference_dict(FULL_REFERENCE)) == {f.name for f in fields(Reference)}
    assert set(_fingerprint_dict(FULL_FINGERPRINT)) == {f.name for f in fields(Fingerprint)}
    card = Card(target_language="en", voice=FULL_VOICE, style={"notes": "n"}, meta={"title": "x"})
    assert set(_card_dict(card, identity=False)) == {f.name for f in fields(Card)}
    assert set(_card_dict(card, identity=True)) == {f.name for f in fields(Card)} - {"meta"}


def test_adding_a_defaulted_field_would_not_change_existing_ids():
    # Fields at their default are left out, so the canonical form of a minimal card names
    # only what it sets.
    assert set(json.loads(canonical_form(parse_card(STOCK_CARD)))["voice"]) == {
        "backend",
        "model_revision",
        "source",
        "voice",
    }


# --- normalization -------------------------------------------------------------------------


def test_empty_style_is_no_style():
    plain = parse_card(STOCK_CARD)
    empty = parse_card(STOCK_CARD + "\n[style]\n")
    assert empty.style is None
    assert card_id(empty) == card_id(plain)
    assert card_id(replace(plain, style={})) == card_id(plain)


def test_option_numbers_are_normalized():
    a = parse_card(STOCK_CARD + "\n[voice.options]\nsteps = 3\n")
    b = parse_card(STOCK_CARD + "\n[voice.options]\nsteps = 3.0\n")
    assert a == b and card_id(a) == card_id(b)
    code_built = _stock(options={"steps": 3.0})
    assert card_id(code_built) == card_id(a)
    assert card_id(_stock(rate=2)) == card_id(_stock(rate=2.0))


def test_text_is_nfc_normalized():
    composed = "\u30ac"  # ガ
    decomposed = "\u30ab\u3099"  # カ + combining voiced mark
    a = parse_card(SPEC_CARD.replace("なんだよ", composed))
    b = parse_card(SPEC_CARD.replace("なんだよ", decomposed))
    assert a.voice.references[0].transcript.startswith(composed)
    assert card_id(a) == card_id(b)
    sa = parse_card(SPEC_CARD.replace("燃えてきたぞ", composed))
    sb = parse_card(SPEC_CARD.replace("燃えてきたぞ", decomposed))
    assert card_id(sa) == card_id(sb)
    da = _stock(description="caf\u00e9")
    db = _stock(description="cafe\u0301")
    assert card_id(da) == card_id(db)


# --- ranges --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("rate = 0", "rate"),
        ("rate = -1.0", "rate"),
        ("temperature = 0.0", "temperature"),
        ("top_p = 0.0", "top_p"),
        ("top_p = 1.5", "top_p"),
        ("seed = -1", "seed"),
        ("emotion_strength = -0.1", "emotion_strength"),
        ("emotion_strength = 1.5", "emotion_strength"),
        ("rate = nan", "rate"),
        ("pitch_semitones = inf", "pitch_semitones"),
        ("temperature = -inf", "temperature"),
    ],
)
def test_voice_numbers_out_of_range_are_refused(line: str, key: str):
    with pytest.raises(CardError, match=key):
        parse_card(STOCK_CARD + line + "\n")


def test_top_p_one_and_emotion_bounds_are_accepted():
    parse_card(STOCK_CARD + "top_p = 1.0\nemotion_strength = 0.0\nseed = 0\n")


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ('\n[voice.blend]\nwith = "af_bella"\nweight = -0.1\n', "weight"),
        ('\n[voice.blend]\nwith = "af_bella"\nweight = nan\n', "weight"),
        ("\n[voice.post]\ngain_db = inf\n", "gain_db"),
        ("\n[voice.options]\nsteps = nan\n", "options"),
    ],
)
def test_table_numbers_out_of_range_are_refused(extra: str, key: str):
    with pytest.raises(CardError, match=key):
        parse_card(STOCK_CARD + extra)


def test_blend_voice_weights_must_be_finite_and_non_negative():
    base = STOCK_CARD.replace('source = "stock"', 'source = "blend"')
    for w in ("-0.5", "inf"):
        with pytest.raises(CardError, match="weight"):
            parse_card(
                base.replace('voice = "af_heart"', f'voices = [{{ id = "a", weight = {w} }}]')
            )
    with pytest.raises(CardError, match="threshold"):
        parse_card(SPEC_CARD.replace("threshold = 0.85", "threshold = nan"))


# --- meta ----------------------------------------------------------------------------------


def test_unknown_meta_keys_are_kept_and_outside_the_id():
    c = parse_card(SPEC_CARD.replace('author = "lokkju"', 'author = "lokkju"\nrating = 5'))
    assert c.meta["rating"] == 5
    assert "rating = 5" in dump_card(c)
    assert parse_card(dump_card(c)) == c
    assert card_id(c) == card_id(parse_card(SPEC_CARD))


def test_known_meta_keys_are_type_checked():
    with pytest.raises(CardError, match="title"):
        parse_card(SPEC_CARD.replace('title = "Natsu', 'title = 3\nx = "Natsu'))
    with pytest.raises(CardError, match="created"):
        parse_card(SPEC_CARD.replace("created = 2026-11-08", "created = 3"))


# --- authored option values are kept exactly; only the id form unifies 1 and 1.0 -----------

_OPTS_FLOAT = "\n[voice.options]\ncfg = 1.0\nlist = [1.0, 2]\n[voice.options.inner]\nk = 1.0\n"
_OPTS_INT = "\n[voice.options]\ncfg = 1\nlist = [1, 2.0]\n[voice.options.inner]\nk = 1\n"


class Recorder(Fake):
    seen: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        cls.seen.append({**options, "__source__": source} if source is None else options)


def _types(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_types(v) for v in value]
    return type(value)


def test_option_values_keep_their_authored_type():
    f = parse_card(STOCK_CARD + _OPTS_FLOAT)
    i = parse_card(STOCK_CARD + _OPTS_INT)
    assert _types(f.voice.options) == {"cfg": float, "list": [float, int], "inner": {"k": float}}
    assert _types(i.voice.options) == {"cfg": int, "list": [int, float], "inner": {"k": int}}
    assert "cfg = 1.0" in dump_card(f) and "cfg = 1\n" in dump_card(i)
    assert parse_card(dump_card(f)) == f and _types(parse_card(dump_card(f)).voice.options) == (
        _types(f.voice.options)
    )
    assert _types(parse_card(dump_card(i)).voice.options) == _types(i.voice.options)
    Recorder.seen.clear()
    validate_for_backend(f, Recorder)
    validate_for_backend(i, Recorder)
    assert _types(Recorder.seen[0]) == _types(f.voice.options)
    assert _types(Recorder.seen[1]) == _types(i.voice.options)
    assert card_id(f) == card_id(i)


def test_origin_keeps_its_authored_type_and_shares_an_id():
    a = parse_card(SPEC_CARD.replace("start = 512.40", "start = 512.0"))
    b = parse_card(SPEC_CARD.replace("start = 512.40", "start = 512"))
    assert type(a.voice.references[0].origin["start"]) is float  # type: ignore[index]
    assert type(b.voice.references[0].origin["start"]) is int  # type: ignore[index]
    assert "start = 512.0" in dump_card(a)
    assert card_id(a) == card_id(b)


def _styled(style: str) -> str:
    return STOCK_CARD + "\n[style]\n" + style


@pytest.mark.parametrize(
    ("style", "message"),
    [
        (f'register = "{"r" * 301}"\n', "register.*300"),
        (f'address = "{"a" * 301}"\n', "address.*300"),
        (f'tics = "{"t" * 301}"\n', "tics.*300"),
        (f'notes = "{"n" * 1001}"\n', "notes.*1000"),
        (f'mood = "{"m" * 301}"\n', "mood.*300"),
        (
            "catchphrases = ["
            + ", ".join(f'{{ source = "s{i}", target = "t{i}" }}' for i in range(21))
            + "]\n",
            "20",
        ),
        (f'catchphrases = [ {{ source = "{"s" * 81}", target = "t" }} ]\n', "source.*80"),
        (f'catchphrases = [ {{ source = "s", target = "{"t" * 81}" }} ]\n', "target.*80"),
    ],
)
def test_style_fields_are_capped_at_load(style, message):
    with pytest.raises(CardError, match=message):
        parse_card(_styled(style))


def test_style_at_the_caps_loads_and_keeps_its_id():
    text = _styled(
        f'register = "{"r" * 300}"\nnotes = "{"n" * 1000}"\ncatchphrases = ['
        + ", ".join(f'{{ source = "{"s" * 80}", target = "{i}" }}' for i in range(20))
        + "]\n"
    )
    card = parse_card(text)
    assert card_id(parse_card(dump_card(card))) == card_id(card)


def test_caps_leave_existing_card_ids_alone():
    # ids computed before the style caps existed (CD1): a card within the caps keeps its id
    assert card_id(parse_card(SPEC_CARD)) == "31a12ee8813d8ba9"
    styled = _styled('register = "rough"\ncatchphrases = [ { source = "a", target = "" } ]\n')
    assert card_id(parse_card(styled)) == "6f04a42dacae1eca"
