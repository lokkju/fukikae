"""Character cards: one TOML file per (character voice, target language), content-addressed.

A card's id is the first 16 hex digits of the SHA-256 of its canonical form: the card as a
dict, without ``[meta]`` and ``[voice.fingerprint]``, serialized with
``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)``. Fields equal to
their default are left out of that dict (and out of the TOML), so spelling a default
explicitly never changes the id. Sidecar files are referenced as ``sha256:<hex>`` of their
bytes, so their hashes are part of the canonical form already.

Defaults are frozen forever (ruling CD1). Because a field at its default is not in the
canonical form, changing a default would silently change what every existing card id means.
A new behaviour gets a new field name with a default that reproduces the old behaviour;
adding such a field leaves every existing id unchanged. Never change a default's value.

Before hashing and storing, every text field is NFC-normalized, typed float fields are floats,
and integral floats in free-form tables (``[voice.options]``, reference ``origin``) become
ints, so ``1`` and ``1.0`` give one id. NaN and infinity are refused everywhere. An empty
``[style]`` is the same card as no ``[style]``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import tomli_w

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer

CARD_SOURCES = ("stock", "blend", "clone", "latent", "design")

# The field each source needs (a non-empty value of it).
SOURCE_FIELDS = {
    "stock": "voice",
    "blend": "voices",
    "clone": "references",
    "latent": "latent",
    "design": "description",
}

# Named [voice.post] eq presets; only "none" exists so far.
EQ_PRESETS = frozenset({"none"})

_SHA_REF = re.compile(r"sha256:[0-9a-f]{64}")
_CARD_ID = re.compile(r"[0-9a-f]{16}")

_TOP_KEYS = frozenset({"target_language", "meta", "voice", "style"})
# Known [meta] keys are type-checked; unknown ones are kept and written back, since [meta] is
# outside the id and a newer fukikae may add keys an older one must not reject.
_META_STR_KEYS = frozenset({"title", "author", "derived_from", "notes"})
_VOICE_REQUIRED = ("backend", "model_revision", "source")
_VOICE_KEYS = frozenset(
    {
        *_VOICE_REQUIRED,
        "seed",
        "temperature",
        "top_p",
        "rate",
        "pitch_semitones",
        "emotion_strength",
        "voice",
        "voices",
        "references",
        "latent",
        "description",
        "blend",
        "options",
        "post",
        "fingerprint",
    }
)
_REFERENCE_KEYS = frozenset({"clip", "transcript", "language", "origin"})
_BLEND_VOICE_KEYS = frozenset({"id", "weight"})
_BLEND_KEYS = frozenset({"with", "weight"})
_POST_KEYS = frozenset({"gain_db", "eq"})
_FINGERPRINT_KEYS = frozenset({"embedder", "vector", "threshold"})
_CATCHPHRASE_KEYS = frozenset({"source", "target"})

# [voice] keys a backend must list in ``voice_keys`` to accept a non-default value. The
# source's own field is covered by ``sources``; options are the backend's validate_options;
# post and fingerprint are applied by fukikae, not the backend.
_BACKEND_VOICE_KEYS = (
    "seed",
    "temperature",
    "top_p",
    "rate",
    "pitch_semitones",
    "emotion_strength",
    "voice",
    "voices",
    "references",
    "latent",
    "description",
    "blend",
)


class CardError(Exception):
    """A card that can't be parsed, stored, found or honored by its backend."""


@dataclass(frozen=True)
class Reference:
    clip: str
    transcript: str
    language: str
    origin: dict[str, Any] | None = None


@dataclass(frozen=True)
class Fingerprint:
    embedder: str
    vector: str
    threshold: float


@dataclass(frozen=True)
class Voice:
    backend: str
    model_revision: str
    source: str
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    rate: float = 1.0
    pitch_semitones: float = 0.0
    emotion_strength: float = 0.8
    voice: str | None = None
    voices: tuple[tuple[str, float], ...] = ()
    references: tuple[Reference, ...] = ()
    latent: str | None = None
    description: str | None = None
    blend: dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    post: dict[str, Any] = field(default_factory=dict)
    fingerprint: Fingerprint | None = None


@dataclass(frozen=True)
class Card:
    target_language: str
    voice: Voice
    style: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_VOICE_DEFAULTS = Voice(backend="", model_revision="", source="")
_FLOAT_VOICE_KEYS = ("temperature", "top_p", "rate", "pitch_semitones", "emotion_strength")


# --- parsing -------------------------------------------------------------------------------


def _check_keys(table: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise CardError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CardError(f"{where} must be a table, got {type(value).__name__}")
    return value


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _str(value: Any, key: str, where: str) -> str:
    if not isinstance(value, str):
        raise CardError(f"{where} {key} must be a string, got {type(value).__name__}")
    return _nfc(value)


def _float(
    value: Any,
    key: str,
    where: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CardError(f"{where} {key} must be a number, got {type(value).__name__}")
    f = float(value)
    if not math.isfinite(f):
        raise CardError(f"{where} {key} must be finite, got {value}")
    if lo is not None and (f <= lo if lo_open else f < lo):
        raise CardError(f"{where} {key} must be {'>' if lo_open else '>='} {lo:g}, got {value}")
    if hi is not None and f > hi:
        raise CardError(f"{where} {key} must be <= {hi:g}, got {value}")
    return f


def _norm_free(value: Any, where: str) -> Any:
    """Normalize a free-form value for storage and hashing: NFC text, integral floats as
    ints, finite numbers only, recursively through tables and arrays."""
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CardError(f"{where} must hold finite numbers, got {value}")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {_nfc(k): _norm_free(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_norm_free(v, where) for v in value]
    return value


def _int(value: Any, key: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CardError(f"{where} {key} must be an integer, got {type(value).__name__}")
    return value


def _sha_ref(value: Any, key: str, where: str) -> str:
    ref = _str(value, key, where)
    if not _SHA_REF.fullmatch(ref):
        raise CardError(f"{where} {key} must be 'sha256:<64 lowercase hex>', got {ref!r}")
    return ref


def _require(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise CardError(f"missing required key {key!r} in {where}")
    return table[key]


def _parse_reference(raw: Any) -> Reference:
    where = "[[voice.references]]"
    t = _table(raw, where)
    _check_keys(t, _REFERENCE_KEYS, where)
    origin = t.get("origin")
    if origin is not None:
        origin = _norm_free(dict(_table(origin, f"{where} origin")), f"{where} origin")
    return Reference(
        clip=_sha_ref(_require(t, "clip", where), "clip", where),
        transcript=_str(_require(t, "transcript", where), "transcript", where),
        language=_str(_require(t, "language", where), "language", where),
        origin=origin,
    )


def _parse_blend_voice(raw: Any) -> tuple[str, float]:
    where = "[voice] voices entry"
    t = _table(raw, where)
    _check_keys(t, _BLEND_VOICE_KEYS, where)
    return (
        _str(_require(t, "id", where), "id", where),
        _float(_require(t, "weight", where), "weight", where, lo=0.0),
    )


def _parse_blend(raw: Any) -> dict[str, Any]:
    where = "[voice.blend]"
    t = _table(raw, where)
    _check_keys(t, _BLEND_KEYS, where)
    return {
        "with": _str(_require(t, "with", where), "with", where),
        "weight": _float(_require(t, "weight", where), "weight", where, lo=0.0),
    }


def _parse_post(raw: Any) -> dict[str, Any]:
    where = "[voice.post]"
    t = _table(raw, where)
    _check_keys(t, _POST_KEYS, where)
    post: dict[str, Any] = {}
    if "gain_db" in t:
        post["gain_db"] = _float(t["gain_db"], "gain_db", where)
    if "eq" in t:
        post["eq"] = _str(t["eq"], "eq", where)
    return post


def _parse_fingerprint(raw: Any) -> Fingerprint:
    where = "[voice.fingerprint]"
    t = _table(raw, where)
    _check_keys(t, _FINGERPRINT_KEYS, where)
    return Fingerprint(
        embedder=_str(_require(t, "embedder", where), "embedder", where),
        vector=_sha_ref(_require(t, "vector", where), "vector", where),
        threshold=_float(_require(t, "threshold", where), "threshold", where),
    )


def _parse_voice(raw: Any) -> Voice:
    where = "[voice]"
    t = _table(raw, where)
    _check_keys(t, _VOICE_KEYS, where)
    for key in _VOICE_REQUIRED:
        _str(_require(t, key, where), key, where)
    source = t["source"]
    if source not in CARD_SOURCES:
        raise CardError(f"unknown [voice] source {source!r}; use one of {', '.join(CARD_SOURCES)}")

    kw: dict[str, Any] = {k: t[k] for k in _VOICE_REQUIRED}
    if "seed" in t:
        kw["seed"] = _int(t["seed"], "seed", where)
        if kw["seed"] < 0:
            raise CardError(f"[voice] seed must be >= 0, got {kw['seed']}")
    ranges: dict[str, dict[str, Any]] = {
        "temperature": {"lo": 0.0, "lo_open": True},
        "top_p": {"lo": 0.0, "lo_open": True, "hi": 1.0},
        "rate": {"lo": 0.0, "lo_open": True},
        "pitch_semitones": {},
        "emotion_strength": {"lo": 0.0, "hi": 1.0},
    }
    for key, bounds in ranges.items():
        if key in t:
            kw[key] = _float(t[key], key, where, **bounds)
    for key in ("voice", "description"):
        if key in t:
            kw[key] = _str(t[key], key, where)
    if "latent" in t:
        kw["latent"] = _sha_ref(t["latent"], "latent", where)
    if "voices" in t:
        if not isinstance(t["voices"], list):
            raise CardError("[voice] voices must be an array of { id, weight } tables")
        kw["voices"] = tuple(_parse_blend_voice(v) for v in t["voices"])
    if "references" in t:
        if not isinstance(t["references"], list):
            raise CardError("[voice] references must be an array of tables")
        kw["references"] = tuple(_parse_reference(r) for r in t["references"])
    if "blend" in t:
        kw["blend"] = _parse_blend(t["blend"])
    if "options" in t:
        kw["options"] = _norm_free(dict(_table(t["options"], "[voice.options]")), "[voice.options]")
    if "post" in t:
        kw["post"] = _parse_post(t["post"])
    if "fingerprint" in t:
        kw["fingerprint"] = _parse_fingerprint(t["fingerprint"])

    needed = SOURCE_FIELDS[source]
    if not kw.get(needed):
        raise CardError(f"[voice] source = {source!r} requires a non-empty {needed!r}")
    return Voice(**kw)


def _parse_style(raw: Any) -> dict[str, Any]:
    where = "[style]"
    t = _table(raw, where)
    style: dict[str, Any] = {}
    for key, value in t.items():
        if key == "catchphrases":
            if not isinstance(value, list):
                raise CardError("[style] catchphrases must be an array of { source, target }")
            phrases = []
            for entry in value:
                e = _table(entry, "[style] catchphrases entry")
                _check_keys(e, _CATCHPHRASE_KEYS, "[style] catchphrases entry")
                phrases.append(
                    {
                        k: _str(_require(e, k, "[style] catchphrases entry"), k, where)
                        for k in ("source", "target")
                    }
                )
            style[key] = phrases
        else:
            style[key] = _str(value, key, where)
    return style


def parse_card(text: str) -> Card:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise CardError(f"card is not valid TOML: {e}") from e
    _check_keys(raw, _TOP_KEYS, "the top level")
    target = _str(_require(raw, "target_language", "the top level"), "target_language", "card")
    if not target:
        raise CardError("target_language must not be empty")
    voice = _parse_voice(_require(raw, "voice", "the card ([voice])"))
    style = (_parse_style(raw["style"]) if "style" in raw else None) or None
    meta: dict[str, Any] = {}
    if "meta" in raw:
        meta = dict(_table(raw["meta"], "[meta]"))
        for key in _META_STR_KEYS & set(meta):
            meta[key] = _str(meta[key], key, "[meta]")
        created = meta.get("created")
        if created is not None and not isinstance(created, str | dt.date):
            raise CardError(
                f"[meta] created must be a date or a string, got {type(created).__name__}"
            )
    return Card(target_language=target, voice=voice, style=style, meta=meta)


# --- serialization -------------------------------------------------------------------------


def _voice_dict(v: Voice, *, fingerprint: bool) -> dict[str, Any]:
    """Every Voice field is handled here (tests/cards/test_card.py checks it), each left out
    while it equals its frozen default (CD1)."""
    d: dict[str, Any] = {
        "backend": _nfc(v.backend),
        "model_revision": _nfc(v.model_revision),
        "source": v.source,
    }
    for key in ("seed", "voice", "latent", "description"):
        value = getattr(v, key)
        if value != getattr(_VOICE_DEFAULTS, key):
            d[key] = _nfc(value) if isinstance(value, str) else value
    for key in _FLOAT_VOICE_KEYS:
        value = getattr(v, key)
        if value != getattr(_VOICE_DEFAULTS, key):
            d[key] = float(value)
    for key in ("blend", "options", "post"):
        value = getattr(v, key)
        if value != getattr(_VOICE_DEFAULTS, key):
            d[key] = _norm_free(value, f"[voice.{key}]")
    if "blend" in d:
        d["blend"]["weight"] = float(d["blend"]["weight"])
    if "post" in d and "gain_db" in d["post"]:
        d["post"]["gain_db"] = float(d["post"]["gain_db"])
    if v.voices:
        d["voices"] = [{"id": _nfc(i), "weight": float(w)} for i, w in v.voices]
    if v.references:
        d["references"] = [_reference_dict(r) for r in v.references]
    if fingerprint and v.fingerprint is not None:
        d["fingerprint"] = _fingerprint_dict(v.fingerprint)
    return d


def _reference_dict(r: Reference) -> dict[str, Any]:
    rd: dict[str, Any] = {
        "clip": r.clip,
        "transcript": _nfc(r.transcript),
        "language": _nfc(r.language),
    }
    if r.origin is not None:
        rd["origin"] = _norm_free(r.origin, "[[voice.references]] origin")
    return rd


def _fingerprint_dict(f: Fingerprint) -> dict[str, Any]:
    return {"embedder": _nfc(f.embedder), "vector": f.vector, "threshold": float(f.threshold)}


def _card_dict(card: Card, *, identity: bool) -> dict[str, Any]:
    """Every Card field is handled here; an empty style is no style."""
    d: dict[str, Any] = {"target_language": _nfc(card.target_language)}
    if card.meta and not identity:
        d["meta"] = card.meta
    d["voice"] = _voice_dict(card.voice, fingerprint=not identity)
    if card.style:
        d["style"] = _norm_free(card.style, "[style]")
    return d


def dump_card(card: Card) -> str:
    return tomli_w.dumps(_card_dict(card, identity=False))


def _json_default(value: Any) -> str:
    if isinstance(value, dt.date | dt.time):  # datetime is a date
        return value.isoformat()
    raise TypeError(f"card value of type {type(value).__name__} has no canonical form")


def canonical_form(card: Card) -> str:
    """The string a card id hashes: the card without [meta] and [voice.fingerprint]."""
    return json.dumps(
        _card_dict(card, identity=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def card_id(card: Card) -> str:
    return hashlib.sha256(canonical_form(card).encode("utf-8")).hexdigest()[:16]


def is_card_id(value: str) -> bool:
    return bool(_CARD_ID.fullmatch(value))


def is_sidecar_ref(value: str) -> bool:
    return bool(_SHA_REF.fullmatch(value))


def sidecars(card: Card) -> list[str]:
    """Every ``sha256:`` sidecar the card refers to, in card order, without duplicates."""
    v = card.voice
    refs = [r.clip for r in v.references]
    if v.latent:
        refs.append(v.latent)
    if v.fingerprint is not None:
        refs.append(v.fingerprint.vector)
    return list(dict.fromkeys(r for r in refs if r.startswith("sha256:")))


# --- backend checks ------------------------------------------------------------------------


def validate_for_backend(
    card: Card,
    cls: type[Synthesizer],
    lookup: Callable[[str], Card] | None = None,
) -> None:
    """Fail, naming the key, when ``cls`` can't honor ``card`` exactly.

    ``lookup`` resolves a card id; it is needed only when ``[voice.blend] with`` names a card.
    """
    name = getattr(cls, "name", cls.__name__)
    v = card.voice
    if v.backend != name:
        raise CardError(
            f"card is for backend {v.backend!r}, not {name!r}; "
            "a card renders only on its own backend"
        )
    if v.source not in cls.sources:
        raise CardError(f"{name} does not support source {v.source!r}")
    own = SOURCE_FIELDS[v.source]
    for key in _BACKEND_VOICE_KEYS:
        if key == own or key in cls.voice_keys:
            continue
        if getattr(v, key) != getattr(_VOICE_DEFAULTS, key):
            raise CardError(f"{name} does not support [voice] key {key!r}")
    cls.validate_options(dict(v.options), source=v.source)
    eq = v.post.get("eq", "none")
    if eq not in EQ_PRESETS:
        raise CardError(
            f"[voice.post] eq = {eq!r} is not a known preset; "
            f"use one of {', '.join(sorted(EQ_PRESETS))}"
        )
    if v.blend is not None:
        other = v.blend["with"]
        if other.startswith("card:"):
            other_id = other.removeprefix("card:")
            if lookup is None:
                raise CardError(
                    f"[voice.blend] with = {other!r} names a card, and no card store was given "
                    "to check its backend"
                )
            blended = lookup(other_id)
            if blended.voice.backend != v.backend:
                raise CardError(
                    f"[voice.blend] with = {other!r} is a {blended.voice.backend} card; "
                    f"blending across backends is refused (this card is {v.backend})"
                )
