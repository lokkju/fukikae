"""Stock voices: the pools fukikae picks from, per TTS backend and target language, and the
card a stock voice becomes.

A stock card holds nothing but the backend, its pinned model revision, ``source = "stock"`` and
the voice id, so the same voice on the same model revision is always the same card id (CD1).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from typing import Literal

from fukikae.cards.card import Card, CardError, Voice

Gender = Literal["f", "m", "u"]

# backend -> target language -> pool -> voice ids, best first. "f" and "m" are for characters,
# "extras_f" and "extras_m" for voices no character uses, so an extra never sounds like one.
STOCK_POOLS: dict[str, dict[str, dict[str, list[str]]]] = {
    "kokoro": {
        "en": {
            "f": [
                "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "bf_emma", "bf_isabella",
            ],
            "m": [
                "am_adam", "am_michael", "am_fenrir", "am_puck", "bm_george", "bm_lewis",
                "bm_fable",
            ],
            "extras_f": ["af_alloy", "bf_alice"],
            "extras_m": ["am_echo", "bm_daniel"],
        }
    }
}  # fmt: skip


def _pools(backend: str, language: str) -> dict[str, list[str]]:
    by_lang = STOCK_POOLS.get(backend)
    if by_lang is None:
        raise CardError(
            f"no stock voice pools for TTS backend {backend!r}; "
            f"backends with pools: {', '.join(sorted(STOCK_POOLS))}"
        )
    pools = by_lang.get(language)
    if pools is None:
        raise CardError(
            f"no {backend} stock voice pool for target language {language!r} "
            f"(pools exist for {', '.join(sorted(by_lang))}); assign each character a card "
            "with `fukikae voices assign`"
        )
    return pools


def pick_stock(
    backend: str,
    language: str,
    gender: Gender,
    used: Collection[str],
    extras: bool = False,
) -> str:
    """The next voice of ``gender``'s pool that isn't in ``used``. ``u`` takes whichever of the
    f and m pools has fewer voices used (f on a tie), so unknown genders alternate. Once a pool
    is exhausted, voices are reused from the start: the least used first, counting repeats in
    ``used`` when it is a list."""
    pools = _pools(backend, language)
    prefix = "extras_" if extras else ""
    if gender == "u":
        f_used = sum(1 for v in used if v in pools[f"{prefix}f"])
        m_used = sum(1 for v in used if v in pools[f"{prefix}m"])
        gender = "f" if f_used <= m_used else "m"
    pool = pools[f"{prefix}{gender}"]
    for voice in pool:
        if voice not in used:
            return voice
    counts = Counter(v for v in used if v in pool)
    return min(pool, key=lambda v: (counts[v], pool.index(v)))


def stock_card(backend: str, voice_id: str, target_language: str, revision: str) -> Card:
    return Card(
        target_language=target_language,
        voice=Voice(backend=backend, model_revision=revision, source="stock", voice=voice_id),
        meta={"title": f"{backend} stock voice {voice_id}"},
    )


def parse_stock_ref(ref: str) -> tuple[str, str]:
    """``stock:<backend>:<voice>`` as (backend, voice)."""
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "stock" or not parts[1] or not parts[2]:
        raise CardError(f"{ref!r} is not a stock voice; write it as stock:<backend>:<voice>")
    return parts[1], parts[2]


def revision(backend: str) -> str:
    """The model revision a TTS backend renders with, which its stock cards record."""
    from fukikae.backends import loader

    rev = getattr(loader.backend_class("tts", backend), "model_revision", None)
    if not isinstance(rev, str) or not rev:
        raise CardError(f"TTS backend {backend!r} declares no model_revision")
    return rev


def check_stock_voice(backend: str, voice_id: str) -> None:
    """Refuse a voice id the backend doesn't ship (when it lists its voices)."""
    from fukikae.backends import loader

    known = getattr(loader.backend_class("tts", backend), "stock_voices", None)
    if known is not None and voice_id not in known:
        raise CardError(
            f"{backend} has no stock voice {voice_id!r}; it has {', '.join(sorted(known))}"
        )
