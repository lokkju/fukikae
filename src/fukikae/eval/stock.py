"""Stock voice pools for ``eval tts`` baselines (D41: the pipeline never picks a stock voice;
an evaluation may still compare a clone against Kokoro's stock voices)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Literal

from fukikae.cards.card import Card, CardError
from fukikae.cards.store import CardStore

Gender = Literal["f", "m", "u"]
# The stock baseline backend when a workspace names none (the profiles name no [tts] backend).
STOCK_BASELINE = "kokoro"

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


class ScratchStore(CardStore):
    """A card store for an evaluation's stock baseline cards (D41: a stock card never enters
    the workspace's store): cards are written under ``scratch`` (a temporary directory), while
    reads fall through to ``base``, so the workspace's cards and sidecars render beside them."""

    def __init__(self, scratch: Path, base: CardStore) -> None:
        super().__init__(scratch)
        self.base = base

    def has(self, card_id: str) -> bool:
        return super().has(card_id) or self.base.has(card_id)

    def get(self, card_id: str) -> Card:
        return super().get(card_id) if super().has(card_id) else self.base.get(card_id)

    def _find(self, ref: str) -> Path | None:
        return super()._find(ref) or self.base._find(ref)
