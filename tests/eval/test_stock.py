"""Stock voice pools for `eval tts` baselines (D41: never picked by the pipeline)."""

import pytest

from fukikae.cards.card import CardError
from fukikae.eval.stock import STOCK_POOLS, pick_stock

POOL = STOCK_POOLS["kokoro"]["en"]


def test_pick_stock_by_gender_then_reuse():
    assert pick_stock("kokoro", "en", "f", set()) == POOL["f"][0]
    assert pick_stock("kokoro", "en", "m", {POOL["m"][0]}) == POOL["m"][1]
    # exhausted: reuse from the start, then the least reused
    assert pick_stock("kokoro", "en", "m", set(POOL["m"])) == POOL["m"][0]
    assert pick_stock("kokoro", "en", "m", [*POOL["m"], POOL["m"][0]]) == POOL["m"][1]
    assert pick_stock("kokoro", "en", "f", set(), extras=True) == POOL["extras_f"][0]


def test_pick_stock_unknown_gender_alternates():
    used: list[str] = []
    picks = []
    for _ in range(4):
        v = pick_stock("kokoro", "en", "u", used)
        used.append(v)
        picks.append(v)
    assert picks == [POOL["f"][0], POOL["m"][0], POOL["f"][1], POOL["m"][1]]


def test_pick_stock_without_a_pool_names_backend_and_language():
    with pytest.raises(CardError, match="kokoro.*'de'"):
        pick_stock("kokoro", "de", "f", set())
