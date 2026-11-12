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


def test_scratch_store_writes_apart_and_reads_through(tmp_path):
    """Stock baseline cards go to a scratch store (D41: never the workspace's), while every
    workspace card and sidecar stays readable through it."""
    from fukikae.cards.card import parse_card
    from fukikae.cards.stock import stock_card
    from fukikae.cards.store import CardStore
    from fukikae.eval.stock import ScratchStore

    base = CardStore(tmp_path / "ws")
    design = base.put(parse_card(
        'target_language = "en"\n\n[voice]\nbackend = "fake"\nmodel_revision = "1.0"\n'
        'source = "design"\ndescription = "calm"\n'
    ))  # fmt: skip
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFF....")
    ref = base.add_sidecar(clip, "clip")
    scratch = ScratchStore(tmp_path / "scratch", base)
    stock = scratch.put(stock_card("kokoro", "af_heart", "en", "r1"))
    assert scratch.has(stock) and not base.has(stock)
    assert not list(base.root.glob(f"{stock}.toml"))
    assert scratch.has(design) and scratch.get(design) == base.get(design)
    assert scratch.sidecar_path(ref) == base.sidecar_path(ref)
    assert scratch.missing_sidecars(base.get(design)) == []
