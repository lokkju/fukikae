"""Blind labels for an A/B listening test, shared by ``fukikae eval tts-ab`` and the studio.

`blind_order` gives each candidate of one slot (a cue, a scene, a studio line) a letter: the
names sorted, then shuffled by ``random.Random(f"ab-blind:{seed}:{slot}")``. ``seed`` is the
caller's secret (eval keeps it only in ``key.toml``; the studio draws one per server and keeps
it in memory), so the letters say nothing until the key is opened. `blind_pair` is the two-way
case the studio's compare uses. It lives outside `fukikae.eval` because the studio is
pipeline code (R1: pipeline code never imports eval).
"""

from __future__ import annotations

import random
import string
from collections.abc import Sequence

LABELS = string.ascii_uppercase

Pair = tuple[str, str]  # the two candidates in play order: what A is, then what B is
Key = dict[str, str]  # label -> candidate


def blind_order(candidates: Sequence[str], *, seed: int, slot: str) -> dict[str, str]:
    """Label -> candidate for one cue or scene: the names sorted, then shuffled by
    ``random.Random(f"ab-blind:{seed}:{slot}")``."""
    order = sorted(candidates)
    random.Random(f"ab-blind:{seed}:{slot}").shuffle(order)  # noqa: S311 - reproducible
    return {LABELS[n]: c for n, c in enumerate(order)}


def blind_pair(a: str, b: str, seed: int, slot: str = "pair") -> tuple[Pair, Key]:
    """Two candidates blinded as A and B (`blind_order` of the two): the pair in play order
    and its key. The same two in either order get the same letters. ValueError unless they
    differ."""
    if a == b:
        raise ValueError(f"a blind pair needs two different candidates, not {a!r} twice")
    key = blind_order([a, b], seed=seed, slot=slot)
    return (key["A"], key["B"]), key


__all__ = ["LABELS", "Key", "Pair", "blind_order", "blind_pair"]
