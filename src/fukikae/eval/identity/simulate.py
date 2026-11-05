"""Replay the registry's episode-by-episode matching in memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fukikae.eval.identity.segments import Segment
from fukikae.registry.matchers import Matcher, RegistryState


def simulate(
    order: list[str],
    matcher: Matcher,
    segs: Mapping[str, Sequence[Segment]],
    embs: Mapping[str, np.ndarray],
    state: RegistryState | None = None,
) -> dict[tuple[str, int], int | None]:
    """Resolve the episodes in `order`, each against the state the earlier ones built (a fresh
    registry unless `state` is given; it is updated in place). Returns (episode, segment
    index) -> character id, None for segments no character takes."""
    state = RegistryState() if state is None else state
    out: dict[tuple[str, int], int | None] = {}
    for ep in order:
        ids = matcher.fit_episode(state, ep, embs[ep], segs[ep])
        if len(ids) != len(segs[ep]):
            raise ValueError(f"{ep}: matcher returned {len(ids)} ids for {len(segs[ep])} segments")
        for i, cid in enumerate(ids):
            out[(ep, i)] = cid
    return out
