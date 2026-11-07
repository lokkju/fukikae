"""Replay the registry's episode-by-episode matching in memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fukikae.eval.identity.segments import Segment
from fukikae.registry.matchers import AllEpisodesMatcher, Matcher, RegistryState


def simulate(
    order: list[str],
    matcher: Matcher | AllEpisodesMatcher,
    segs: Mapping[str, Sequence[Segment]],
    embs: Mapping[str, np.ndarray],
    state: RegistryState | None = None,
) -> dict[tuple[str, int], int | None]:
    """Resolve the episodes in `order`, each against the state the earlier ones built (a fresh
    registry unless `state` is given; it is updated in place). Returns (episode, segment
    index) -> character id, None for segments no character takes.

    A matcher with `fit_all` (`joint`) sees every episode at once and keeps no registry
    state, so passing `state` with one is an error."""
    out: dict[tuple[str, int], int | None] = {}
    fit_all = getattr(matcher, "fit_all", None)
    if fit_all is not None:
        if state is not None:
            raise ValueError(f"{matcher.name} keeps no registry state; pass none")
        got = fit_all(list(order), {e: embs[e] for e in order}, {e: segs[e] for e in order})
        for ep in order:
            if len(got[ep]) != len(segs[ep]):
                raise ValueError(
                    f"{ep}: matcher returned {len(got[ep])} ids for {len(segs[ep])} segments"
                )
            for i, cid in enumerate(got[ep]):
                out[(ep, i)] = cid
        return out
    state = RegistryState() if state is None else state
    for ep in order:
        ids = matcher.fit_episode(state, ep, embs[ep], segs[ep])  # type: ignore[union-attr]
        if len(ids) != len(segs[ep]):
            raise ValueError(f"{ep}: matcher returned {len(ids)} ids for {len(segs[ep])} segments")
        for i, cid in enumerate(ids):
            out[(ep, i)] = cid
    return out
