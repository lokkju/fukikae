"""Seed an empty registry from a whole season by clustering every episode's clusters."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from fukikae.registry.matching import ClusterEvidence


def seed_clusters(
    evidence: dict[str, list[ClusterEvidence]],
    distance_threshold: float,
    min_new_seconds: float,
) -> list[list[tuple[str, str]]]:
    """Group (episode, cluster) pairs by average-linkage cosine distance.

    Clusters with no embedding are left out. A group whose members total fewer than
    `min_new_seconds` of clean speech is dropped. Groups and members are sorted.
    """
    items = sorted(
        (ep, ev.cluster, ev)
        for ep, evs in evidence.items()
        for ev in evs
        if ev.embedding is not None
    )
    if not items:
        return []
    if len(items) == 1:
        labels = np.array([1])
    else:
        x = np.stack([ev.embedding for _, _, ev in items if ev.embedding is not None])
        labels = fcluster(
            linkage(x, method="average", metric="cosine"),
            t=distance_threshold,
            criterion="distance",
        )
    groups: dict[int, list[tuple[str, str]]] = {}
    seconds: dict[int, float] = {}
    for (ep, cluster, ev), g in zip(items, labels, strict=True):
        groups.setdefault(int(g), []).append((ep, cluster))
        seconds[int(g)] = seconds.get(int(g), 0.0) + ev.clean_seconds
    return sorted(sorted(m) for g, m in groups.items() if seconds[g] >= min_new_seconds)
