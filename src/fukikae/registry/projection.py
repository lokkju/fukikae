"""Linear projections of speaker embeddings, fit from labeled episodes and stored per series.

`fit_lda` is the fit `registry.matchers.LdaMatcher` makes in the identity harness
(scikit-learn's LinearDiscriminantAnalysis, eigen solver, automatic shrinkage); `registry
adapt` stores its result in the registry (`Registry.save_projection`) and the characters
stage with ``[characters] matcher = "lda"`` matches in that space (`projected_tops`).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fukikae.registry.matching import ClusterEvidence, mean_embedding

if TYPE_CHECKING:
    from fukikae.registry.store import Registry, StoredProjection

KIND_LDA = "lda"
DEFAULT_SERIES = "default"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Projection:
    """``x -> unit rows of (x - mean) @ matrix``; `matrix` is (dim, k). The LDA fit leaves
    `mean` at zero: scikit-learn's eigen-solver transform does not center."""

    matrix: np.ndarray
    mean: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def truncated(self, dims: int | None) -> Projection:
        """The first `dims` components (the eigen solver orders them by eigenvalue)."""
        if dims is None or dims >= self.matrix.shape[1]:
            return self
        return Projection(self.matrix[:, :dims], self.mean)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.dim:
            raise ValueError(
                f"embedding dimension {x.shape[1]} does not match the projection's {self.dim}"
            )
        y = (x - self.mean) @ self.matrix
        n = np.linalg.norm(y, axis=1, keepdims=True)
        return (y / np.where(n == 0, 1.0, n)).astype(np.float32)


def fit_lda(x: np.ndarray, labels: Sequence[str], dims: int | None = None) -> Projection | None:
    """LDA on rows `x` labeled `labels`, keeping labels with at least two rows; None with
    fewer than two such labels. `dims` caps the components (at most labels - 1)."""
    counts = Counter(labels)
    keep = [i for i, y in enumerate(labels) if counts[y] >= 2]
    classes = {labels[i] for i in keep}
    if len(classes) < 2:
        return None
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    xs = np.asarray(x, dtype=np.float64)[keep]
    ys = [labels[i] for i in keep]
    top = min(len(classes) - 1, xs.shape[1])
    n = top if dims is None else min(dims, top)
    lda: Any = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=n)
    lda.fit(xs, ys)
    return Projection(np.asarray(lda.scalings_[:, :n], dtype=np.float64), np.zeros(xs.shape[1]))


def series_of(cfg: Any) -> str:
    """The series a workspace's projections are stored under: ``[series] title`` when the
    workspace sets one explicitly, else the older top-level ``series = "Name"``, else
    "default" (a workspace has one registry, so one series). The title's own default (the
    workspace directory's name) does not count: only a title written in fukikae.toml keys
    projections. A projection stored under "default" before the title was set is still found
    by `load_series_projection`."""
    value = cfg.raw.get("series")
    if isinstance(value, dict):
        title = value.get("title")
        if isinstance(title, str) and title:
            return title
    elif isinstance(value, str) and value:
        return value
    return DEFAULT_SERIES


def load_series_projection(
    reg: Registry, cfg: Any, kind: str, *, warn: bool = True
) -> tuple[StoredProjection | None, list[str]]:
    """The workspace's stored projection of `kind` and the series keys tried, in order:
    `series_of(cfg)`, then "default" when that differs and missed (a projection fit before
    ``[series] title`` was set). Using the fallback logs a warning naming both keys unless
    `warn` is False."""
    series = series_of(cfg)
    tried = [series]
    stored = reg.load_projection(series, kind)
    if stored is None and series != DEFAULT_SERIES:
        tried.append(DEFAULT_SERIES)
        stored = reg.load_projection(DEFAULT_SERIES, kind)
        if stored is not None and warn:
            log.warning(
                "no %s projection stored for series %r; using the one stored under %r. "
                "Run `fukikae registry adapt` to fit one under %r.",
                kind,
                series,
                DEFAULT_SERIES,
                series,
            )
    return stored, tried


def projected_tops(
    evidence: Sequence[ClusterEvidence],
    rows: Mapping[str, np.ndarray],
    actor_rows: Mapping[int, np.ndarray],
    projection: Projection,
) -> dict[str, tuple[int, float] | None]:
    """Each cluster's best (actor id, cosine) in the projected space: the normalized mean of
    its projected rows against each actor's (the normalized mean of its projected exemplar
    rows), first best on ties; `registry.matching.match`'s scoring, projected."""
    cents = [
        (aid, c)
        for aid in sorted(actor_rows)
        if len(actor_rows[aid])
        and (c := mean_embedding(projection.apply(actor_rows[aid]))) is not None
    ]
    tops: dict[str, tuple[int, float] | None] = {}
    for ev in evidence:
        r = rows.get(ev.cluster)
        mean = None if r is None or not len(r) else mean_embedding(projection.apply(r))
        top: tuple[int, float] | None = None
        if mean is not None:
            for aid, c in cents:
                s = float(np.dot(c, mean))
                if top is None or s > top[1]:
                    top = (aid, s)
        tops[ev.cluster] = top
    return tops
