"""Re-embed the registry after an embedder change (Ruling R19): `fukikae registry reembed`.

Each exemplar is one row of its episode cluster's ``04-diarize/clusters.npz`` (the characters
stage, `registry cluster` and `registry resolve` add a cluster's rows in order). Its span is
the same row of ``<cluster>/spans``, so the configured embedder can embed it again from the
episode's separated stem (``01-separate/vocals_16k.wav``, what the diarize stage embeds).

Before anything is embedded every episode holding exemplars is checked: its diarize outputs
and stem must exist, and each of its exemplar clusters must still be the cluster the
exemplars came from. That holds when the stored rows equal the cluster's ``clusters.npz``
rows (diarize has not re-run), or when the episode's ``assignments.json`` recorded exactly
the current ``turns.json`` turns (diarize re-ran and produced the same turns, so the same
spans) and the cluster has one span per exemplar. Anything else is refused, naming the
episodes: their cluster labels may mean other speakers now, and guessing would file voices
under the wrong actors.

The new rows then replace the old in one registry transaction (`Registry.replace_embeddings`):
unlocked actors with exemplars get the mean of their new rows, mismatched projections are
dropped, curation is untouched.

Locked actors and actors with no exemplars are never recomputed from exemplars (Ruling R20): a
lock fixes the centroid at lock time while later exemplars keep arriving, and an actor kept by
prune protection has nothing to recompute from. Their stored centroid is kept when it already
has the new embedder's dimension. Otherwise it is carried across by a linear map fit on the
paired old and new exemplar rows of its dimension: ridge least squares with a bias column,
``W = (X'X + lam*P)^-1 X'Y`` with ``lam = RIDGE * trace(X'X) / old_dim`` and the bias left
unpenalized (``P`` is the identity with a zero for the bias), then L2-normalized as every
stored centroid is. The fit needs at least ``MIN_PAIRS_PER_DIM`` pairs per old dimension (two
per parameter of each output's weight vector, roughly); with fewer, reembed refuses, naming
the actors, and changes nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from fukikae import runctx
from fukikae.artifacts import read_json
from fukikae.backends.base import Embedder
from fukikae.backends.loader import create
from fukikae.config import BackendConfig
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.models import Assignments, Diarization
from fukikae.registry.store import Exemplar, ReembedResult, Registry
from fukikae.workspace import Workspace

RIDGE = 1e-3
MIN_PAIRS_PER_DIM = 2  # pairs needed: 2 x the old dimension


class ReembedRefusedError(ValueError):
    """Some episode's exemplars cannot be re-embedded; nothing was changed."""


@dataclass
class ReembedPlan:
    embedder: BackendConfig
    dim: int  # the configured embedder's dimension
    exemplar_dims: Counter[int] = field(default_factory=Counter)
    centroid_dims: Counter[int] = field(default_factory=Counter)
    projection_dims: list[tuple[str, str, int]] = field(default_factory=list)
    # episode -> [(exemplar ids in npz-row order, spans)] per cluster
    work: dict[str, list[tuple[list[int], list[tuple[float, float]]]]] = field(default_factory=dict)
    # locked or exemplar-less actors with a centroid: id -> (description, centroid dim)
    fixed: dict[int, tuple[str, int]] = field(default_factory=dict)
    centroids: dict[int, np.ndarray] = field(default_factory=dict)  # their stored centroids
    old: dict[int, np.ndarray] = field(default_factory=dict)  # exemplar id -> stored row

    def treatment(self, aid: int) -> str:
        desc, d = self.fixed[aid]
        if d == self.dim:
            return f"actor {aid} ({desc}): centroid kept as stored ({d}-d)"
        n = sum(1 for v in self.old.values() if len(v) == d)
        return (
            f"actor {aid} ({desc}): centroid mapped {d}-d -> {self.dim}-d by a ridge fit on "
            f"{n} exemplar pairs"
        )

    @property
    def spans(self) -> int:
        return sum(len(s) for groups in self.work.values() for _, s in groups)


def _check_episode(
    ws: Workspace, eid: str, groups: dict[str | None, list[Exemplar]]
) -> tuple[list[tuple[list[int], list[tuple[float, float]]]], str | None]:
    """The episode's (exemplar ids, spans) per cluster, or the reason it cannot be done."""
    work = ws.work_dir(eid)
    turns_path = work / "04-diarize" / "turns.json"
    npz = work / "04-diarize" / "clusters.npz"
    stem = work / "01-separate" / "vocals_16k.wav"
    missing = [p.relative_to(work).as_posix() for p in (turns_path, npz, stem) if not p.exists()]
    if missing:
        return [], f"missing {', '.join(missing)}"
    recorded = None
    assignments = work / "05-characters" / "assignments.json"
    if assignments.exists():
        recorded = read_json(assignments, Assignments).turns
    same_turns = recorded is not None and recorded == read_json(turns_path, Diarization).turns
    out: list[tuple[list[int], list[tuple[float, float]]]] = []
    with np.load(npz) as z:
        labels = {str(x) for x in z["labels"]}
        for cluster, exs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            if cluster is None or cluster not in labels:
                return [], f"cluster {cluster} is not in its clusters.npz"
            spans = [
                (float(a), float(b)) for a, b in np.asarray(z[f"{cluster}/spans"]).reshape(-1, 2)
            ]
            rows = np.asarray(z[f"{cluster}/embeddings"], dtype=np.float32)
            stored = np.stack([e.embedding for e in exs])
            same_rows = stored.shape == rows.shape and np.allclose(stored, rows, atol=1e-6)
            if len(spans) != len(exs) or not (same_rows or same_turns):
                return [], (
                    f"cluster {cluster} no longer matches its {len(exs)} exemplars (diarize "
                    "re-ran since they were added)"
                )
            out.append(([e.id for e in exs], spans))
    return out, None


def plan(ws: Workspace) -> ReembedPlan:
    """Check every episode holding exemplars; raise ReembedRefusedError naming the ones that
    cannot be re-embedded. Reads only."""
    cfg = ws.config
    emb_cfg = cfg.resolve("embed")
    p = ReembedPlan(emb_cfg, int(cast(Embedder, create("embed", emb_cfg, "cpu")).dim))
    with Registry(ws.registry_path) as reg:
        exemplars = reg.exemplars()
        actors = reg.actors()
        p.centroid_dims = Counter(len(a.centroid) for a in actors if a.centroid is not None)
        p.projection_dims = [(s.series, s.kind, int(s.matrix.shape[0])) for s in reg.projections()]
        curated = reg.curated_characters()
        named_chars = {c.actor_id for c in reg.characters() if c.id in curated}
    p.exemplar_dims = Counter(len(e.embedding) for e in exemplars)
    p.old = {e.id: e.embedding for e in exemplars}
    holders = {e.actor_id for e in exemplars}
    for a in actors:
        if a.centroid is None or not (a.locked or a.id not in holders):
            continue
        tags = ["locked" if a.locked else "no exemplars"]
        if a.name:
            tags.append(f"named {a.name}")
        elif a.id in named_chars:
            tags.append("curated character")
        p.fixed[a.id] = (", ".join(tags), len(a.centroid))
        p.centroids[a.id] = a.centroid
    by_ep: dict[str, dict[str | None, list[Exemplar]]] = {}
    for e in exemplars:
        by_ep.setdefault(e.episode, {}).setdefault(e.cluster, []).append(e)
    refused: list[str] = []
    for eid in sorted(by_ep):
        groups, why = _check_episode(ws, eid, by_ep[eid])
        if why is not None:
            refused.append(f"{eid} ({why})")
        else:
            p.work[eid] = groups
    if refused:
        raise ReembedRefusedError(
            "cannot re-embed the exemplars of " + "; ".join(refused) + ". reembed needs each "
            "episode's diarize output from when its exemplars were added (run it before "
            "re-running diarize with the new embedder). Nothing was changed"
        )
    short = [
        aid
        for aid, (_, d) in p.fixed.items()
        if d != p.dim and p.exemplar_dims.get(d, 0) < MIN_PAIRS_PER_DIM * d
    ]
    if short:
        raise ReembedRefusedError(
            "too few exemplars to map the fixed centroids of "
            + "; ".join(
                f"actor {a} ({p.fixed[a][0]}, {p.fixed[a][1]}-d: "
                f"{p.exemplar_dims.get(p.fixed[a][1], 0)} pairs, need "
                f"{MIN_PAIRS_PER_DIM * p.fixed[a][1]})"
                for a in short
            )
            + f" into the {p.dim}-d space. Unlock them (`fukikae registry lock <workspace> <id> "
            "--unlock`) before reembed and lock them again afterwards; an actor with no "
            "exemplars needs some (resolve or run an episode it speaks in) first. Nothing was "
            "changed"
        )
    return p


def fit_map(old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge least squares ``new ~ old @ w + b`` (see the module docstring): (w, b)."""
    x = np.hstack([np.asarray(old, np.float64), np.ones((len(old), 1))])
    y = np.asarray(new, np.float64)
    xtx = x.T @ x
    d = old.shape[1]
    lam = RIDGE * float(np.trace(xtx[:d, :d])) / d
    pen = np.eye(d + 1)
    pen[d, d] = 0.0
    sol = np.linalg.solve(xtx + lam * pen, x.T @ y)
    return sol[:d], sol[d]


def run(ws: Workspace, p: ReembedPlan) -> ReembedResult:
    """Embed every planned span with the configured embedder, then replace the exemplars'
    embeddings in one transaction."""
    cfg = ws.config
    device = device_string(cfg)
    embedder = cast(Embedder, create("embed", p.embedder, device))
    new: dict[int, np.ndarray] = {}
    if p.work:
        if device.startswith("cuda"):
            wait_for_vram(cfg.gpu_device, embedder.vram_mb, runctx.vram_timeout_s, query=free_mb)
        embedder.load()
        try:
            for eid, groups in p.work.items():
                spans = [s for _, ss in groups for s in ss]
                rows = np.asarray(
                    embedder.embed(ws.work_dir(eid) / "01-separate" / "vocals_16k.wav", spans),
                    dtype=np.float32,
                )
                if rows.shape != (len(spans), p.dim):
                    raise ValueError(
                        f"{eid}: the embedder returned {rows.shape} for {len(spans)} spans, "
                        f"expected ({len(spans)}, {p.dim})"
                    )
                ids = [i for idl, _ in groups for i in idl]
                new.update(zip(ids, rows, strict=True))
        finally:
            embedder.unload()
    keep = {a for a, (_, d) in p.fixed.items() if d == p.dim}
    mapped: dict[int, np.ndarray] = {}
    for d in sorted({d for _, d in p.fixed.values() if d != p.dim}):
        ids = [i for i, v in p.old.items() if len(v) == d]
        w, b = fit_map(np.stack([p.old[i] for i in ids]), np.stack([new[i] for i in ids]))
        for aid, (_, cd) in p.fixed.items():
            if cd == d:
                v = np.asarray(p.centroids[aid], np.float64) @ w + b
                mapped[aid] = (v / np.linalg.norm(v)).astype(np.float32)
    with Registry(ws.registry_path) as reg:
        return reg.replace_embeddings(new, keep=keep, fixed=mapped)
