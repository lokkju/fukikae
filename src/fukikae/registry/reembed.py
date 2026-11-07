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
centroids recomputed, mismatched projections dropped, curation untouched.
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
        p.centroid_dims = Counter(len(a.centroid) for a in reg.actors() if a.centroid is not None)
        p.projection_dims = [(s.series, s.kind, int(s.matrix.shape[0])) for s in reg.projections()]
    p.exemplar_dims = Counter(len(e.embedding) for e in exemplars)
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
    return p


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
    with Registry(ws.registry_path) as reg:
        return reg.replace_embeddings(new)
