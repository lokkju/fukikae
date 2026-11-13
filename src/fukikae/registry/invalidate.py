"""Which episodes a registry change makes stale, and keeping their files in step with it.

Shared by `fukikae registry` and `fukikae.studio.apply`, so an identity change made on the
command line and the same change made in the studio invalidate the same stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from fukikae.workspace import Episode, Workspace

if TYPE_CHECKING:
    from collections.abc import Iterable


class Invalidation(NamedTuple):
    """One stage of one episode that reruns on the next `fukikae run`."""

    episode: str
    stage: str


def episode(ws: Workspace, episode_id: str) -> Episode:
    """The episode by id alone (its source file needn't exist for its manifest to change)."""
    return Episode(
        source=Path(episode_id), id=episode_id, work=ws.work_dir(episode_id), workspace_root=ws.root
    )


def stale_message(episodes: Iterable[str], stage: str = "characters") -> str:
    """The line `registry` prints after marking ``stage`` stale on ``episodes``."""
    eps = list(episodes)
    if not eps:
        return "no episode has exemplars under it; nothing is stale"
    return f"stale ({stage} will rerun on the next `fukikae run`): {', '.join(eps)}"


def mark_stale(ws: Workspace, episodes: Iterable[str], stage: str = "characters") -> str:
    """Delete each episode's ``stage`` manifest record; returns `stale_message`."""
    from fukikae import job

    eps = list(episodes)
    for eid in eps:
        job.mark_stale(episode(ws, eid), stage)
    return stale_message(eps, stage)


def character_episodes(ws: Workspace, reg: Any, character_id: int, actor_id: int) -> list[str]:
    """The episodes a character speaks in: those with its actor's exemplars and those whose
    assignments name it."""
    from fukikae.registry.usage import episodes_naming

    found = set(reg.episode_counts(actor_id)) | set(episodes_naming(ws.root, (), (character_id,)))
    return sorted(found)


def episodes_with_key(ws: Workspace, key: str) -> list[str]:
    """Episodes whose names stage used a line of this key."""
    from fukikae.artifacts import read_json
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamesResult

    out = []
    episodes = ws.root / ".fukikae" / "episodes"
    for work in sorted(episodes.iterdir()) if episodes.is_dir() else []:
        path = names_path(work)
        if not path.is_file():
            continue
        try:
            r = read_json(path, NamesResult)
        except ValueError:
            continue
        if r.used and any(ln.key == key for ln in r.visible_lines()):
            out.append(work.name)
    return out


def remap_assignments(
    ws: Workspace,
    episode: str,
    actor_map: dict[int, int],
    only_cluster: str | None = None,
    character: tuple[int, int] | None = None,
    character_map: dict[int, int] | None = None,
) -> None:
    """Keep an episode's assignments.json in step with a merge or split.

    Pinning copies actor and character ids from it, so stale ids would survive a rerun. A
    split (``character``) also makes the split label a voice decision (``source = "voice"``,
    no ``name``): the operator has overruled whatever name put it there, and the label's pin
    to the new actor and character then carries over like any acoustic one.
    """
    from fukikae.artifacts import read_json, write_json
    from fukikae.models import Assignments

    path = ws.work_dir(episode) / "05-characters" / "assignments.json"
    if not path.exists():
        return
    a = read_json(path, Assignments)
    out = []
    for c in a.clusters:
        if only_cluster is not None and c.cluster != only_cluster:
            out.append(c)
            continue
        upd: dict[str, int | str | None] = {}
        if c.actor_id in actor_map:
            upd["actor_id"] = actor_map[c.actor_id]
        if character is not None and c.cluster == only_cluster:
            upd["character_id"] = character[1]
            upd["source"] = "voice"
            upd["name"] = None
        if character_map and c.character_id in character_map:
            upd["character_id"] = character_map[c.character_id]
        out.append(c.model_copy(update=upd))
    write_json(path, a.model_copy(update={"clusters": out}))
