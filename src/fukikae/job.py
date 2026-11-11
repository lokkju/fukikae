"""Manifest, cache keys and the resumable stage runner."""

import hashlib
import json
import logging
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from fukikae import progress
from fukikae.artifacts import read_json, write_json
from fukikae.config import Config
from fukikae.workspace import Episode

log = logging.getLogger(__name__)
# Episodes' runners share the workspace's hash cache file (run_episodes runs two at once).
_HASHCACHE_SAVE = threading.Lock()


class Stage(Protocol):
    name: str

    def inputs(self, ep: Episode) -> list[Path]: ...
    def outputs(self, ep: Episode) -> list[Path]: ...
    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]: ...
    def run(self, ep: Episode, cfg: Config) -> None: ...


def _prepare(stage: Stage, ep: Episode, cfg: Config) -> None:
    """A stage may define ``prepare(ep, cfg)``: registry work that decides its inputs (the
    tts stage's automatic cards), run before its key is computed."""
    prepare = getattr(stage, "prepare", None)
    if prepare is not None:
        prepare(ep, cfg)


class StageRecord(BaseModel):
    key: str
    status: Literal["done", "failed"]
    started: datetime
    finished: datetime | None = None
    error: str | None = None


class Manifest(BaseModel):
    stages: dict[str, StageRecord] = Field(default_factory=dict)


def _manifest_path(ep: Episode) -> Path:
    return ep.work / "manifest.json"


def _cache_path(ep: Episode) -> Path:
    return ep.workspace_root / ".fukikae" / "hashcache.json"


def load_manifest(ep: Episode) -> Manifest:
    path = _manifest_path(ep)
    if not path.exists():
        return Manifest()
    try:
        return read_json(path, Manifest)
    except ValueError:
        return Manifest()


def save_manifest(ep: Episode, manifest: Manifest) -> None:
    ep.work.mkdir(parents=True, exist_ok=True)
    write_json(_manifest_path(ep), manifest)


class _HashCache:
    """sha256 of file contents, memoised by (path, size, mtime_ns)."""

    def __init__(self, ep: Episode) -> None:
        self.path = _cache_path(ep)
        self.entries: dict[str, list[Any]] = {}
        self.dirty = False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.entries = data
        except (OSError, ValueError):
            pass

    def digest(self, path: Path) -> str:
        st = path.stat()
        key = str(path)
        hit = self.entries.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return str(hit[2])
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        self.entries[key] = [st.st_size, st.st_mtime_ns, digest]
        self.dirty = True
        return digest

    def save(self) -> None:
        if not self.dirty:
            return
        from fukikae.artifacts import atomic_path

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _HASHCACHE_SAVE:
            # merge what another runner saved meanwhile (it is a memo: a lost entry only costs
            # a rehash, but there is no need to lose it)
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = {}
            merged = {**saved, **self.entries} if isinstance(saved, dict) else self.entries
            with atomic_path(self.path) as tmp:
                tmp.write_text(json.dumps(merged), encoding="utf-8")
        self.dirty = False


def _fingerprint(path: Path, ep: Episode, cache: _HashCache) -> str:
    resolved = path.resolve()
    if not resolved.exists():
        # Named relative to the workspace (ruling CD6), so a copied or moved workspace isn't
        # stale because an optional input such as glossary.toml is absent.
        root = ep.workspace_root.resolve()
        if resolved.is_relative_to(root):
            return f"missing:<workspace>/{resolved.relative_to(root).as_posix()}"
        return f"missing:{resolved}"
    if resolved.is_dir():
        parts = [
            f"{p.relative_to(resolved)}={_fingerprint(p, ep, cache)}"
            for p in sorted(resolved.rglob("*"))
            if p.is_file()
        ]
        return "dir:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()
    if resolved.is_relative_to(ep.workspace_root.resolve()):
        return "sha256:" + cache.digest(resolved)
    st = resolved.stat()
    return f"stat:{resolved}:{st.st_size}:{st.st_mtime_ns}"


def _cache_key(stage: Stage, ep: Episode, cfg: Config, cache: _HashCache) -> str:
    payload = {
        "stage": stage.name,
        "config": json.dumps(stage.config_key(cfg, ep), sort_keys=True),
        "inputs": [_fingerprint(p, ep, cache) for p in stage.inputs(ep)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_key(stage: Stage, ep: Episode, cfg: Config) -> str:
    cache = _HashCache(ep)
    key = _cache_key(stage, ep, cfg, cache)
    cache.save()
    return key


def run_stages(
    ep: Episode,
    cfg: Config,
    stages: Sequence[Stage],
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[str]:
    names = [s.name for s in stages]
    for label, value in (("from_stage", from_stage), ("to_stage", to_stage)):
        if value is not None and value not in names:
            raise ValueError(f"unknown {label} {value!r}; stages are {names}")
    start = names.index(from_stage) if from_stage else 0
    stop = names.index(to_stage) + 1 if to_stage else len(stages)

    manifest = load_manifest(ep)
    cache = _HashCache(ep)
    ran: list[str] = []
    try:
        for i, stage in enumerate(stages[:stop]):
            _prepare(stage, ep, cfg)
            key = _cache_key(stage, ep, cfg, cache)
            record = manifest.stages.get(stage.name)
            stale = (
                i >= start
                and from_stage is not None
                or record is None
                or record.status != "done"
                or record.key != key
                or any(not p.exists() for p in stage.outputs(ep))
            )
            if not stale:
                continue
            started = datetime.now(UTC)
            try:
                with progress.stage(ep.id, stage.name):
                    stage.run(ep, cfg)
            except BaseException as e:
                manifest.stages[stage.name] = StageRecord(
                    key=key,
                    status="failed",
                    started=started,
                    finished=datetime.now(UTC),
                    error=f"{type(e).__name__}: {e}",
                )
                save_manifest(ep, manifest)
                raise
            # Outputs changed; hash them fresh for later stages' keys.
            manifest.stages[stage.name] = StageRecord(
                key=key, status="done", started=started, finished=datetime.now(UTC)
            )
            save_manifest(ep, manifest)
            ran.append(stage.name)
    finally:
        cache.save()
    return ran


@dataclass
class EpisodeResult:
    """What ``run_episodes`` did for one episode: the stages it ran, and the error that stopped
    it (None: none)."""

    ran: list[str] = field(default_factory=list)
    error: BaseException | None = None


def run_episodes(
    episodes: Sequence[Episode],
    cfg: Config,
    stages: Sequence[Stage],
    *,
    tail: Sequence[str],
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> dict[str, EpisodeResult]:
    """Run ``stages`` for several episodes, in order, overlapping each episode's CPU ``tail``
    (the trailing stages named there: timefit and mix) with the next episode's head (D26). One
    thread runs the heads, where every GPU stage is, so GPU work stays serial in episode order;
    one more runs the tails, at most one at a time, each after its own head. A tail reads only
    its episode's files, and every stage writes only its own episode's directory, so the
    outputs and manifests are those of running the episodes one after another. After a
    failure no further head starts; a tail already started finishes. ``from_stage`` and
    ``to_stage`` apply to every episode as in ``run_stages``."""
    names = [s.name for s in stages]
    for label, value in (("from_stage", from_stage), ("to_stage", to_stage)):
        if value is not None and value not in names:
            raise ValueError(f"unknown {label} {value!r}; stages are {names}")
    split = len(stages)
    while split > 0 and stages[split - 1].name in tail:
        split -= 1
    head, rest = list(stages[:split]), list(stages[split:])
    stop = names.index(to_stage) + 1 if to_stage else len(stages)
    if stop <= split:
        head, rest = head[:stop], []
    else:
        rest = rest[: stop - split]
    head_from = from_stage if from_stage in {s.name for s in head} else None
    if from_stage is None:
        tail_from = None
    elif head_from is not None:
        tail_from = rest[0].name if rest else None  # --from forces every later stage
    else:
        tail_from = from_stage

    results: dict[str, EpisodeResult] = {}
    failed = threading.Event()

    def run_tail(ep: Episode) -> None:
        try:
            progress.episode(ep.id, "tail")
            results[ep.id].ran += run_stages(ep, cfg, rest, tail_from)
            progress.episode(ep.id, "done")
        except BaseException as e:
            results[ep.id].error = e
            failed.set()
            progress.episode(ep.id, "failed")
            log.error("%s: %s: %s", ep.id, type(e).__name__, e)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fukikae-tail") as pool:
        pending: Future[None] | None = None
        for ep in episodes:
            if failed.is_set():
                break
            results[ep.id] = EpisodeResult()
            try:
                progress.episode(ep.id, "head")
                results[ep.id].ran += run_stages(ep, cfg, head, head_from) if head else []
            except BaseException as e:
                results[ep.id].error = e
                failed.set()
                progress.episode(ep.id, "failed")
                if not isinstance(e, Exception):
                    raise
                break
            if not rest:
                progress.episode(ep.id, "done")
            else:
                progress.episode(ep.id, "waiting for tail")
                if pending is not None:
                    pending.result()  # one tail at a time
                pending = pool.submit(run_tail, ep)
        if pending is not None:
            pending.result()
    return results


def mark_stale(ep: Episode, stage: str) -> bool:
    """Forget a stage's manifest record so the next run redoes it; True if there was one."""
    manifest = load_manifest(ep)
    if manifest.stages.pop(stage, None) is None:
        return False
    save_manifest(ep, manifest)
    return True
