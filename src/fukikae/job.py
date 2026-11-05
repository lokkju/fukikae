"""Manifest, cache keys and the resumable stage runner."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config
from fukikae.workspace import Episode


class Stage(Protocol):
    name: str

    def inputs(self, ep: Episode) -> list[Path]: ...
    def outputs(self, ep: Episode) -> list[Path]: ...
    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]: ...
    def run(self, ep: Episode, cfg: Config) -> None: ...


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
        with atomic_path(self.path) as tmp:
            tmp.write_text(json.dumps(self.entries), encoding="utf-8")
        self.dirty = False


def _fingerprint(path: Path, ep: Episode, cache: _HashCache) -> str:
    resolved = path.resolve()
    if not resolved.exists():
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


def mark_stale(ep: Episode, stage: str) -> bool:
    """Forget a stage's manifest record so the next run redoes it; True if there was one."""
    manifest = load_manifest(ep)
    if manifest.stages.pop(stage, None) is None:
        return False
    save_manifest(ep, manifest)
    return True
