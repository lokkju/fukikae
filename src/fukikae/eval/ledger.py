"""The results ledger: one append-only JSONL file of evaluation rows, pooled across workspaces.

Every evaluator (``eval asr``, ``eval identity``, later ``eval embedders`` and ``eval tts``)
appends one row per run and split; ``eval report`` reads them back. The file is user-level
(``$XDG_DATA_HOME/fukikae/results.jsonl``), overridden by ``FUKIKAE_RESULTS`` and by a
command's ``--ledger``. A row from a newer schema, or a line cut short by a killed process,
is skipped with a warning on read, never fatal.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fukikae.config import BackendConfig

SCHEMA = 1
STAGES = ("asr", "identity", "embedder", "tts")
SPLITS = ("tune", "holdout")

_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LedgerRow:
    schema: int
    run_id: str
    date: str  # UTC, ISO 8601 to the second
    commit: str  # fukikae's git commit, "-dirty" when the checkout has changes
    stage: str
    backend: str
    model: str | None
    revision: str | None
    options_hash: str
    options: dict[str, Any]
    language: str
    genre: str
    series: str
    test_set: str
    split: str
    episodes: list[str]
    metrics: dict[str, float | int | str]
    detail: dict[str, Any] = field(default_factory=dict)
    workspace: str | None = None
    tag: str | None = None


_FIELDS = tuple(f.name for f in dataclasses.fields(LedgerRow))
_REQUIRED = tuple(
    f.name
    for f in dataclasses.fields(LedgerRow)
    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
)


def ledger_path(override: Path | None = None) -> Path:
    """`override`, else ``FUKIKAE_RESULTS``, else ``$XDG_DATA_HOME/fukikae/results.jsonl``
    (``~/.local/share`` when XDG_DATA_HOME is unset)."""
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get("FUKIKAE_RESULTS")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "fukikae" / "results.jsonl"


def options_hash(options: Mapping[str, Any]) -> str:
    """The first 12 hex digits of sha256 over the options as canonical JSON."""
    text = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 - argument list, no shell
        ["git", "-C", str(_PACKAGE_DIR), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return out.stdout


def git_commit() -> str:
    """fukikae's commit (12 hex digits), with ``-dirty`` for uncommitted changes; "unknown"
    when the package is not in a git checkout."""
    try:
        commit = _git("rev-parse", "--short=12", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not commit:
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_identity(kind: str, cfg: BackendConfig) -> tuple[str | None, str | None]:
    """(model, revision) for a backend config, without loading the model.

    The model is the unloaded backend's ``model`` attribute (``model_name`` for the ASR
    backends), else the ``model`` option. The revision is the ``revision`` option, else the
    commit the Hugging Face cache's ``refs/main`` points at, else None. Any failure gives
    ``(options["model"], None)``.
    """
    fallback = cfg.options.get("model")
    try:
        from fukikae.backends.loader import backend_class

        inst = backend_class(kind, cfg.backend)(dict(cfg.options), "cpu")
        model = next(
            (
                v
                for v in (getattr(inst, "model", None), getattr(inst, "model_name", None))
                if isinstance(v, str)
            ),
            fallback,
        )
        revision = cfg.options.get("revision")
        if revision is None and isinstance(model, str) and "/" in model:
            org, _, name = model.partition("/")
            ref = _hub_cache() / f"models--{org}--{name}" / "refs" / "main"
            if ref.is_file():
                revision = ref.read_text(encoding="utf-8").strip() or None
        return (
            None if model is None else str(model),
            None if revision is None else str(revision),
        )
    except Exception:  # noqa: BLE001 - identity is best effort; a row must still be written
        return (None if fallback is None else str(fallback), None)


def make_row(
    *,
    stage: str,
    backend: str,
    model: str | None,
    revision: str | None,
    options: Mapping[str, Any],
    language: str,
    genre: str,
    series: str,
    test_set: str,
    split: str,
    episodes: Sequence[str],
    metrics: Mapping[str, float | int | str],
    detail: Mapping[str, Any] | None = None,
    workspace: str | None = None,
    tag: str | None = None,
) -> LedgerRow:
    if stage not in STAGES:
        raise ValueError(f"unknown ledger stage {stage!r}; use one of {', '.join(STAGES)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; use one of {', '.join(SPLITS)}")
    opts = dict(options)
    return LedgerRow(
        schema=SCHEMA,
        run_id=uuid.uuid4().hex,
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=git_commit(),
        stage=stage,
        backend=backend,
        model=model,
        revision=revision,
        options_hash=options_hash(opts),
        options=opts,
        language=language,
        genre=genre,
        series=series,
        test_set=test_set,
        split=split,
        episodes=list(episodes),
        metrics=dict(metrics),
        detail=dict(detail or {}),
        workspace=workspace,
        tag=tag,
    )


def row_json(row: LedgerRow) -> str:
    """One row as a single JSON line (no trailing newline)."""
    return json.dumps(dataclasses.asdict(row), ensure_ascii=False, default=str)


def append_rows(rows: Sequence[LedgerRow], path: Path | None = None) -> Path:
    """Append `rows` under an exclusive lock in one write. A file whose last line was cut
    short gets a newline first, so the new rows never join the broken one."""
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(row_json(r) + "\n" for r in rows)
    with target.open("a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    data = "\n" + data
            f.write(data.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return target


def _parse(obj: Any) -> LedgerRow | str:
    """A row, or why the line is not one."""
    if not isinstance(obj, dict):
        return "not a ledger row"
    if obj.get("schema") != SCHEMA:
        return f"schema {obj.get('schema')}"
    missing = [k for k in _REQUIRED if k not in obj]
    unknown = sorted(set(obj) - set(_FIELDS))
    if missing or unknown:
        why = []
        if missing:
            why.append(f"missing {', '.join(missing)}")
        if unknown:
            why.append(f"unknown {', '.join(unknown)}")
        return f"not a ledger row ({'; '.join(why)})"
    if not isinstance(obj["metrics"], dict) or not isinstance(obj["episodes"], list):
        return "not a ledger row (bad metrics or episodes)"
    return LedgerRow(**obj)


def read_rows(path: Path | None = None) -> tuple[list[LedgerRow], list[str]]:
    """Every readable row, and a warning per skipped line (``results.jsonl:7: not JSON,
    skipped``). A missing ledger has no rows."""
    target = ledger_path(path)
    rows: list[LedgerRow] = []
    warnings: list[str] = []
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return rows, warnings
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"{target.name}:{n}: not JSON, skipped")
            continue
        parsed = _parse(obj)
        if isinstance(parsed, str):
            warnings.append(f"{target.name}:{n}: {parsed}, skipped")
        else:
            rows.append(parsed)
    return rows, warnings
