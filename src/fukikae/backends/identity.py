"""The model a backend config resolves to, for the results ledger and for cache keys.

Ruling GT6: every embedding or stage cache key includes the resolved model identity (the
model file or name, plus the revision when known), because a backend's default model can
change with no change to the config's options. This lives outside ``fukikae.eval`` so the
pipeline can use it (R1: pipeline code never imports ``eval``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fukikae.config import BackendConfig

log = logging.getLogger(__name__)
_warned: set[tuple[str, str]] = set()


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_identity(kind: str, cfg: BackendConfig) -> tuple[str | None, str | None]:
    """(model, revision) for a backend config, without loading the model.

    The model is the unloaded backend's ``model`` attribute (``model_name`` for the ASR
    backends), else the ``model`` option. The revision is the ``revision`` option, else
    the backend's pinned ``revision`` attribute, else the
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
        revision = cfg.options.get("revision") or getattr(inst, "revision", None)
        # B3: a worker backend's environment is part of what renders (its own uv.lock), so a
        # worker env bump changes the identity; the card pin (model_revision) is unchanged
        lock_revision = getattr(inst, "lock_revision", None)
        locked = lock_revision() if callable(lock_revision) else None
        if locked and revision is not None:
            revision = f"{revision}+lock:{locked}"
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


def identity_key(kind: str, cfg: BackendConfig) -> dict[str, Any]:
    """A deterministic, JSON-ready identity for a cache key: ``{"model", "revision"}``.

    The model is what `model_identity` finds; failing that, the backend class's declared
    default (``default_file`` or its module's ``DEFAULT_MODEL``); failing that,
    ``unresolved:<kind>:<backend>`` with a warning (once per backend). The last form can't
    equal a real model's name, and it stays the same between runs, so a stage with an
    unknowable model doesn't re-run every time.
    """
    model, revision = model_identity(kind, cfg)
    if model is None:
        try:
            from fukikae.backends.loader import backend_class

            cls = backend_class(kind, cfg.backend)
            found = getattr(cls, "default_file", None) or getattr(
                sys.modules.get(cls.__module__), "DEFAULT_MODEL", None
            )
            model = None if found is None else str(found)
        except Exception:  # noqa: BLE001 - an unknown backend is reported by the stage itself
            model = None
    if model is None:
        model = f"unresolved:{kind}:{cfg.backend}"
        if (kind, cfg.backend) not in _warned:
            _warned.add((kind, cfg.backend))
            log.warning(
                "cannot tell which model the %s backend %r uses; its cache key says %r",
                kind,
                cfg.backend,
                model,
            )
    return {"model": model, "revision": revision}
