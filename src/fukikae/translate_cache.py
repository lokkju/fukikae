"""Translator answers by request content, so a rerun reuses what was translated (#41).

A request's key hashes everything the translator reads: the cues' speakers, source text,
budgets and marks, the previous lines, style notes, glossary, languages, mode and the strict
flag, plus a fingerprint of the backend (its name, model identity and output-affecting
options) and the stage's ``PROMPT_VERSION``. Cue ids are replaced by their position in the
window, so a line added earlier in the episode, which renumbers every cue after it, misses
only the windows whose content or context changed. Entries live in
``<workspace>/.fukikae/translate-cache/<key>.json`` as the window's renderings in order.

Only answers that passed validation and the language check are stored (the stage decides);
an unreadable entry counts as a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fukikae.artifacts import atomic_path
from fukikae.translation import Rendering, TranslateRequest

log = logging.getLogger(__name__)

# Bump when the key's layout or an entry's format changes.
CACHE_VERSION = 1


class TranslationCache:
    """``root`` None is a cache that never hits and stores nothing (``[translate] cache =
    false``)."""

    def __init__(self, root: Path | None, fingerprint: dict[str, Any]) -> None:
        self.root = root
        self.fingerprint = fingerprint
        self.hits = 0
        self.misses = 0

    def key(self, req: TranslateRequest) -> str:
        data = req.model_dump(mode="json")
        for i, cue in enumerate(data["cues"]):
            cue["id"] = str(i)
        payload = json.dumps(
            {"version": CACHE_VERSION, "backend": self.fingerprint, "request": data},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, req: TranslateRequest) -> Path | None:
        return None if self.root is None else self.root / f"{self.key(req)}.json"

    def get(self, req: TranslateRequest) -> dict[str, Rendering] | None:
        path = self._path(req)
        if path is None or not path.exists():
            self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            renderings = [Rendering.model_validate(r) for r in entry["renderings"]]
            if len(renderings) != len(req.cues):
                raise ValueError(f"{len(renderings)} renderings for {len(req.cues)} cues")
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as e:
            log.debug("translate cache: ignoring %s (%s)", path.name, e)
            self.misses += 1
            return None
        self.hits += 1
        return {c.id: r for c, r in zip(req.cues, renderings, strict=True)}

    def put(self, req: TranslateRequest, got: dict[str, Rendering]) -> None:
        path = self._path(req)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"renderings": [got[c.id].model_dump(mode="json") for c in req.cues]}
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
