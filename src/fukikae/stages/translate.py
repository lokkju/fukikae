"""Translate stage: dub cues with a speakable target-language line each.

Reads ``03-asr/transcript.json``, ``05-characters/assignments.json`` (with its ``turns``),
``04b-tag/tags.json`` when the tag stage is on, ``<workspace>/glossary.toml`` when present
(its ``[terms]`` table maps source terms to renderings) and the cards of the speaking
characters; writes ``06-translate/cues.<target>.json`` (a ``DubCues``).

Cues go to the translator in scene windows with the previous window's lines, the speakers'
``[style]`` notes and catchphrases, and the glossary. A window whose answer is malformed is
asked once more; a second failure leaves its cues with ``error`` and no ``text`` (tts skips
them, and the mix keeps the original vocals there). Cues that come back longer than their
budget are asked once more, together, with 85% of the budget; the shorter of the two
renderings is kept, and one still long is flagged (``over_budget``, ``overrun``) and left to
time fitting, never cut. A backend error (HTTP error, timeout) is retried once, then marks
only that window (``error = "translate: backend: ..."``), and the stage goes on; three such
windows in a row stop it. A prompt too large for the context is sent again without the
previous lines, then in halves.

``[translate] source``: ``auto`` and ``asr`` build cues from the transcript; ``subs`` is the
subtitle-first path, not built yet. Stage settings (``max_speed``, ``source``,
``subs_condense``) are not backend options.
"""

from __future__ import annotations

import logging
import math
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from fukikae import runctx
from fukikae.artifacts import read_json, write_json
from fukikae.backends.base import Translator
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import create
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.cues import build_asr_cues, is_marked, windows
from fukikae.gpu import device_string, free_mb, gpu_processes, wait_for_vram
from fukikae.models import Assignments, DubCue, DubCues, Tags, Transcript
from fukikae.registry.store import Registry
from fukikae.stages.language import LanguageResult
from fukikae.translation import (
    PromptTooLargeError,
    Rendering,
    RequestCue,
    StyleNote,
    TranslateBackendError,
    TranslateFormatError,
    TranslateRequest,
)
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)
_sleep = time.sleep

STAGE_KEYS = ("max_speed", "source", "subs_condense")
SOURCES = ("auto", "asr", "subs")
DEFAULT_MAX_SPEED = 1.25
RETRY_BUDGET = 0.85
# Windows in a row that fail on the backend (HTTP error, timeout) before the stage gives up:
# one bad window is marked and skipped, but a dead endpoint shouldn't mark a whole episode.
MAX_BACKEND_FAILURES = 3
# Bump when the prompt or the cue rules change what the stage writes for the same inputs.
PROMPT_VERSION = 3


class TranslateStage:
    name = "translate"

    def __init__(self, target_language: str | None = None, tags: bool = False) -> None:
        # Both come from the config the CLI resolved; inputs() has no config of its own.
        self.target_language = target_language
        self.tags = tags

    # -- configuration -------------------------------------------------------------------

    @staticmethod
    def _source_language(ep: Episode, cfg: Config) -> str:
        path = ep.work / "02-language" / "language.json"
        if path.exists():
            return read_json(path, LanguageResult).language
        if cfg.source_language:
            return cfg.source_language
        transcript = ep.work / "03-asr" / "transcript.json"
        if transcript.exists():
            return read_json(transcript, Transcript).language
        raise ConfigError("translate: the source language is unknown; run the language stage")

    def _settings(self, ep: Episode, cfg: Config) -> tuple[BackendConfig, dict[str, Any]]:
        src = self._source_language(ep, cfg)
        bc = cfg.resolve("translate", src, f"{src}-{cfg.target_language}")
        stage = {k: bc.options[k] for k in STAGE_KEYS if k in bc.options}
        options = {k: v for k, v in bc.options.items() if k not in STAGE_KEYS}
        max_speed = stage.get("max_speed", DEFAULT_MAX_SPEED)
        if isinstance(max_speed, bool) or not isinstance(max_speed, int | float) or max_speed <= 0:
            raise ConfigError(f"[translate] max_speed must be a positive number, not {max_speed!r}")
        source = stage.get("source", "auto")
        if source not in SOURCES:
            raise ConfigError(
                f"[translate] source must be one of {', '.join(SOURCES)}, not {source!r}"
            )
        settings = {
            "source_language": src,
            "max_speed": float(max_speed),
            "source": source,
            "subs_condense": bool(stage.get("subs_condense", True)),
        }
        return BackendConfig(bc.backend, options), settings

    # -- characters and cards ------------------------------------------------------------

    @staticmethod
    def _speaking(ep: Episode) -> set[int]:
        path = ep.work / "05-characters" / "assignments.json"
        if not path.exists():
            return set()
        asg = read_json(path, Assignments)
        return {
            a.character_id
            for a in asg.clusters
            if a.character_id is not None and a.status not in ("extra", "review")
        }

    @staticmethod
    def _cards(ep: Episode, characters: set[int], target: str) -> dict[int, str]:
        path = Workspace(ep.workspace_root).registry_path
        if not characters or not path.exists():
            return {}
        out: dict[int, str] = {}
        with Registry(path) as reg:
            for cid in sorted(characters):
                a = reg.card_for(cid, target)
                if a is not None and a.card_id is not None:
                    out[cid] = a.card_id
        return out

    def _target(self, ep: Episode) -> str:
        if self.target_language is not None:
            return self.target_language
        return Workspace(ep.workspace_root).config.target_language

    def _use_tags(self) -> bool:
        return self.tags

    # -- Stage protocol --------------------------------------------------------------------

    def inputs(self, ep: Episode) -> list[Path]:
        paths = [
            ep.work / "03-asr" / "transcript.json",
            ep.work / "05-characters" / "assignments.json",
        ]
        if self._use_tags():
            paths.append(ep.work / "04b-tag" / "tags.json")
        paths.append(ep.workspace_root / "glossary.toml")
        store = CardStore(ep.workspace_root)
        cards = self._cards(ep, self._speaking(ep), self._target(ep))
        paths += [store.card_path(card) for _, card in sorted(cards.items())]
        return paths

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "06-translate" / f"cues.{self._target(ep)}.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        bc, settings = self._settings(ep, cfg)
        cards = self._cards(ep, self._speaking(ep), cfg.target_language)
        return {
            "backend": bc.backend,
            "options": bc.options,
            "identity": identity_key("translate", bc),
            "target_language": cfg.target_language,
            "max_speed": settings["max_speed"],
            "source": settings["source"],
            "cards": {str(k): v for k, v in sorted(cards.items())},
            "prompt_version": PROMPT_VERSION,
        }

    # -- run -------------------------------------------------------------------------------

    @staticmethod
    def _glossary(ep: Episode) -> dict[str, str]:
        path = ep.workspace_root / "glossary.toml"
        if not path.exists():
            return {}
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e
        terms = data.get("terms", {})
        if not isinstance(terms, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in terms.items()
        ):
            raise ConfigError(f'{path}: [terms] must map strings to strings ("source" = "target")')
        return dict(terms)

    @staticmethod
    def _styles(ep: Episode, cards: dict[int, str], names: dict[int, str]) -> dict[int, StyleNote]:
        store = CardStore(ep.workspace_root)
        out: dict[int, StyleNote] = {}
        for cid, card_id in cards.items():
            style = store.get(card_id).style
            if not style:
                continue
            out[cid] = StyleNote(
                register=style.get("register", ""),
                address=style.get("address", ""),
                tics=style.get("tics", ""),
                notes=style.get("notes", ""),
                # a derived card's catchphrase has an empty target: no fixed rendering
                catchphrases=[
                    (p["source"], p["target"])
                    for p in style.get("catchphrases", [])
                    if p["source"].strip() and p["target"].strip()
                ],
            )
            log.debug("style for %s from card %s", names.get(cid, cid), card_id)
        return out

    def run(self, ep: Episode, cfg: Config) -> None:
        bc, settings = self._settings(ep, cfg)
        if settings["source"] == "subs":
            raise ConfigError(
                '[translate] source = "subs" (the subtitle-first path) is not built yet; '
                'use "asr" or "auto"'
            )
        target = cfg.target_language
        transcript_path = ep.work / "03-asr" / "transcript.json"
        assignments_path = ep.work / "05-characters" / "assignments.json"
        transcript = read_json(transcript_path, Transcript)
        assignments = read_json(assignments_path, Assignments)
        if assignments.turns is None:
            raise ConfigError(
                f"{assignments_path} has no turns (written by an older fukikae); "
                "re-run the characters stage"
            )
        tags_path = ep.work / "04b-tag" / "tags.json"
        tags = read_json(tags_path, Tags) if self._use_tags() and tags_path.exists() else None

        registry = Workspace(ep.workspace_root).registry_path
        names: dict[int, str] = {}
        if registry.exists():
            with Registry(registry) as reg:
                names = {c.id: c.name for c in reg.characters() if c.name}
        cues = build_asr_cues(transcript, assignments, tags, names, target, settings["max_speed"])
        speaking = {c.character_id for c in cues if c.character_id is not None}
        cards = self._cards(ep, speaking, target)
        styles = self._styles(ep, cards, names)
        glossary = self._glossary(ep)

        todo = [c for c in cues if c.skip is None]
        started = time.monotonic()
        if todo:
            options = {**bc.options, "gpu_index": cfg.gpu_device}
            device = device_string(cfg)
            backend = create("translate", BackendConfig(bc.backend, options), device)
            # Only a managed server takes GPU memory here; an external endpoint takes none.
            managed = getattr(backend, "mode", "managed") == "managed"
            if device.startswith("cuda") and managed:
                leftover = [
                    (pid, name) for pid, name in gpu_processes(cfg.gpu_device) if "llama" in name
                ]
                if leftover:
                    listed = ", ".join(f"pid {pid} ({name})" for pid, name in leftover)
                    raise RuntimeError(
                        f"GPU {cfg.gpu_device} already runs {listed}, probably left over from an "
                        "earlier run; stop it (`kill <pid>`) and run again"
                    )
            if device.startswith("cuda") and managed and backend.vram_mb > 0:
                wait_for_vram(
                    cfg.gpu_device,
                    backend.vram_mb,
                    runctx.vram_timeout_s,
                    query=free_mb,
                    sleep=_sleep,
                )
            try:
                backend.load()
                self._translate_all(
                    cast(Translator, backend),
                    todo,
                    styles,
                    glossary,
                    settings["source_language"],
                    target,
                )
            finally:
                backend.unload()
        for c in todo:
            if c.text is not None:
                c.overrun = max(0, len(c.text) - c.budget)
                c.over_budget = c.overrun > 0
        over = sum(1 for c in todo if c.over_budget)
        failed = sum(1 for c in todo if c.error is not None)
        log.info(
            "translate %s: %d cue(s), %d skipped, %d over budget, %d failed, %.0f s",
            ep.id, len(cues), len(cues) - len(todo), over, failed, time.monotonic() - started,
        )  # fmt: skip
        out = ep.work / "06-translate" / f"cues.{target}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            out,
            DubCues(
                episode=ep.id,
                source_language=settings["source_language"],
                target_language=target,
                path="asr",
                cues=cues,
            ),
        )

    @staticmethod
    def _request(
        window: list[DubCue],
        styles: dict[int, StyleNote],
        glossary: dict[str, str],
        previous: list[tuple[str, str]],
        src: str,
        target: str,
        budgets: dict[str, int] | None = None,
    ) -> TranslateRequest:
        speakers = {c.speaker: styles[c.character_id] for c in window if c.character_id in styles}
        return TranslateRequest(
            source_language=src,
            target_language=target,
            mode="translate",
            cues=[
                RequestCue(
                    id=c.id,
                    speaker=c.speaker,
                    text=c.source_text,
                    budget=(budgets or {}).get(c.id, c.budget),
                    marked=is_marked(c),
                )
                for c in window
            ],
            previous=previous,
            styles=speakers,
            glossary={k: v for k, v in glossary.items() if any(k in c.source_text for c in window)},
        )

    @staticmethod
    def _ask(backend: Translator, req: TranslateRequest) -> dict[str, Rendering]:
        """One retry on a format or backend error; the second error propagates."""
        try:
            return backend.translate(req)
        except (TranslateFormatError, TranslateBackendError) as e:
            log.warning("translate: %s; asking once more", e)
            return backend.translate(req)

    def _render(
        self,
        backend: Translator,
        cues: list[DubCue],
        previous: list[tuple[str, str]],
        styles: dict[int, StyleNote],
        glossary: dict[str, str],
        src: str,
        target: str,
    ) -> None:
        """Translate ``cues``, or mark them with ``error``. A prompt too large for the context
        is sent again without the previous lines, then split in halves, down to one cue."""
        req = self._request(cues, styles, glossary, previous, src, target)
        try:
            got = self._ask(backend, req)
        except PromptTooLargeError as e:
            if previous:
                self._render(backend, cues, [], styles, glossary, src, target)
            elif len(cues) > 1:
                half = len(cues) // 2
                for part in (cues[:half], cues[half:]):
                    self._render(backend, part, [], styles, glossary, src, target)
            else:
                cues[0].error = f"translate: {e}"
            return
        except TranslateFormatError as e:
            for c in cues:
                c.error = f"translate: {e}"
            return
        except TranslateBackendError as e:
            for c in cues:
                c.error = f"translate: backend: {e}"
            return
        for c in cues:
            r = got[c.id]
            c.literal, c.text = r.literal, r.text
            c.delivery.direction = r.direction
        long = [c for c in cues if len(cast(str, c.text)) > c.budget]
        if not long:
            return
        tight = {c.id: math.floor(c.budget * RETRY_BUDGET) for c in long}
        retry = self._request(long, styles, glossary, previous, src, target, tight)
        try:
            again = backend.translate(retry)
        except (TranslateFormatError, TranslateBackendError, PromptTooLargeError) as e:
            log.warning("translate: over-budget retry failed (%s); keeping the first", e)
            again = {}
        for c in long:
            r = again.get(c.id)
            # keep the shorter rendering; anything still long is left to time fitting
            if r is not None and len(r.text) < len(cast(str, c.text)):
                c.literal, c.text = r.literal, r.text
                c.delivery.direction = r.direction or c.delivery.direction

    def _translate_all(
        self,
        backend: Translator,
        todo: list[DubCue],
        styles: dict[int, StyleNote],
        glossary: dict[str, str],
        src: str,
        target: str,
    ) -> None:
        previous: list[tuple[str, str]] = []
        in_a_row = 0
        for window in windows(todo):
            self._render(backend, window, previous, styles, glossary, src, target)
            if all((c.error or "").startswith("translate: backend:") for c in window):
                in_a_row += 1
                if in_a_row >= MAX_BACKEND_FAILURES:
                    raise TranslateBackendError(
                        f"{in_a_row} windows in a row failed on the backend; giving up "
                        f"(last: {window[-1].error})"
                    )
            else:
                in_a_row = 0
            done = [c for c in window if c.text is not None]
            previous = (
                [(c.source_text, cast(str, c.text)) for c in done]
                if len(done) == len(window)
                else []
            )
