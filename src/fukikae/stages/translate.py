"""Translate stage: dub cues with a speakable target-language line each.

Reads ``03-asr/transcript.json``, ``05-characters/assignments.json`` (with its ``turns``),
``04b-tag/tags.json`` when the tag stage is on, ``<workspace>/glossary.toml`` when present
(its ``[terms]`` table maps source terms to renderings) and the cards of the speaking
characters (the key holds a hash of each one's ``[style]``, not the card id, ruling CD6, so a
voice-only change never re-translates); writes ``06-translate/cues.<target>.json`` (a ``DubCues``).

Cues go to the translator in scene windows with the previous window's lines, the speakers'
``[style]`` notes and catchphrases, and the glossary. A window whose answer is malformed is
asked once more; a second failure leaves its cues with ``error`` and no ``text`` (tts skips
them, and the mix keeps the original vocals there). Cues that come back longer than their
budget are asked once more, together, with 85% of the budget (a budget is what the ``[tts]``
backend's voice says in the slot at ``max_speed``: its ``speech_cps``, else the language's
typical rate); the shorter of the two
renderings is kept, and one still long is flagged (``over_budget``, ``overrun``) and left to
time fitting, never cut. A backend error (HTTP error, timeout) is retried once, then marks
only that window (``error = "translate: backend: ..."``), and the stage goes on; three such
windows in a row stop it. A prompt too large for the context is sent again without the
previous lines, then in halves.

``[translate] source``: ``asr`` builds cues from the transcript; ``subs`` is the subtitle-first
path: cues come from target-language subtitles (``media.find_target_subtitles``: ``[subs] file``,
whose ``{episode}`` is replaced by the episode id, then ``[subs] track``, a sidecar, an embedded
track), each spoken by the diarize cluster that overlaps it most, and a cue within its budget
is spoken verbatim. Only a cue over budget goes to the translator, in condense mode (English to
shorter English), unless ``subs_condense = false``; a condensed line that fails or comes back no
shorter leaves the subtitle line in place, flagged over budget for time fitting. Signs, songs
and SDH notes are filtered by ``subs_filter`` (default all three rules). ``auto`` takes the
subtitle path when subtitles are found, else the ASR path. Stage settings (``max_speed``,
``source``, ``subs_condense``, ``subs_filter``) are not backend options.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from fukikae import media, runctx
from fukikae.artifacts import read_json, write_json
from fukikae.backends.base import Translator
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import create
from fukikae.cards.card import CardError
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.cues import assign_subtitle_speakers, build_asr_cues, is_marked, windows
from fukikae.gpu import device_string, free_mb, gpu_processes, wait_for_vram
from fukikae.models import Assignments, DubCue, DubCues, Tags, Transcript
from fukikae.registry.store import Registry
from fukikae.stages.language import LanguageResult
from fukikae.subtitles import RULES, read_cues
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

STAGE_KEYS = ("max_speed", "source", "subs_condense", "subs_filter")
SOURCES = ("auto", "asr", "subs")
DEFAULT_MAX_SPEED = 1.25
RETRY_BUDGET = 0.85
# Windows in a row that fail on the backend (HTTP error, timeout) before the stage gives up:
# one bad window is marked and skipped, but a dead endpoint shouldn't mark a whole episode.
MAX_BACKEND_FAILURES = 3
# Bump when the prompt or the cue rules change what the stage writes for the same inputs.
PROMPT_VERSION = 3


def speech_cps(cfg: Config) -> float:
    """Characters per second the target TTS backend speaks ``cfg.target_language`` at
    (``Synthesizer.speech_cps``, ruling CD7), so a budget is what that voice can say in the
    slot. Without a ``[tts]`` backend, or one that declares no figure for the language, the
    language's typical rate (``cues.CPS``)."""
    from fukikae.backends import loader
    from fukikae.cues import CPS, DEFAULT_CPS

    fallback = CPS.get(cfg.target_language, DEFAULT_CPS)
    try:
        name = cfg.resolve("tts").backend
    except ConfigError:
        return fallback
    try:
        cls = loader.backend_class("tts", name)
    except (ConfigError, loader.BackendUnavailable) as e:
        log.warning("translate: budgets use %s cps, not the tts backend's: %s", fallback, e)
        return fallback
    figure = getattr(cls, "speech_cps", {}).get(cfg.target_language)
    return float(figure) if figure else fallback


class TranslateStage:
    name = "translate"

    def __init__(
        self, target_language: str | None = None, tags: bool = False, cfg: Config | None = None
    ) -> None:
        # All three come from the config the CLI resolved; inputs() has no config of its own.
        self.target_language = target_language
        self.tags = tags
        self.cfg = cfg
        self._probes: dict[tuple[str, int, int], media.ProbeResult] = {}

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
        rules = stage.get("subs_filter", list(RULES))
        if not isinstance(rules, list) or not all(r in RULES for r in rules):
            raise ConfigError(
                f"[translate] subs_filter must be a list of {', '.join(RULES)}, not {rules!r}"
            )
        settings = {
            "cps": speech_cps(cfg),
            "source_language": src,
            "max_speed": float(max_speed),
            "source": source,
            "subs_condense": bool(stage.get("subs_condense", True)),
            "subs_filter": [r for r in RULES if r in rules],
        }
        return BackendConfig(bc.backend, options), settings

    # -- subtitles -----------------------------------------------------------------------

    def _probe(self, ep: Episode) -> media.ProbeResult:
        if not ep.source.is_file():
            return media.ProbeResult()
        st = ep.source.stat()
        key = (str(ep.source.resolve()), st.st_size, st.st_mtime_ns)
        if key not in self._probes:
            self._probes[key] = media.probe(ep.source)
        return self._probes[key]

    def _subtitles(
        self, ep: Episode, cfg: Config, source: str
    ) -> tuple[Path | None, media.SubTrack | None]:
        """The target-language subtitles the subtitle path reads, or (None, None) for the ASR
        path; ConfigError when ``source = "subs"`` finds none or ``[subs]`` names a missing
        file or track."""
        if source == "asr":
            return None, None
        configured = dict(cfg.raw.get("subs", {}) or {})
        if configured.get("file"):
            configured["file"] = str(configured["file"]).replace("{episode}", ep.id)
        try:
            found = media.find_target_subtitles(
                ep.source, self._probe(ep), cfg.target_language, configured
            )
        except media.MediaError as e:
            raise ConfigError(str(e)) from e
        if source == "subs" and found == (None, None):
            raise ConfigError(
                f'[translate] source = "subs", but there are no {cfg.target_language} '
                f"subtitles for {ep.id}: no [subs] file or track, no sidecar "
                f"{ep.source.stem}.{cfg.target_language}.srt|ass, no embedded track"
            )
        return found

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

    @staticmethod
    def _style_hashes(ep: Episode, cards: dict[int, str]) -> dict[int, str]:
        """Character -> sha256 of its card's [style] (catchphrases included), the only part of
        a card translate reads (ruling CD6): a voice-only change never re-translates, and a
        style-less card (the stock cards tts assigns) contributes nothing. A card that can't be
        read contributes its id, and run() reports it."""
        store = CardStore(ep.workspace_root)
        out: dict[int, str] = {}
        for cid, card in cards.items():
            try:
                style = store.get(card).style
            except CardError:
                out[cid] = f"unreadable:{card}"
                continue
            if style:
                text = json.dumps(style, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                out[cid] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return out

    def _target(self, ep: Episode) -> str:
        if self.target_language is not None:
            return self.target_language
        return Workspace(ep.workspace_root).config.target_language

    def _use_tags(self) -> bool:
        return self.tags

    # -- Stage protocol --------------------------------------------------------------------

    def _config(self, ep: Episode) -> Config:
        return self.cfg if self.cfg is not None else Workspace(ep.workspace_root).config

    def _subtitle_input(self, ep: Episode) -> list[Path]:
        """The subtitle file (or, for an embedded track, the source) as a stage input; nothing
        when that can't be worked out here, since run() raises the same error with context."""
        cfg = self._config(ep)
        try:
            source = self._settings(ep, cfg)[1]["source"]
            file, track = self._subtitles(ep, cfg, source)
        except (ConfigError, media.MediaError):
            return []
        if file is not None:
            return [file]
        return [ep.source] if track is not None else []

    def inputs(self, ep: Episode) -> list[Path]:
        paths = [
            ep.work / "03-asr" / "transcript.json",
            ep.work / "05-characters" / "assignments.json",
        ]
        if self._use_tags():
            paths.append(ep.work / "04b-tag" / "tags.json")
        paths.append(ep.workspace_root / "glossary.toml")
        # Card files are not inputs: the key holds a hash of each speaking character's style.
        return paths + self._subtitle_input(ep)

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "06-translate" / f"cues.{self._target(ep)}.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        bc, settings = self._settings(ep, cfg)
        styles = self._style_hashes(ep, self._cards(ep, self._speaking(ep), cfg.target_language))
        key: dict[str, Any] = {
            "backend": bc.backend,
            "options": bc.options,
            "identity": identity_key("translate", bc),
            "target_language": cfg.target_language,
            "max_speed": settings["max_speed"],
            "source": settings["source"],
            "styles": {str(k): v for k, v in sorted(styles.items())},
            "cps": settings["cps"],
            "prompt_version": PROMPT_VERSION,
        }
        file, track = self._subtitles(ep, cfg, settings["source"])
        if file is not None or track is not None:  # the ASR path's key stays as it was
            key |= {
                "path": "subs",
                "subs_file": None if file is None else str(file),
                "subs_track": None if track is None else track.index,
                "subs_condense": settings["subs_condense"],
                "subs_filter": settings["subs_filter"],
            }
        return key

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
        target = cfg.target_language
        sub_file, sub_track = self._subtitles(ep, cfg, settings["source"])
        subs = sub_file is not None or sub_track is not None
        transcript_path = ep.work / "03-asr" / "transcript.json"
        assignments_path = ep.work / "05-characters" / "assignments.json"
        # The subtitle path uses the transcript only to break ties between two speakers.
        transcript = (
            read_json(transcript_path, Transcript) if not subs or transcript_path.exists() else None
        )
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
        if subs:
            if sub_file is None:
                sub_file = media.extract_subtitle(
                    ep.source,
                    cast(media.SubTrack, sub_track),
                    ep.work / "06-translate" / f"subs.{target}",
                )
            cues = assign_subtitle_speakers(
                read_cues(sub_file),
                assignments.turns,
                assignments,
                transcript,
                tags,
                names,
                target,
                settings["max_speed"],
                settings["subs_filter"],
                cps=settings["cps"],
            )
        else:
            cues = build_asr_cues(
                cast(Transcript, transcript),
                assignments,
                tags,
                names,
                target,
                settings["max_speed"],
                cps=settings["cps"],
            )
        speaking = {c.character_id for c in cues if c.character_id is not None}
        cards = self._cards(ep, speaking, target)
        styles = self._styles(ep, cards, names)

        spoken = [c for c in cues if c.skip is None]
        if not subs:
            todo, mode, src = spoken, "translate", settings["source_language"]
            glossary = self._glossary(ep)
        else:
            long = [c for c in spoken if len(cast(str, c.text)) > c.budget]
            todo = long if settings["subs_condense"] else []
            # condense is English to shorter English: the glossary maps source-language terms
            mode, src, glossary = "condense", target, {}
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
                    cast(Translator, backend), todo, styles, glossary, src, target, mode
                )
            finally:
                backend.unload()
        if subs:
            self._keep_subtitle_lines(todo)
        for c in spoken:
            if c.text is not None:
                c.overrun = max(0, len(c.text) - c.budget)
                c.over_budget = c.overrun > 0
        over = sum(1 for c in spoken if c.over_budget)
        failed = sum(1 for c in spoken if c.error is not None)
        if subs:
            condensed = sum(1 for c in todo if c.text != c.source_text)
            log.info(
                "translate %s (subtitles): %d cue(s), %d skipped, %d verbatim, %d condensed, "
                "%d over budget, %.0f s",
                ep.id, len(cues), len(cues) - len(spoken), len(spoken) - condensed, condensed,
                over, time.monotonic() - started,
            )  # fmt: skip
        else:
            log.info(
                "translate %s: %d cue(s), %d skipped, %d over budget, %d failed, %.0f s",
                ep.id, len(cues), len(cues) - len(spoken), over, failed,
                time.monotonic() - started,
            )  # fmt: skip
        out = ep.work / "06-translate" / f"cues.{target}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            out,
            DubCues(
                episode=ep.id,
                source_language=settings["source_language"],
                target_language=target,
                path="subs" if subs else "asr",
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
        mode: str = "translate",
    ) -> TranslateRequest:
        speakers = {c.speaker: styles[c.character_id] for c in window if c.character_id in styles}
        return TranslateRequest(
            source_language=src,
            target_language=target,
            mode=cast(Any, mode),
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
        mode: str = "translate",
    ) -> None:
        """Translate ``cues``, or mark them with ``error``. A prompt too large for the context
        is sent again without the previous lines, then split in halves, down to one cue."""
        req = self._request(cues, styles, glossary, previous, src, target, mode=mode)
        try:
            got = self._ask(backend, req)
        except PromptTooLargeError as e:
            if previous:
                self._render(backend, cues, [], styles, glossary, src, target, mode)
            elif len(cues) > 1:
                half = len(cues) // 2
                for part in (cues[:half], cues[half:]):
                    self._render(backend, part, [], styles, glossary, src, target, mode)
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
        retry = self._request(long, styles, glossary, previous, src, target, tight, mode)
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
        mode: str = "translate",
    ) -> None:
        previous: list[tuple[str, str]] = []
        in_a_row = 0
        for window in windows(todo):
            self._render(backend, window, previous, styles, glossary, src, target, mode)
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

    @staticmethod
    def _keep_subtitle_lines(condensed: list[DubCue]) -> None:
        """A condensed cue that failed, or came back no shorter than the subtitle, speaks the
        subtitle line instead (a human line left long for time fitting beats a lost one), and
        ``note`` says why."""
        kept = 0
        for c in condensed:
            if c.text is None or c.error is not None:
                reason = (c.error or "no rendering").removeprefix("translate: ")
                log.warning("translate: condensing %s failed (%s); speaking the subtitle line",
                            c.id, reason)  # fmt: skip
                c.note = f"condense failed: {reason}"
            elif len(c.text) >= len(c.source_text):
                c.note = "condense not shorter"
            else:
                continue
            c.text, c.literal, c.error = c.source_text, None, None
            kept += 1
        if kept:
            log.info("translate: %d over-budget subtitle line(s) kept as written", kept)
