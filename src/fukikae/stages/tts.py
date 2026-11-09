"""TTS stage: one wav per spoken cue, from the speaking character's card.

Reads ``06-translate/cues.<target>.json``, ``01-separate/vocals.wav`` and the card of every
character with a cue; writes ``07-tts/<cue-id>.wav`` and ``07-tts/plan.json`` (a ``TtsPlan``,
which holds nothing that depends on the run, so the same audio gives the same bytes). What a run
did (renders, cache hits, cards it assigned) goes to ``07-tts/run.json``, which is not an
output; each automatic assignment is also appended to ``07-tts/assignments.log`` and kept as
the history row's note.

Before the stage's key is computed (``prepare``, called by the job runner) and again in
``run``, each character with a cue to speak and no card for the target language gets
one (``cards.auto.ensure_assignments``: a gender-matched stock voice of the ``[tts]`` backend,
``by = "auto"``, unlocked; an existing card is never replaced). Cues with no character (extras,
and clusters the characters stage left for review) speak with a stock voice from the extras
pools, one per label, fixed by first appearance in the episode. Every card is then checked (it
exists, its backend can honor it at the model revision it names, its sidecars are present), so a
broken card stops the stage, naming the character, before any model loads.

Renders are cached per cue in ``<workspace>/.fukikae/tts-cache/<key>.wav`` and hard-linked (or
copied) into ``07-tts``. The key (``cue_key``) hashes the card id, the text, the part of the
delivery record the backend takes, the backend name, its options and its model identity
(GT6), so swapping one character's card re-renders only that character's cues, and a cached
key is never rendered again. Each render's leading and trailing silence is trimmed to 30 ms
(Kokoro pads every line by about 0.3 s, which the rate fit would otherwise count), then
``[voice.post] gain_db`` is applied, both before caching. Cues with
``skip``, ``error`` or no text are listed in the plan and not rendered. Cues are grouped by
their card's backend; each backend loads once (only when something isn't cached), renders and
unloads before the next.

Rate fit: when a render is longer than its cue and the backend honors ``rate``, it is rendered
once more at ``rate * min(seconds / slot, max_speedup)``, cached under a key derived from the
first and the factor. ``max_speedup`` is ``[timefit] max_speedup`` (default 1.25), the budget
rate and time-stretch share. A backend with the ``duration`` capability gets the slot as its
target length instead.

Delivery: each cue passes what ``capabilities_for(card.voice.source)`` allows, scaled by the
card's ``emotion_strength``: ``emotion_audio`` (the cue's ``delivery.reference`` span cut from
``01-separate/vocals.wav`` into ``07-tts/ref/<cue-id>.wav``; the key hashes those samples, so
new vocals under the same span render again), ``emotion_vector`` (the tag label at its
confidence), ``instruction`` (``delivery.direction``) and ``target_seconds`` (the slot, for
``duration``). ``emotion_strength = 0`` passes none of the first three. ``plan.json`` records
``passed`` per cue and ``neutral_only`` when none of them was passed; the stage logs how many
cues rendered neutral-only.

``[tts]`` names the backend for automatic and extras cards; its options go to that backend
only (a card on another backend renders with that backend's defaults). ``device`` is a stage
setting (``cpu`` or ``cuda``; default cuda when the backend can use it), not an option, and stays
out of every key: the device doesn't change which voice a card is. ``[tts.gender]``
(``male_below``, ``female_above`` in Hz) sets the bands of the automatic gender estimate.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from fukikae import runctx
from fukikae.artifacts import atomic_path, read_json, write_json
from fukikae.backends import loader
from fukikae.backends.base import Synthesizer
from fukikae.backends.identity import identity_key
from fukikae.cards.auto import character_spans, ensure_assignments, extras_voices
from fukikae.cards.card import Card, CardError, validate_for_backend
from fukikae.cards.stock import check_stock_voice, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.models import (
    Assignments,
    Delivery,
    DubCue,
    DubCues,
    TtsPlan,
    TtsPlanCue,
    TtsRun,
)
from fukikae.registry.store import Registry
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)
_sleep = time.sleep

# Bump when the stage changes what it renders for the same inputs.
RENDER_VERSION = 2  # 2: leading and trailing silence trimmed (ruling CD7)
STAGE_KEYS = ("device", "gender")
# Silence trimming: 10 ms frames quieter than TRIM_DB below the loudest frame are silence;
# TRIM_KEEP_S of it is kept at each end.
TRIM_FRAME_S = 0.010
TRIM_DB = 40.0
TRIM_KEEP_S = 0.030
DELIVERY_CAPS = frozenset({"emotion_audio", "emotion_vector", "instruction"})
# A source span shorter than this is too short to carry a delivery; no emotion_audio then.
MIN_REFERENCE_S = 0.1
# The delivery fields each capability hands the backend.
_DELIVERY_BY_CAP = {
    "emotion_audio": ("reference",),
    "emotion_vector": ("emotion", "emotion_confidence"),
    "instruction": ("direction",),
}


def max_speedup(cfg: Config) -> float:
    """``[timefit] max_speedup``, the workspace's over the profile's; 1.25 when unset (read by
    ``fukikae.timefit.fit_settings``, the one reader of ``[timefit]``)."""
    from fukikae.timefit import fit_settings

    return fit_settings(cfg).max_speedup


def delivery_inputs(
    delivery: Delivery, capabilities: frozenset[str] | None, slot: float | None = None
) -> dict[str, Any]:
    """The part of a cue's delivery a backend with ``capabilities`` takes (all of it when
    ``capabilities`` is None), plus the slot for a ``duration`` backend."""
    d = delivery.model_dump(mode="json")
    if capabilities is None:
        return d
    out = {k: d[k] for cap in sorted(capabilities) for k in _DELIVERY_BY_CAP.get(cap, ())}
    if "duration" in capabilities and slot is not None:
        out["target_seconds"] = round(slot, 4)
    return out


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cue_key(
    card_id: str,
    text: str,
    delivery: Delivery,
    backend_cfg: BackendConfig,
    *,
    capabilities: frozenset[str] | None = None,
    identity: dict[str, Any] | None = None,
    slot: float | None = None,
    reference_audio: str | None = None,
) -> str:
    """First 16 hex of the sha256 of the card id, text, delivery (only what a backend with
    ``capabilities`` receives, when given), backend config and model identity, plus the digest
    of the emotion reference's samples when one is passed."""
    payload: dict[str, Any] = {
        "card": card_id,
        "text": text,
        "delivery": delivery_inputs(delivery, capabilities, slot),
        "backend": backend_cfg.backend,
        "options": backend_cfg.options,
        "identity": identity,
        "version": RENDER_VERSION,
    }
    if reference_audio is not None:
        payload["reference_audio"] = reference_audio
    return _hash(payload)


def span_digest(vocals: Path, span: tuple[float, float]) -> str:
    """First 16 hex of the sha256 of the samples of ``span`` in ``vocals``."""
    from scipy.io import wavfile

    rate, data = wavfile.read(vocals, mmap=True)
    seg = np.ascontiguousarray(data[max(0, int(span[0] * rate)) : max(0, int(span[1] * rate))])
    return hashlib.sha256(f"{rate}:{seg.dtype.str}:".encode() + seg.tobytes()).hexdigest()[:16]


def fitted_key(key: str, factor: float) -> str:
    return _hash({"render": key, "rate_factor": round(factor, 4)})


def wav_seconds(path: Path) -> float:
    from scipy.io import wavfile

    rate, data = wavfile.read(path, mmap=True)
    return len(data) / rate


def trim_silence(path: Path) -> None:
    """Cut leading and trailing silence from the wav in place, keeping ``TRIM_KEEP_S`` at each
    end and the sample format as written. A render with no sound is left whole."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = max(1, int(TRIM_FRAME_S * rate))
    frames = len(x) // n
    if frames == 0:
        return
    rms = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    loudest = rms.max()
    if loudest <= 0:
        return
    loud = np.flatnonzero(rms >= loudest * 10 ** (-TRIM_DB / 20))
    keep = int(TRIM_KEEP_S * rate)
    start = max(0, loud[0] * n - keep)
    end = min(len(x), (loud[-1] + 1) * n + keep)
    if start > 0 or end < len(x):
        wavfile.write(path, rate, data[start:end])


def _apply_gain(path: Path, gain_db: float) -> None:
    """Scale the wav in place by ``gain_db``, written as float32 so nothing clips."""
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    x = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        x /= float(np.iinfo(data.dtype).max) + 1.0
    wavfile.write(path, rate, (x * np.float32(10 ** (gain_db / 20))).astype(np.float32))


@dataclass(frozen=True)
class RenderJob:
    """Everything one cue's render depends on (``prepare_cue``): the card, its backend class
    and config, the cache key, the slot and the delivery fields the backend is handed."""

    cue: DubCue
    card_id: str
    card: Card
    cls: type[Synthesizer]
    backend_cfg: BackendConfig
    key: str
    slot: float
    passed: dict[str, Any]
    vocals: Path | None = None  # the emotion_audio source


class LazySynth:
    """One backend, created and loaded on first use, unloaded by ``close``."""

    def __init__(self, bc: BackendConfig, device: str, cfg: Config) -> None:
        self.bc, self.device, self.cfg = bc, device, cfg
        self.inst: Synthesizer | None = None

    def get(self) -> Synthesizer:
        if self.inst is None:
            inst = cast(Synthesizer, loader.create("tts", self.bc, self.device))
            if self.device.startswith("cuda"):
                wait_for_vram(
                    self.cfg.gpu_device,
                    inst.vram_mb,
                    runctx.vram_timeout_s,
                    query=free_mb,
                    sleep=_sleep,
                )
            self.inst = inst
            inst.load()
        return self.inst

    def close(self) -> None:
        if self.inst is not None:
            self.inst.unload()
            self.inst = None


# -- the render path, shared by the stage and `eval tts` (ruling CD12) -------------------------


def stage_settings(cfg: Config) -> tuple[BackendConfig, str | None]:
    """``[tts]`` as (backend config without the stage keys, device preference)."""
    bc = cfg.resolve("tts")
    device = bc.options.get("device")
    if device is not None and device not in ("cpu", "cuda"):
        raise ConfigError(f'[tts] device must be "cpu" or "cuda", not {device!r}')
    options = {k: v for k, v in bc.options.items() if k not in STAGE_KEYS}
    return BackendConfig(bc.backend, options), device


def backend_config(cfg: Config, backend: str) -> BackendConfig:
    """The config a card on ``backend`` renders with: ``[tts]``'s options when it names that
    backend, else the backend's defaults."""
    bc, _ = stage_settings(cfg)
    return bc if bc.backend == backend else BackendConfig(backend, {})


def synth_device(cfg: Config, cls: type[Synthesizer]) -> str:
    """The device ``cls`` renders on: ``[tts] device``, else cuda when it can use it."""
    _, pref = stage_settings(cfg)
    want = pref or ("cuda" if "cuda" in cls.devices else "cpu")
    if want not in cls.devices:
        raise ConfigError(f"[tts] device {want!r}: {cls.name} runs on {sorted(cls.devices)}")
    return device_string(cfg) if want == "cuda" else "cpu"


def check_card(
    store: CardStore, card_id: str, who: str, target: str
) -> tuple[Card, type[Synthesizer]]:
    """The card and its backend class, or a CardError naming ``who``, the card and what
    to do."""
    fix = "assign another card with `fukikae voices assign`"
    try:
        card = store.get(card_id)
    except CardError as e:
        raise CardError(f"{who}: {e}; {fix}") from e
    if card.target_language != target:
        raise CardError(
            f"{who}: card {card_id} is for {card.target_language!r}, not {target!r}; {fix}"
        )
    backend = card.voice.backend
    try:
        cls = cast(type[Synthesizer], loader.backend_class("tts", backend))
    except (ConfigError, loader.BackendUnavailable) as e:
        raise CardError(f"{who}: card {card_id}: {e}") from e
    try:
        validate_for_backend(card, cls, lookup=store.get)
        v = card.voice
        if v.source == "stock" and v.voice:
            check_stock_voice(backend, v.voice)
        for voice_id, _ in v.voices:
            check_stock_voice(backend, voice_id)
    except CardError as e:
        raise CardError(f"{who}: card {card_id}: {e}; {fix}") from e
    rev = getattr(cls, "model_revision", None)
    if rev and card.voice.model_revision != rev:
        raise CardError(
            f"{who}: card {card_id} was made for {backend} model revision "
            f"{card.voice.model_revision!r}, and this fukikae renders with {rev!r}; the voice "
            f"could differ, so it is not rendered. {fix[0].upper()}{fix[1:]} (a stock voice "
            f"is `stock:{backend}:<voice>`)"
        )
    if not loader.supports(cls, target):
        raise CardError(f"{who}: card {card_id}: {backend} does not speak {target!r}; {fix}")
    needed = [(r.clip, "clips") for r in card.voice.references]
    if card.voice.latent:
        needed.append((card.voice.latent, "latents"))
    missing = set(store.missing_sidecars(card))
    for ref, sub in needed:
        if ref in missing:
            raise CardError(
                f"{who}: card {card_id} needs sidecar {ref} which is not in "
                f"{store.root / sub}; import the pack with clips or `fukikae voices assign` "
                "another card"
            )
    return card, cls


def delivery_passed(c: DubCue, card: Card, cls: type[Synthesizer], slot: float) -> dict[str, Any]:
    """What the backend gets beyond the card and the text, by its capabilities for the card's
    source; with ``emotion_strength`` 0, no emotion audio, vector or instruction."""
    caps = cls.capabilities_for(card.voice.source)
    if card.voice.emotion_strength == 0:
        caps = caps - DELIVERY_CAPS
    out: dict[str, Any] = {}
    d = c.delivery
    if "emotion_audio" in caps and d.reference[1] - d.reference[0] >= MIN_REFERENCE_S:
        out["emotion_audio"] = [round(d.reference[0], 4), round(d.reference[1], 4)]
    if "emotion_vector" in caps and d.emotion:
        conf = d.emotion_confidence if d.emotion_confidence is not None else 1.0
        out["emotion_vector"] = {d.emotion: conf}
    if "instruction" in caps and d.direction:
        out["instruction"] = d.direction
    if "duration" in caps:
        out["target_seconds"] = round(slot, 4)
    if out.keys() & DELIVERY_CAPS:
        out["emotion_strength"] = card.voice.emotion_strength
    return out


def neutral_only(passed: dict[str, Any]) -> bool:
    """No emotion audio, vector or instruction was passed: the render is neutral speech."""
    return not (passed.keys() & DELIVERY_CAPS)


def prepare_cue(
    cue: DubCue,
    card_id: str,
    cfg: Config,
    store: CardStore,
    *,
    who: str,
    checked: tuple[Card, type[Synthesizer]] | None = None,
    identity: dict[str, Any] | None = None,
    vocals: Path | None = None,
) -> RenderJob:
    """The render job of one cue with ``card_id``: the card checked (``check_card``, unless
    ``checked`` holds its result), its backend config, the cache key and the delivery passed.
    ``identity`` is the backend's ``identity_key``, looked up when not given. ``vocals`` (the
    episode's separated vocals) is needed when the backend takes ``emotion_audio``."""
    card, cls = (
        checked if checked is not None else check_card(store, card_id, who, cfg.target_language)
    )
    bc = backend_config(cfg, card.voice.backend)
    ident = identity if identity is not None else identity_key("tts", bc)
    slot = cue.end - cue.start
    passed = delivery_passed(cue, card, cls, slot)
    ref_digest = None
    if "emotion_audio" in passed:
        if vocals is None:
            raise ValueError(f"{who}: {cls.name} takes emotion_audio, and no vocals were given")
        ref_digest = span_digest(vocals, cast(tuple[float, float], tuple(passed["emotion_audio"])))
    key = cue_key(
        card_id,
        cast(str, cue.text),
        cue.delivery,
        bc,
        capabilities=cls.capabilities_for(card.voice.source),
        identity=ident,
        slot=slot,
        reference_audio=ref_digest,
    )
    return RenderJob(cue, card_id, card, cls, bc, key, slot, passed, vocals)


def render_cue(
    job: RenderJob,
    synth: LazySynth,
    dest: Path,
    store: CardStore,
    *,
    rate: float | None = None,
    ref_dir: Path | None = None,
) -> None:
    """Render ``job`` to ``dest`` (atomically) with the backend ``synth`` loads, at the card's
    rate or ``rate``, then trim its silence and apply ``[voice.post] gain_db``. An emotion
    reference is cut to ``ref_dir/<cue-id>.wav`` (default ``ref/`` next to ``dest``)."""
    from fukikae.cards.select import cut_clip

    voice = job.card.voice if rate is None else replace(job.card.voice, rate=rate)
    p = job.passed
    emotion_audio: Path | None = None
    if "emotion_audio" in p:
        assert job.vocals is not None, "prepare_cue checks vocals for emotion_audio"
        folder = ref_dir if ref_dir is not None else dest.parent / "ref"
        folder.mkdir(parents=True, exist_ok=True)
        emotion_audio = folder / f"{job.cue.id}.wav"
        start, end = p["emotion_audio"]
        with atomic_path(emotion_audio) as tmp_ref:
            cut_clip(job.vocals, start, end, tmp_ref)
    with atomic_path(dest) as tmp:
        synth.get().synthesize(
            cast(str, job.cue.text),
            voice,
            job.card.target_language,
            tmp,
            resolve=store.sidecar_path,
            emotion_audio=emotion_audio,
            emotion_vector=p.get("emotion_vector"),
            instruction=p.get("instruction"),
            emotion_strength=p.get("emotion_strength", job.card.voice.emotion_strength),
            target_seconds=p.get("target_seconds"),
        )
        trim_silence(tmp)
        gain = float(job.card.voice.post.get("gain_db", 0.0))
        if gain:
            _apply_gain(tmp, gain)


class TtsStage:
    name = "tts"

    def __init__(self, cfg: Config | None = None) -> None:
        # The CLI's resolved config; inputs() gets none of its own.
        self.cfg = cfg
        self._prepared: list[str] = []

    # -- configuration -------------------------------------------------------------------

    def _config(self, ep: Episode) -> Config:
        return self.cfg if self.cfg is not None else Workspace(ep.workspace_root).config

    @staticmethod
    def _cues_path(ep: Episode, cfg: Config) -> Path:
        return ep.work / "06-translate" / f"cues.{cfg.target_language}.json"

    def _cues(self, ep: Episode, cfg: Config) -> list[DubCue]:
        path = self._cues_path(ep, cfg)
        return read_json(path, DubCues).cues if path.exists() else []

    @staticmethod
    def _assigned(ep: Episode, cues: list[DubCue], target: str) -> dict[int, str]:
        speaking = sorted({c.character_id for c in cues if c.character_id is not None})
        path = Workspace(ep.workspace_root).registry_path
        if not speaking or not path.exists():
            return {}
        out: dict[int, str] = {}
        with Registry(path) as reg:
            for cid in speaking:
                a = reg.card_for(cid, target)
                if a is not None and a.card_id is not None:
                    out[cid] = a.card_id
        return out

    # -- Stage protocol --------------------------------------------------------------------

    def prepare(self, ep: Episode, cfg: Config) -> None:
        """Give characters without a card theirs before the stage's key is computed (the job
        runner calls this first), so the key the run records already holds the automatic
        cards and the next run with the same audio is up to date."""
        path = self._cues_path(ep, cfg)
        if not path.exists():
            return
        cues = read_json(path, DubCues).cues
        store = CardStore(ep.workspace_root)
        self._prepared += self._assign(
            ep, cfg, cues, store, ep.work / "01-separate" / "vocals.wav"
        )[2]

    def inputs(self, ep: Episode) -> list[Path]:
        cfg = self._config(ep)
        cards = self._assigned(ep, self._cues(ep, cfg), cfg.target_language)
        store = CardStore(ep.workspace_root)
        return [
            self._cues_path(ep, cfg),
            ep.work / "01-separate" / "vocals.wav",
            *(store.card_path(card) for _, card in sorted(cards.items())),
        ]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "07-tts" / "plan.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        bc, _ = stage_settings(cfg)
        cues = self._cues(ep, cfg)
        cards = self._assigned(ep, cues, cfg.target_language)
        store = CardStore(ep.workspace_root)
        backends = {bc.backend}
        for card in cards.values():
            if store.has(card):
                with contextlib.suppress(CardError):  # run() reports it, naming the character
                    backends.add(store.get(card).voice.backend)
        return {
            "backend": bc.backend,
            "options": bc.options,
            "identity": {b: identity_key("tts", backend_config(cfg, b)) for b in sorted(backends)},
            "max_speedup": max_speedup(cfg),
            "cards": {str(k): v for k, v in sorted(cards.items())},
            "extras": extras_voices(cues, cfg) if cues else {},
            "version": RENDER_VERSION,
        }

    # -- run -------------------------------------------------------------------------------

    def run(self, ep: Episode, cfg: Config) -> None:
        target = cfg.target_language
        doc = read_json(self._cues_path(ep, cfg), DubCues)
        if doc.target_language != target:
            raise ConfigError(
                f"{self._cues_path(ep, cfg)} is for {doc.target_language!r}, not {target!r}"
            )
        cues = doc.cues
        vocals = ep.work / "01-separate" / "vocals.wav"
        store = CardStore(ep.workspace_root)
        default_bc, _ = stage_settings(cfg)
        speedup = max_speedup(cfg)

        cards, names, report = self._assign(ep, cfg, cues, store, vocals)
        report = self._prepared + report
        self._prepared = []
        extra_cards = self._extra_cards(cues, cfg, store, default_bc.backend)

        def who(c: DubCue) -> str:
            if c.character_id is not None:
                name = names.get(c.character_id)
                return f"character {c.character_id}" + (f" ({name})" if name else "")
            return f"extra {c.extra or 'unknown'}"

        spoken = [c for c in cues if self._skipped(c) is None]
        users: dict[str, str] = {}
        card_of: dict[str, str] = {}
        for c in spoken:
            if c.character_id is not None:
                card = cards.get(c.character_id)
                if card is None:
                    raise CardError(
                        f"{who(c)} has no {target!r} card; assign one with `fukikae voices assign`"
                    )
            else:
                card = extra_cards[c.extra or "unknown"]
            card_of[c.id] = card
            users.setdefault(card, who(c))

        # Every card is checked before any model loads.
        checked = {cid: check_card(store, cid, users[cid], target) for cid in users}
        identities = {
            b: identity_key("tts", backend_config(cfg, b))
            for b in sorted({card.voice.backend for card, _ in checked.values()})
        }

        jobs: dict[str, list[RenderJob]] = {}
        for c in spoken:
            card_id = card_of[c.id]
            backend = checked[card_id][0].voice.backend
            job = prepare_cue(
                c,
                card_id,
                cfg,
                store,
                who=users[card_id],
                checked=checked[card_id],
                identity=identities[backend],
                vocals=vocals,
            )
            jobs.setdefault(backend, []).append(job)

        cache = ep.workspace_root / ".fukikae" / "tts-cache"
        cache.mkdir(parents=True, exist_ok=True)
        out_dir = ep.work / "07-tts"
        out_dir.mkdir(parents=True, exist_ok=True)
        done: dict[str, TtsPlanCue] = {}
        rendered = 0
        fresh: list[str] = []
        for backend in sorted(jobs):
            cls = jobs[backend][0].cls
            lazy = LazySynth(backend_config(cfg, backend), synth_device(cfg, cls), cfg)
            try:
                for job in jobs[backend]:
                    key, seconds, first, factor, calls = self._render(
                        job, lazy, cache, store, speedup, out_dir / "ref"
                    )
                    rendered += calls
                    if calls:
                        fresh.append(job.cue.id)
                    self._link(cache / f"{key}.wav", out_dir / f"{job.cue.id}.wav")
                    done[job.cue.id] = TtsPlanCue(
                        cue_id=job.cue.id,
                        character_id=job.cue.character_id,
                        extra=job.cue.extra,
                        card_id=job.card_id,
                        backend=backend,
                        key=key,
                        seconds=round(seconds, 4),
                        rendered_seconds=round(first, 4),
                        slot=round(job.slot, 4),
                        rate_factor=factor,
                        passed=job.passed,
                        neutral_only=neutral_only(job.passed),
                    )
            finally:
                lazy.close()

        plan: list[TtsPlanCue] = []
        for c in cues:
            if c.id in done:
                plan.append(done[c.id])
            else:
                plan.append(
                    TtsPlanCue(
                        cue_id=c.id,
                        character_id=c.character_id,
                        extra=c.extra,
                        slot=round(c.end - c.start, 4),
                        skipped=self._skipped(c),
                    )
                )
        for stale in out_dir.glob("*.wav"):
            if stale.stem not in done:
                stale.unlink()
        write_json(out_dir / "plan.json", TtsPlan(episode=ep.id, cues=plan))
        cached = [cid for cid in done if cid not in fresh]
        write_json(
            out_dir / "run.json",
            TtsRun(
                episode=ep.id,
                finished=datetime.now(UTC).isoformat(timespec="seconds"),
                renders=rendered,
                rendered=fresh,
                cached=cached,
                report=report,
            ),
        )
        log.info(
            "tts: %d cues, %d renders, %d from the cache, %d not rendered, %d neutral-only "
            "(no emotion audio, vector or instruction passed)",
            len(cues),
            rendered,
            len(cached),
            len(cues) - len(done),
            sum(1 for c in done.values() if c.neutral_only),
        )

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _skipped(c: DubCue) -> str | None:
        if c.skip is not None:
            return f"skip: {c.skip}"
        if c.error is not None:
            return f"error: {c.error}"
        if not c.text or not c.text.strip():
            return "no text"
        return None

    @classmethod
    def _assign(
        cls, ep: Episode, cfg: Config, cues: list[DubCue], store: CardStore, vocals: Path
    ) -> tuple[dict[int, str], dict[int, str], list[str]]:
        """(character -> card id, character -> name, report lines), after giving every
        character with a cue to speak (not all skipped or errored) and no card one. New lines
        are appended to ``07-tts/assignments.log``."""
        speaking = sorted(
            {c.character_id for c in cues if c.character_id is not None and cls._skipped(c) is None}
        )
        if not speaking:
            return {}, {}, []
        asg_path = ep.work / "05-characters" / "assignments.json"
        spans = character_spans(read_json(asg_path, Assignments)) if asg_path.exists() else {}
        with Registry(Workspace(ep.workspace_root).registry_path) as reg:
            names = {c.id: c.name for c in reg.characters() if c.name}
            report = ensure_assignments(
                reg, store, cfg, {cid: spans.get(cid, []) for cid in speaking}, vocals
            )
            cards: dict[int, str] = {}
            for cid in speaking:
                a = reg.card_for(cid, cfg.target_language)
                if a is not None and a.card_id is not None:
                    cards[cid] = a.card_id
        if report:
            log_path = ep.work / "07-tts" / "assignments.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as f:
                for line in report:
                    log.info("tts: %s", line)
                    f.write(f"{stamp} {ep.id} {cfg.target_language}: {line}\n")
        return cards, names, report

    @staticmethod
    def _extra_cards(
        cues: list[DubCue], cfg: Config, store: CardStore, backend: str
    ) -> dict[str, str]:
        voices = extras_voices(cues, cfg)
        if not voices:
            return {}
        rev = revision(backend)
        out: dict[str, str] = {}
        for label, voice in voices.items():
            check_stock_voice(backend, voice)
            out[label] = store.put(stock_card(backend, voice, cfg.target_language, rev))
        return out

    def _render(
        self,
        job: RenderJob,
        lazy: LazySynth,
        cache: Path,
        store: CardStore,
        speedup: float,
        ref_dir: Path,
    ) -> tuple[str, float, float, float, int]:
        """(key of the final wav, its seconds, the first render's seconds, rate factor,
        renders made)."""
        calls = 0
        base = cache / f"{job.key}.wav"
        if not base.exists():
            render_cue(job, lazy, base, store, ref_dir=ref_dir)
            calls += 1
        seconds = wav_seconds(base)
        if (
            job.slot > 0
            and seconds > job.slot
            and "rate" in job.cls.voice_keys
            and "duration" not in job.cls.capabilities
        ):
            factor = round(min(seconds / job.slot, speedup), 4)
            if factor > 1.0:
                key = fitted_key(job.key, factor)
                fitted = cache / f"{key}.wav"
                if not fitted.exists():
                    render_cue(
                        job, lazy, fitted, store, rate=job.card.voice.rate * factor,
                        ref_dir=ref_dir,
                    )  # fmt: skip
                    calls += 1
                return key, wav_seconds(fitted), seconds, factor, calls
        return job.key, seconds, seconds, 1.0, calls

    @staticmethod
    def _link(src: Path, dest: Path) -> None:
        dest.unlink(missing_ok=True)
        try:
            os.link(src, dest)
        except OSError:
            with atomic_path(dest) as tmp:
                shutil.copyfile(src, tmp)
