"""Tag stage (optional): sound events and per-turn emotion from the vocals.

Reads ``01-separate/vocals_16k.wav`` and ``04-diarize/turns.json``; writes
``04b-tag/tags.json`` (a ``Tags``). ``[tag] backend`` makes the events (and the emotions, unless
``emotion_backend`` names a second backend, with its options in ``[tag.emotion]``: then that
backend's scored emotions replace the first's, which is handed no turns). Off unless
``[tag] enabled = true`` (the anime and live-action profiles turn it on); the workspace's table
wins over the profile's. ``enabled`` is not a backend option, so it stays out of the cache key.

Device per backend (D26, like the tts stage's ruling CD14): the emotion backend's is
``[tag.emotion] device``, then ``[tag] device``; the events backend's is ``[tag] device``; either
falls back to the backend's ``default_device`` (SenseVoice and emotion2vec: cuda), else cuda when
the backend can use it. cuda means the configured GPU under the GPU lock (ruling CD15), and the
CPU when there is no usable GPU. ``device`` is never a backend option; the preferences (not the
card found) are in the key.

Keys are checked (C9), whether or not the stage is on: ``[tag]`` takes its stage keys and the
events backend's ``option_keys``, ``[tag.emotion]`` ``device`` and the emotion backend's, so a
typo (``enable = true``) is an error instead of tagging silently left off. For a backend that
declares no ``option_keys`` only near-misses of the stage keys are refused.
"""

from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any, cast

from fukikae import runctx
from fukikae.artifacts import read_json, write_json
from fukikae.backends.base import Tagger
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import backend_class, create
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.gpulock import gpu_job
from fukikae.models import Diarization, Tags, Turn
from fukikae.workspace import Episode

_sleep = time.sleep


STAGE_KEYS = ("backend", "enabled", "device", "emotion_backend", "emotion", "lang", "pair")


def _merged(cfg: Config) -> dict[str, Any]:
    """``[tag]``, the workspace's keys over the profile's (``[tag.emotion]`` merged too)."""
    out: dict[str, Any] = {}
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("tag")
        if not isinstance(table, dict):
            continue
        for k, v in table.items():
            if k == "emotion" and isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
    return out


def _option_keys(backend: Any) -> frozenset[str] | None:
    if not isinstance(backend, str) or not backend:
        return None
    try:
        cls = backend_class("tag", backend)
    except Exception:  # noqa: BLE001 - an unknown backend fails at create(), with its message
        return None
    keys = getattr(cls, "option_keys", None)
    return frozenset(keys) if keys is not None else None


def _check_keys(table: dict[str, Any], stage: tuple[str, ...], backend: Any, where: str) -> None:
    allowed = _option_keys(backend)
    for key in table:
        if key in stage or (allowed is not None and key in allowed):
            continue
        near = difflib.get_close_matches(key, [*stage, *(allowed or ())], n=1, cutoff=0.75)
        if allowed is None and not near:
            continue  # a backend that lists no options: its own code checks them
        hint = f"; did you mean {near[0]!r}?" if near else ""
        takes = ", ".join(sorted({*stage, *(allowed or ())} - {"lang", "pair"}))
        raise ConfigError(f"{where} has no key {key!r}{hint} (it takes: {takes})")


def check_tag_config(cfg: Config) -> None:
    """Refuse unknown ``[tag]`` and ``[tag.emotion]`` keys and a non-boolean ``enabled``."""
    t = _merged(cfg)
    if not t:
        return
    if "enabled" in t and not isinstance(t["enabled"], bool):
        raise ConfigError(f"[tag] enabled must be true or false, not {t['enabled']!r}")
    _check_keys(t, STAGE_KEYS, t.get("backend"), "[tag]")
    emotion = t.get("emotion")
    if isinstance(emotion, dict):
        _check_keys(emotion, ("device",), t.get("emotion_backend"), "[tag.emotion]")


def tag_enabled(cfg: Config) -> bool:
    """`[tag] enabled`, workspace over profile; false when neither sets it. The ``[tag]``
    keys are checked first (``check_tag_config``)."""
    check_tag_config(cfg)
    return _merged(cfg).get("enabled", False) is True


class TagStage:
    name = "tag"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "01-separate" / "vocals_16k.wav", ep.work / "04-diarize" / "turns.json"]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "04b-tag" / "tags.json"]

    @staticmethod
    def _backend_cfg(cfg: Config) -> tuple[BackendConfig, BackendConfig | None, str, str | None]:
        """(events backend, emotion backend or None, events device, emotion device or None):
        the device preferences, ``cpu`` or ``cuda``."""
        check_tag_config(cfg)
        bc = cfg.resolve("tag")
        skip = ("enabled", "device", "emotion_backend", "emotion")
        options = {k: v for k, v in bc.options.items() if k not in skip}
        stage_device = bc.options.get("device")
        name = bc.options.get("emotion_backend")
        if name is not None and not str(name).strip():
            raise ConfigError(
                "[tag] emotion_backend is empty; leave the key out to use no second backend"
            )
        events = BackendConfig(bc.backend, options)
        device = TagStage._device(events.backend, stage_device, "[tag] device")
        emotion, emotion_device = None, None
        if name:
            sub = bc.options.get("emotion")
            sub = dict(sub) if isinstance(sub, dict) else {}
            own = sub.pop("device", None)
            emotion = BackendConfig(str(name), sub)
            where = "[tag.emotion] device" if own is not None else "[tag] device"
            emotion_device = TagStage._device(
                emotion.backend, own if own is not None else stage_device, where
            )
        return events, emotion, device, emotion_device

    @staticmethod
    def _device(backend: str, want: Any, where: str) -> str:
        if want is not None:
            if want not in ("cpu", "cuda"):
                raise ConfigError(f'{where} must be "cpu" or "cuda", not {want!r}')
            return str(want)
        try:
            cls = backend_class("tag", backend)
        except Exception:  # an unknown backend fails at create(), with its own message
            return "cpu"
        default = getattr(cls, "default_device", None)
        if default:
            return str(default)
        return "cuda" if "cuda" in getattr(cls, "devices", ()) else "cpu"

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        # Both devices are keyed: emotion2vec scores differently on CPU and CUDA (FT S01E12:
        # 271 of 537 emotion confidences differ, by up to 0.0043; SenseVoice's events and
        # every label match), and a confidence is the tts plan's emotion_vector weight.
        bc, ebc, device, emotion_device = self._backend_cfg(cfg)
        return {
            "backend": bc.backend,
            "options": bc.options,
            "device": device,
            "identity": identity_key("tag", bc),
            "emotion_backend": ebc.backend if ebc else None,
            "emotion_options": ebc.options if ebc else None,
            "emotion_identity": identity_key("tag", ebc) if ebc else None,
            "emotion_device": emotion_device,
        }

    @staticmethod
    def _run_one(
        bc: BackendConfig, device: str, cfg: Config, audio: Path, turns: list[Turn]
    ) -> Tags:
        backend = create("tag", bc, device)
        with gpu_job(cfg.gpu_device, device, f"tag {bc.backend}"):  # ruling CD15
            if device.startswith("cuda"):
                wait_for_vram(
                    cfg.gpu_device, backend.vram_mb, runctx.vram_timeout_s, query=free_mb,
                    sleep=_sleep,
                )  # fmt: skip
            try:
                backend.load()
                return cast(Tagger, backend).tag(audio, turns)
            finally:
                backend.unload()

    def run(self, ep: Episode, cfg: Config) -> None:
        audio, turns_path = self.inputs(ep)
        (out,) = self.outputs(ep)
        bc, ebc, want, emotion_want = self._backend_cfg(cfg)
        gpu = device_string(cfg) if "cuda" in (want, emotion_want) else "cpu"
        turns = read_json(turns_path, Diarization).turns
        # With a separate emotion backend the events backend decodes no turns.
        tags = self._run_one(bc, gpu if want == "cuda" else "cpu", cfg, audio, [] if ebc else turns)
        if ebc:
            device = gpu if emotion_want == "cuda" else "cpu"
            scored = self._run_one(ebc, device, cfg, audio, turns)
            tags = tags.model_copy(
                update={"emotions": scored.emotions, "emotion_backend": scored.backend}
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, tags)
