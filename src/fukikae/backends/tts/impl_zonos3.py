"""ZONOS3-Mini inside the TTS worker (``fukikae.backends.tts.worker``); spike, lokkju/fukikae#43.

Runs under ``<checkout>/.venvs/zonos3`` (Python 3.12, ``workers/zonos3``), so the module level
imports only the standard library; ``fukikae.backends.tts.zonos3`` imports this module in
fukikae's own environment to read the pins and to use the pure helpers below.

The runtime is Zyphra's plain-PyTorch reference (``zonos3.reference``: ``Zonos3LM`` and
``synthesize`` of one v2 GenerateRequest), the numerical oracle of their Mini-SGLang and ggml
servers. The servers need a CUDA 12.8 toolkit to JIT their kernels (this host has none) and run
their exact-mode experts only on Hopper; the reference needs neither.

Pins: the model snapshot ``Zyphra/ZONOS3-Mini`` at ``REVISION`` (the LM, the dual codec's
generator and SSL teacher, the KEEP-2 iSTFT decoder head, the MECT-B2 speaker encoder), the
``zonos3`` code at git commit ``CODE_REVISION`` (in ``workers/zonos3/uv.lock``), and the
steering bundle of that commit (``assets/steering``, which isn't in the installed package),
checked by ``STEERING_DIGEST``. ``fetch`` (``fukikae backends setup zonos3``) is the only step
that downloads; ``load`` maps every ``hf://Zyphra/ZONOS3-Mini/...`` location the package names
to the pinned local snapshot (``zonos3.utils.release.resolve_file`` and
``resolve_dir_or_local`` are replaced before anything else of zonos3 is imported) and refuses
any other, and the worker runs with ``HF_HUB_OFFLINE=1`` as well.

What it honours:

- the speaker: the card's reference clips joined in card order with 0.2 s of silence, whole
  clips up to ``PROMPT_MAX_S``. Option ``mode``: ``reference`` (the default, the checkpoint's
  own default) encodes that audio with the dual codec into a raw reference prefix, once per
  card; ``embedding`` conditions on the 192-d MECT-B2 speaker embedding of the same audio;
- ``seed`` (the request's seed; 0 when the card sets none), ``temperature`` and ``top_p``
  (the model's 0.75 and 0.69 when unset);
- ``target_seconds`` only with option ``pace`` on: the speaking-rate condition row set to the
  line's UTF-8 bytes over the target (``speaking_rate_for``), the model's own pace input;
- ``emotion_vector`` as steering controls: each key names a control of the bundle
  (``<set>.<direction>``, a trailing ``-`` for the negative pole) and its value in [0, 1] is
  scaled by option ``dose`` (default ``DOSE``) into the control's raw dose. The spike's emotion
  arm uses this; no capability is declared for it.

No stock voice is ever used (D41): a request without reference clips is refused, and the
bundle's default voices are never loaded (the reference has no speaker ids).
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import random
import wave
from pathlib import Path
from typing import Any

REPO = "Zyphra/ZONOS3-Mini"
REVISION = "f68e6e470fd14d6d1bc50e8461ee4d4f64d6c513"
CODE_URL = "https://github.com/Zyphra/ZONOS3-Mini"
CODE_REVISION = "b171b19149d29b8bcc31e0342d5e3b0063dc479b"
MODEL_PATTERNS = [
    "lm/zonos3_mini/*",
    "lm/zonos3_mini/**",
    "autoencoder/generator.pt",
    "autoencoder/teacher/*",
    "autoencoder/heads/keep2_plain/*",
    "speaker_encoders/ownmect_s2/*",
    "registries/*",
]
# sha256 over "<path>\0<sha256 of the file>\n" of every file under assets/steering at
# CODE_REVISION, paths sorted (steering_digest).
STEERING_DIGEST = "9f03ade602ad1ee75e34dde3b0ff723e2a2e4aba561757ddc5d7603e594bfd97"
STEER_BUNDLE = "studio_bundle_mect60k_x10.json"
SAMPLE_RATE = 44100
PROMPT_MAX_S = 30.0
PROMPT_GAP_S = 0.2
MODES = ("reference", "embedding")
DTYPES = ("bf16", "f32")
DOSE = 3.0
# ISO 639-1 -> the server's text-normalization language codes (registries/languages.json).
LANGUAGE_CODES = {
    "en": "en_us",
    "ja": "ja",
    "de": "de",
    "fr": "fr_fr",
    "es": "es",
    "it": "it",
    "pt": "pt_br",
    "zh": "cmn",
    "ko": "ko",
}
LANGUAGES = tuple(LANGUAGE_CODES)
SETUP = "fukikae backends setup zonos3"
# ZONOS3's own text markup (ruling CD20): pronunciation groups (U+E000 IPA U+E001), the start
# marker U+E002 and pause markers U+E003 + bucket, all private-use characters; the worker strips
# them, and the rest of the private-use block, from text and transcripts.
CONTROL_MARKUP: tuple[str, ...] = (r"[-]",)


# ---- pure helpers (also used in fukikae's environment, by the tests) -----------------------


def language_code(code: str) -> str:
    try:
        return LANGUAGE_CODES[code]
    except KeyError:
        raise ValueError(
            f"ZONOS3-Mini (here) speaks {', '.join(LANGUAGES)}, not {code!r}"
        ) from None


def speaking_rate_for(text: str, target_seconds: float) -> float:
    """The speaking-rate condition value (UTF-8 bytes per second) that fits ``text`` into
    ``target_seconds``."""
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    return len(text.strip().encode("utf-8")) / target_seconds


def controls_for(vector: dict[str, float] | None, dose: float) -> dict[str, float] | None:
    """Steering controls from an emotion vector: ``{"mood.arousal": 0.5}`` at dose 3 is
    ``{"mood.arousal": 1.5}``; a key ending in ``-`` is the negative pole."""
    if not vector:
        return None
    out: dict[str, float] = {}
    for key, value in vector.items():
        name, sign = (key[:-1], -1.0) if key.endswith("-") else (key, 1.0)
        if "." not in name:
            raise ValueError(f"control {key!r} is not <set>.<direction>")
        out[name] = out.get(name, 0.0) + sign * float(value) * dose
    return {k: v for k, v in out.items() if v}


def prompt_plan(refs: list[tuple[str, float]]) -> list[str]:
    """The whole clips, in card order, that fit ``PROMPT_MAX_S`` with ``PROMPT_GAP_S`` between
    them; the first is always taken (cut to fit later). ``refs`` is (clip, seconds)."""
    clips: list[str] = []
    total = 0.0
    for clip, seconds in refs:
        need = seconds + (PROMPT_GAP_S if clips else 0.0)
        if clips and total + need > PROMPT_MAX_S:
            break
        clips.append(clip)
        total += need
    return clips


def steering_digest(assets: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((assets / "steering").rglob("*")):
        if p.is_file():
            rel = p.relative_to(assets).as_posix()
            h.update(
                rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n"
            )
    return h.hexdigest()


def assets_dir() -> Path:
    """Where ``fetch`` puts the steering assets of ``CODE_REVISION``."""
    hub = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface"
    )
    return Path(hub) / "fukikae-zonos3" / CODE_REVISION[:12] / "assets"


# ---- the worker side -------------------------------------------------------------------------


def _snapshot() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            REPO, revision=REVISION, allow_patterns=MODEL_PATTERNS, local_files_only=True
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"{REPO}@{REVISION[:12]} is not in the Hugging Face cache; download it with `{SETUP}`"
        ) from e


def _pin_release(root: str) -> None:
    """Point zonos3's hf:// resolution at the pinned snapshot, offline; refuse anything else."""
    import zonos3.utils.release as release  # pyright: ignore[reportMissingImports]

    prefix = f"hf://{REPO}/"

    def local(path: Any) -> Any:
        if path is None:
            return path
        path = str(path)
        if not path.startswith("hf://"):
            return os.path.expanduser(path)
        if not path.startswith(prefix):
            raise RuntimeError(f"zonos3 asked for {path}, outside the pinned {REPO}")
        out = os.path.join(root, path[len(prefix) :].strip("/"))
        if not os.path.exists(out):
            raise RuntimeError(f"{path} is not in the pinned snapshot; set it up with `{SETUP}`")
        return out

    release.resolve_file = local  # type: ignore[assignment]
    release.resolve_dir_or_local = local  # type: ignore[assignment]


def fetch() -> None:
    """The setup step: the pinned model snapshot, and the steering assets of the pinned code."""
    import sys
    import tarfile
    import tempfile
    import urllib.request

    from huggingface_hub import snapshot_download

    print(f"fetching {REPO}@{REVISION[:12]}", file=sys.stderr, flush=True)
    snapshot_download(REPO, revision=REVISION, allow_patterns=MODEL_PATTERNS)
    dest = assets_dir()
    if (dest / "steering").is_dir() and steering_digest(dest) == STEERING_DIGEST:
        return
    url = f"https://codeload.github.com/Zyphra/ZONOS3-Mini/tar.gz/{CODE_REVISION}"
    print(f"fetching the steering assets of {CODE_REVISION[:12]}", file=sys.stderr, flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tar_path = Path(tmp) / "src.tar.gz"
        urllib.request.urlretrieve(url, tar_path)  # noqa: S310 - fixed https URL
        top = f"ZONOS3-Mini-{CODE_REVISION}/assets/steering/"
        out = Path(tmp) / "assets"
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith(top) and m.isfile()]
            for m in members:
                rel = m.name[len(f"ZONOS3-Mini-{CODE_REVISION}/assets/") :]
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                target.write_bytes(src.read())
        got = steering_digest(out)
        if got != STEERING_DIGEST:
            raise RuntimeError(f"steering assets digest {got} != pinned {STEERING_DIGEST}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        out.rename(dest)


class Impl:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.lm: Any = None
        self.assets: Any = None
        self.device: Any = None
        self.dtype: Any = None
        self.mode = "reference"
        self.pace = False
        self.dose = DOSE
        self.decoder: str | None = None
        self._speakers: dict[tuple[str, ...], dict[str, Any]] = {}

    def load(self, options: dict[str, Any], device: str) -> None:
        self.mode = str(options.get("mode") or "reference")
        if self.mode not in MODES:
            raise ValueError(f"zonos3 mode must be one of {', '.join(MODES)}")
        dtype = str(options.get("dtype") or "bf16")
        if dtype not in DTYPES:
            raise ValueError(f"zonos3 dtype must be one of {', '.join(DTYPES)}")
        self.pace = bool(options.get("pace", False))
        self.dose = float(options.get("dose", DOSE))
        self.decoder = options.get("decoder") or None
        root = _snapshot()
        steer = assets_dir()
        if not (steer / "steering").is_dir() or steering_digest(steer) != STEERING_DIGEST:
            raise RuntimeError(f"the pinned steering assets are missing or changed; run `{SETUP}`")
        _pin_release(root)
        import torch
        from zonos3.models.text_extra import TextExtraRows  # pyright: ignore[reportMissingImports]
        from zonos3.reference.lm import Zonos3LM  # pyright: ignore[reportMissingImports]
        from zonos3.reference.synth import ReferenceAssets  # pyright: ignore[reportMissingImports]

        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        lm_dir = os.path.join(root, "lm", "zonos3_mini")
        self.lm = Zonos3LM.load(lm_dir, device=self.device, dtype=self.dtype)
        self.assets = ReferenceAssets.load(steer_bundle=str(steer / "steering" / STEER_BUNDLE))
        self.assets.grow(self.lm)
        rows = os.path.join(lm_dir, "text_rows", "pause_tokens_s5000.pt")
        self.lm.append_text_rows(TextExtraRows.load(rows))

    def unload(self) -> None:
        self.lm = None
        self.assets = None
        self._speakers.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _speaker(self, refs: list[dict[str, Any]]) -> dict[str, Any]:
        """The card's speaker fields of a GenerateRequest (computed once per clip list)."""
        key = (self.mode, *(r["clip"] for r in refs))
        if key in self._speakers:
            return self._speakers[key]
        import numpy as np
        import soundfile as sf

        clips: list[tuple[str, float]] = []
        for r in refs:
            info = sf.info(r["clip"])
            clips.append((r["clip"], info.frames / info.samplerate))
        parts: list[Any] = []
        rate = None
        for clip in prompt_plan(clips):
            y, sr = sf.read(clip, dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if rate is None:
                rate = sr
            elif sr != rate:
                import torch
                import torchaudio.functional as taf

                y = taf.resample(torch.from_numpy(y), sr, rate).numpy()
            if parts:
                parts.append(np.zeros(int(PROMPT_GAP_S * rate), np.float32))
            parts.append(y)
        assert rate is not None
        audio = np.concatenate(parts)[: int(PROMPT_MAX_S * rate)]
        if self.mode == "reference":
            import torch
            from zonos3.tokenizer import dual_reference  # pyright: ignore[reportMissingImports]

            encode_dual_reference = dual_reference.encode_dual_reference

            index = self.device.index if self.device.index is not None else 0
            codes = encode_dual_reference(
                torch.from_numpy(audio)[None, :],
                int(rate),
                max_seconds=PROMPT_MAX_S,
                device=f"cuda:{index}",
            )
            fields = {"speaker_mode": "reference", "speaker_reference_codes": codes.tolist()}
        else:
            # The MECT-B2 embedding computed here, as zonos3.reference.synth._speaker_embedding
            # does for an uploaded clip (which it would first transcode with an ffmpeg binary).
            import torch

            c = self.lm.config
            encoder = self.assets.speaker_encoder(
                c.speaker_embedding_codec, c.speaker_embedding_dim, self.device
            )
            with torch.inference_mode():
                output = encoder(torch.from_numpy(audio)[None, :], int(rate))
            vec = None
            for cand in output if isinstance(output, tuple) else (output,):
                cand = cand.squeeze(0).to(dtype=torch.float32, device="cpu")
                if cand.numel() == c.speaker_embedding_dim:
                    vec = cand.numpy()
                    break
            if vec is None:
                raise RuntimeError("the speaker encoder returned no embedding of the model's size")
            buf = io.BytesIO()
            np.save(buf, vec, allow_pickle=False)
            fields = {
                "speaker_mode": "embedding",
                "speaker_embedding_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        self._speakers[key] = fields
        return fields

    def synthesize(self, req: dict[str, Any]) -> float:
        if self.lm is None:
            raise RuntimeError("zonos3: load() before synthesize()")
        voice = req["voice"]
        refs = list(voice.get("references", []))
        if not refs:
            raise ValueError("ZONOS3-Mini needs a clone card with reference clips (D41)")
        body: dict[str, Any] = {
            "text": req["text"],
            "language": language_code(req["language"]),
            "stream": False,
            "seed": int(voice.get("seed") or 0),
            **self._speaker(refs),
        }
        if self.decoder:
            body["decoder"] = self.decoder
        if voice.get("temperature") is not None:
            body["temperature"] = float(voice["temperature"])
        if voice.get("top_p") is not None:
            body["top_p"] = float(voice["top_p"])
        if self.pace and req.get("target_seconds"):
            body["condition_values"] = {
                "speaking_rate": speaking_rate_for(req["text"], float(req["target_seconds"]))
            }
        controls = controls_for(req.get("emotion_vector"), self.dose)
        if controls:
            body["controls"] = controls
        import numpy as np
        import torch
        from zonos3.reference.synth import synthesize  # pyright: ignore[reportMissingImports]

        random.seed(body["seed"])
        np.random.seed(body["seed"] % (2**32))
        torch.manual_seed(body["seed"])
        result = synthesize(self.lm, body, assets=self.assets, device=self.device, dtype=self.dtype)
        audio = np.asarray(result.pcm, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("ZONOS3-Mini returned no audio")
        _write_wav(req["out"], audio, SAMPLE_RATE)
        return len(audio) / SAMPLE_RATE


def _write_wav(path: str, audio: Any, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
