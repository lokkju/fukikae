"""IndexTTS2 on upstream's TensorRT backend (D28 lever 2): an opt-in engine for IndexTTS2 cards.

It renders ``indextts2`` cards (``card_backend``) with the same weights, the same prompt handling
and the same capabilities as ``IndexTts2Synthesizer``, through TensorRT and TensorRT-LLM engines
(``impl_indextts2_trt`` has what runs on which). A workspace opts in with
``[tts] engines = { indextts2 = "indextts2_trt" }``; cards stay ``indextts2`` cards and their
renders get this backend's name and identity in the cache key, so they never mix with the
PyTorch worker's.

The worker runs in a local container image built from ``workers/indextts2_trt`` (TensorRT-LLM
needs OpenMPI, which the host doesn't carry), never pushed. ``fukikae backends setup
indextts2_trt`` builds the image, fetches the pinned models (the only download) and builds the
engines for this machine's GPU under the GPU lock, into
``$XDG_CACHE_HOME/fukikae/indextts2_trt/<project digest>/``. At run time fukikae holds the GPU
lock as for any worker and runs ``docker run -i --rm`` with the same JSON-lines protocol over
stdin and stdout. The container gets that one GPU, no network, the caller's uid, the Hugging
Face cache and the engines read-only, fukikae's source read-only, and one read-write directory,
the worker's own temporary directory: each request's reference clips and emotion span are
copied into it (clips by content, so the worker's prompt cache sees one path per clip) and the
render is written there and moved to the path fukikae named. Nothing else of the host is
visible. When the worker stops, the container is removed by name before the lock is released,
so a docker client killed under it can't leave the engines running on a card fukikae thinks is
free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import secrets
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from fukikae.backends.loader import BackendUnavailable
from fukikae.backends.tts import impl_indextts2_trt as _impl
from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
from fukikae.backends.tts.remote import WORKER_MODULE, nvml_index
from fukikae.privacy import TELEMETRY_OFF

if TYPE_CHECKING:
    from fukikae.cards.card import Voice

log = logging.getLogger(__name__)

DOCKER = "docker"
IMAGE = "fukikae-indextts2-trt"
# Variables passed into the container beyond the fixed ones (the tests add their fake impl).
CONTAINER_PASS: frozenset[str] = frozenset()
# The docker client's own settings, which its process (not the container) gets.
DOCKER_ENV = frozenset({"DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_CERT_PATH",
                        "DOCKER_TLS_VERIFY"})  # fmt: skip
SRC_MOUNT = "/fukikae/src"


def _hub_cache() -> Path:
    from fukikae.backends.identity import _hub_cache as hub

    return hub()


def _cache_home() -> Path:
    env = os.environ.get("XDG_CACHE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache"


class IndexTts2TrtSynthesizer(IndexTts2Synthesizer):
    """IndexTTS2 cards rendered through TensorRT (fp16 engines, batch 1)."""

    name: ClassVar[str] = "indextts2_trt"
    extra: ClassVar[str] = "indextts2_trt"
    card_backend: ClassVar[str] = "indextts2"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda"})
    option_keys: ClassVar[frozenset[str]] = frozenset()
    # nvidia-smi peak on GPU 0 during the 40-cue E14 run, 15438 MiB with the CUDA context
    # (TensorRT-LLM reserves its KV cache up front), plus a margin (results, "D28 lever 2").
    vram_mb: ClassVar[int] = 16500
    # The engines and the code that runs them: a voice can differ from the PyTorch worker's.
    revision: ClassVar[str] = f"{_impl.REVISION}+{_impl.CODE_REVISION[:12]}+trt-{_impl.PRECISION}"
    # The cards' pin is the model's, the same as the PyTorch worker's.
    model_revision: str = IndexTts2Synthesizer.revision
    load_timeout_s: ClassVar[float] = 900.0

    def __init__(self, options: dict[str, Any], device: str) -> None:
        super().__init__(options, device)
        self._container: str | None = None
        self._staged: dict[str, Path] = {}
        self._serial = 0

    # -- where things live ------------------------------------------------------------------

    def project_digest(self) -> str | None:
        """The first 12 hex of the sha256 over the worker project's lock and Dockerfile: the
        image tag and the engines' directory."""
        proj = self.project_dir()
        files = [proj / "uv.lock", proj / "Dockerfile"]
        if not all(f.is_file() for f in files):
            return None
        h = hashlib.sha256()
        for f in files:
            h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
        return h.hexdigest()[:12]

    def image(self) -> str:
        return f"{IMAGE}:{self.project_digest()}"

    def engine_dir(self) -> Path:
        return _cache_home() / "fukikae" / self.name / str(self.project_digest())

    def manifest(self) -> dict[str, Any] | None:
        path = self.engine_dir() / _impl.MANIFEST
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def lock_revision(self) -> str | None:
        """The project digest and the engines' build id, for the cache keys' identity: a
        rebuilt image or rebuilt engines can change the audio."""
        digest = self.project_digest()
        if digest is None:
            return None
        manifest = self.manifest()
        return f"{digest}-{manifest['build_id']}" if manifest else digest

    def environment_location(self) -> Path | str:
        return f"{self.image()}, engines {self.engine_dir()}"

    # -- checks -----------------------------------------------------------------------------

    def image_exists(self) -> bool:
        try:
            r = subprocess.run(  # noqa: S603 - fixed argv
                [DOCKER, "image", "inspect", "--format", "{{.Id}}", self.image()],
                capture_output=True, timeout=30, check=False,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    def gpu_compute_capability(self, index: int) -> str | None:
        try:
            r = subprocess.run(  # noqa: S603 - fixed argv
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i",  # noqa: S607
                 str(index)],
                capture_output=True, text=True, timeout=30, check=False,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None

    def environment_problem(self) -> str | None:
        if self.project_digest() is None:
            return f"{self.project_dir()} has no uv.lock and Dockerfile"
        if shutil.which(DOCKER) is None and not Path(DOCKER).is_file():
            return f"{DOCKER} is not on PATH"
        if not self.image_exists():
            return f"no image {self.image()}"
        manifest = self.manifest()
        if manifest is None:
            return f"no TensorRT engines at {self.engine_dir()}"
        if self.device.startswith("cuda"):
            have = self.gpu_compute_capability(nvml_index(self.device))
            built = manifest.get("compute_capability")
            if have and built and have != built:
                return (
                    f"the engines at {self.engine_dir()} were built for compute capability "
                    f"{built}, and this GPU is {have}"
                )
        return None

    def check_environment(self) -> None:
        problem = self.environment_problem()
        if problem:
            raise BackendUnavailable(
                f"tts backend {self.name!r}: {problem}; set it up with `{self.setup_hint()}`"
            )

    # -- the container ----------------------------------------------------------------------

    def worker_env(self, *, online: bool = False) -> dict[str, str]:  # noqa: ARG002
        """The docker client's environment: the worker allowlist and docker's own settings.
        The container gets only what ``command`` passes it."""
        env = super().worker_env(online=False)
        env.pop("HF_TOKEN", None)
        env.update({k: v for k, v in os.environ.items() if k in DOCKER_ENV})
        return env

    def _container_env(self) -> list[str]:
        engines = self.engine_dir()
        values = {
            **TELEMETRY_OFF,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HOME": "/tmp",  # noqa: S108 - the container's own /tmp
            "HF_HUB_CACHE": str(engines / "hub"),
            "FUKIKAE_HUB_REAL": str(_hub_cache()),
            "FUKIKAE_TRT_ENGINES": str(engines),
            "PYTHONPATH": f"{SRC_MOUNT}:{_impl.INDEX_TTS}",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        values.update({k: os.environ[k] for k in sorted(CONTAINER_PASS) if k in os.environ})
        out: list[str] = []
        for k, v in values.items():
            out += ["-e", f"{k}={v}"]
        return out

    def command(self) -> list[str]:
        if self.gpu_index is None or self._cwd is None:
            raise RuntimeError("indextts2_trt: the container starts only from load() on a GPU")
        self._container = f"fukikae-{self.name}-{os.getpid()}-{secrets.token_hex(4)}"
        stage = str(self._cwd)
        engines = str(self.engine_dir())
        hub = str(_hub_cache())
        src = str(self.checkout() / "src")
        return [
            DOCKER, "run", "-i", "--rm", "--init",
            "--name", self._container,
            "--gpus", f"device={self.gpu_index}",
            "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--shm-size", "2g",
            "-w", "/tmp",  # noqa: S108 - the container's own /tmp
            "-v", f"{stage}:{stage}",
            "-v", f"{engines}:{engines}:ro",
            "-v", f"{hub}:{hub}:ro",
            "-v", f"{src}:{SRC_MOUNT}:ro",
            *self._container_env(),
            # the CUDA image's entrypoint prints a banner on stdout, which is the protocol's
            "--entrypoint", "python",
            self.image(),
            "-m", WORKER_MODULE, self.name,
        ]  # fmt: skip

    def _remove_container(self, name: str) -> None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(  # noqa: S603 - fixed argv
                [DOCKER, "rm", "-f", name], capture_output=True, timeout=60, check=False
            )

    def _stop(self) -> None:
        # The container goes before the lock: hold the lock back from the base class's stop.
        name, self._container = self._container, None
        lock, self._lock = self._lock, contextlib.ExitStack()
        self._staged.clear()
        try:
            super()._stop()
        finally:
            try:
                if name:
                    self._remove_container(name)
            finally:
                lock.close()

    # -- staging ----------------------------------------------------------------------------

    def _stage(self, src: Path, sub: str, *, keep: bool) -> Path:
        """A copy of ``src`` in the worker's directory: by content when ``keep`` (reused by
        later lines), else under a fresh name (removed after the line)."""
        assert self._cwd is not None
        src = Path(src).resolve()
        folder = self._cwd / sub
        folder.mkdir(exist_ok=True)
        if keep:
            known = self._staged.get(str(src))
            if known is not None and known.exists():
                return known
            digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
            dest = folder / f"{digest}{src.suffix}"
            if not dest.exists():
                shutil.copyfile(src, dest)
            self._staged[str(src)] = dest
            return dest
        self._serial += 1
        dest = folder / f"{self._serial:06d}{src.suffix}"
        shutil.copyfile(src, dest)
        return dest

    def synthesize(
        self,
        text: str,
        voice: Voice,
        language: str,
        out: Path,
        *,
        resolve: Callable[[str], Path],
        emotion_audio: Path | None = None,
        **kw: Any,
    ) -> float:
        if self._cwd is None:
            return super().synthesize(
                text, voice, language, out, resolve=resolve, emotion_audio=emotion_audio, **kw
            )
        emo = None if emotion_audio is None else self._stage(emotion_audio, "emo", keep=False)
        (self._cwd / "out").mkdir(exist_ok=True)
        self._serial += 1
        staged_out = self._cwd / "out" / f"{self._serial:06d}.wav"
        try:
            seconds = super().synthesize(
                text, voice, language, staged_out,
                resolve=lambda ref: self._stage(resolve(ref), "refs", keep=True),
                emotion_audio=emo, **kw,
            )  # fmt: skip
            dest = Path(os.path.abspath(out))
            if os.path.lexists(dest):
                dest.unlink()
            shutil.move(str(staged_out), dest)
            return seconds
        finally:
            for p in (emo, staged_out):
                if p is not None:
                    with contextlib.suppress(FileNotFoundError):
                        p.unlink()

    # -- setup ------------------------------------------------------------------------------

    @classmethod
    def setup(cls, *, fetch: bool = True, run: Callable[..., Any] = subprocess.run) -> None:
        """Build the image, fetch the pinned models (online, the only download) and build the
        engines for this machine's GPU (the first CUDA device, under its lock)."""
        from fukikae.gpulock import gpu_lock

        inst = cls({}, "cuda:0")
        if inst.project_digest() is None:
            raise BackendUnavailable(f"{inst.project_dir()} has no uv.lock and Dockerfile")
        client = inst.worker_env()
        log.info("building %s from %s", inst.image(), inst.project_dir())
        run([DOCKER, "build", "-t", inst.image(), str(inst.project_dir())], env=client,
            check=True)  # fmt: skip
        hub = _hub_cache()
        hub.mkdir(parents=True, exist_ok=True)
        user = f"{os.getuid()}:{os.getgid()}"
        base = [DOCKER, "run", "--rm", "--init", "--user", user, "-w", "/tmp",  # noqa: S108
                "-v", f"{inst.checkout() / 'src'}:{SRC_MOUNT}:ro",
                "-e", "HOME=/tmp", "-e", f"HF_HUB_CACHE={hub}",
                "-e", f"PYTHONPATH={SRC_MOUNT}:{_impl.INDEX_TTS}"]  # fmt: skip
        for k, v in TELEMETRY_OFF.items():
            base += ["-e", f"{k}={v}"]
        if fetch:
            log.info("fetching the %s model files", cls.name)
            online = [*base, "-v", f"{hub}:{hub}", "-e", "HF_HUB_OFFLINE=0",
                      "-e", "TRANSFORMERS_OFFLINE=0"]  # fmt: skip
            if os.environ.get("HF_TOKEN"):
                online += ["-e", "HF_TOKEN"]  # docker reads the value from its own environment
                client = {**client, "HF_TOKEN": os.environ["HF_TOKEN"]}
            run([*online, "--entrypoint", "python", inst.image(), "-m", WORKER_MODULE, cls.name,
                 "--fetch"],
                env=client, check=True)  # fmt: skip
            client.pop("HF_TOKEN", None)
        engines = inst.engine_dir()
        engines.mkdir(parents=True, exist_ok=True)
        gpu = nvml_index("cuda:0")
        log.info("building the TensorRT engines on GPU %d into %s", gpu, engines)
        with gpu_lock(gpu, f"{cls.name} engine build"):
            run([*base, "--gpus", f"device={gpu}", "--network", "none", "--shm-size", "2g",
                 "-v", f"{hub}:{hub}:ro", "-v", f"{engines}:{engines}",
                 "-e", f"FUKIKAE_HUB_REAL={hub}", "-e", "HF_HUB_OFFLINE=1",
                 "--entrypoint", "python", inst.image(), "-m",
                 "fukikae.backends.tts.impl_indextts2_trt",
                 "build", str(engines)],
                env=client, check=True)  # fmt: skip
