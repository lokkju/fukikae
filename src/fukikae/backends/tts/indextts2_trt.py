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
stdin and stdout. The container gets that one GPU, no network, the caller's uid, no
capabilities, no-new-privileges, a pids limit, a read-only image with /tmp in memory, the Hugging
Face cache and the engines read-only, fukikae's source read-only, and one read-write directory,
the worker's own temporary directory: each request's reference clips and emotion span are
copied into it (clips by content, so the worker's prompt cache sees one path per clip, never
through a link the container left) and the render is written there and copied out from a
descriptor opened without following links. Nothing else of the host is visible. The worker gets
``FUKIKAE_GPU_INDEX``, the host's index of its card, which other cards' leftover scans read.
When the worker stops, the container is removed by name before the lock is released, so a
docker client killed under it can't leave the engines running on a card fukikae thinks is free;
a removal that fails is logged, and the next load removes containers whose fukikae is gone.

With ``[tts] engines`` naming this backend and docker missing or down, a run stops with
``BackendUnavailable``: it never falls back to the PyTorch worker, which would mix two engines'
renders in one episode.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from fukikae.backends.loader import BackendUnavailable
from fukikae.backends.tts import impl_indextts2_trt as _impl
from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
from fukikae.backends.tts.remote import FETCH_NET, WORKER_MODULE, BackendError, nvml_index
from fukikae.gpulock import GPU_INDEX_ENV
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
# Every container this backend starts is named fukikae-indextts2_trt-<fukikae's pid>-<hex>, so
# a later load can tell one whose fukikae is gone (``reap_stale``).
CONTAINER_PREFIX = "fukikae-indextts2_trt-"
_CONTAINER_RE = re.compile(re.escape(CONTAINER_PREFIX) + r"(\d+)-[0-9a-f]+")
# Every container: no capabilities (it runs as the caller's uid anyway), no setuid gains, and
# a bound on processes and threads (torch, TensorRT-LLM and OpenMPI start a few hundred).
HARDENING = ("--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "4096")
RUN_TMPFS = "/tmp:rw,exec,nosuid,nodev,size=4g"  # noqa: S108 - the container's own /tmp
# FETCH_NET variables that name a CA file or directory, mounted read-only into the fetch.
CA_PATHS = frozenset({"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"})
# What setup needs on disk: the image (23.6 GB measured), and the engines (3.1 GB) plus the
# ONNX graphs (4.0 GB) and the converted GPT checkpoint (1.0 GB) that exist while they build.
IMAGE_BYTES = 24 * 10**9
ENGINE_BYTES = 3 * 10**9
INTERMEDIATE_BYTES = 5 * 10**9


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

    def _docker(self, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
        """``docker <args>`` with the client's environment; OSError when there is no docker,
        TimeoutExpired when it doesn't answer."""
        return subprocess.run(  # noqa: S603 - fixed argv
            [DOCKER, *args], capture_output=True, text=True, timeout=timeout, check=False,
            env=self.worker_env(),
        )  # fmt: skip

    def _unavailable(self, problem: str) -> BackendUnavailable:
        return BackendUnavailable(f"tts backend {self.name!r}: {problem}")

    def image_exists(self) -> bool:
        """Whether the image is there: False only when docker says there is no such image.
        An unreachable daemon, a refused socket or no docker at all is ``BackendUnavailable``
        naming that, never "no image", which would send the operator to rebuild it."""
        try:
            r = self._docker("image", "inspect", "--format", "{{.Id}}", self.image())
        except OSError as e:
            raise self._unavailable(f"cannot run {DOCKER}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise self._unavailable(f"{DOCKER} did not answer in 30 s") from e
        if r.returncode == 0:
            return True
        if "no such image" in r.stderr.lower():
            return False
        raise self._unavailable(_docker_trouble(r.stderr))

    def docker_info(self) -> tuple[str, dict[str, Any]]:
        """(the daemon's root directory, its runtimes), or ``BackendUnavailable`` naming why
        docker can't be used."""
        try:
            r = self._docker("info", "--format", "{{.DockerRootDir}} {{json .Runtimes}}")
        except OSError as e:
            raise self._unavailable(f"cannot run {DOCKER}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise self._unavailable(f"{DOCKER} did not answer in 30 s") from e
        if r.returncode != 0:
            raise self._unavailable(_docker_trouble(r.stderr))
        root, _, runtimes = r.stdout.strip().partition(" ")
        try:
            parsed = json.loads(runtimes) if runtimes else {}
        except json.JSONDecodeError:
            parsed = {}
        return root, parsed if isinstance(parsed, dict) else {}

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
        try:
            if not self.image_exists():
                return f"no image {self.image()}"
        except BackendUnavailable as e:
            return str(e).removeprefix(f"tts backend {self.name!r}: ")
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
        # No fallback to the PyTorch worker when docker is missing or down (D30, F5): that
        # would mix two engines' renders in one episode without saying so.
        problem = self.environment_problem()
        if problem:
            raise BackendUnavailable(
                f"tts backend {self.name!r}: {problem}; set it up with `{self.setup_hint()}`, "
                "or drop it from [tts] engines to render on PyTorch"
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
            # the host's card: the worker sees it as device 0 and carries no
            # CUDA_VISIBLE_DEVICES, so other cards' leftover scans read this instead
            GPU_INDEX_ENV: str(self.gpu_index),
        }
        values.update({k: os.environ[k] for k in sorted(CONTAINER_PASS) if k in os.environ})
        out: list[str] = []
        for k, v in values.items():
            out += ["-e", f"{k}={v}"]
        return out

    def container_name(self) -> str:
        """A fresh name carrying this process's pid, which ``reap_stale`` reads back."""
        return f"{CONTAINER_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"

    def command(self) -> list[str]:
        if self.gpu_index is None or self._cwd is None:
            raise RuntimeError("indextts2_trt: the container starts only from load() on a GPU")
        self._container = self.container_name()
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
            *HARDENING,
            # the image read-only; /tmp (HOME, numba's and Triton's caches, the impl's
            # temporary prompts) in memory, exec because Triton loads the helper it builds
            "--read-only", "--tmpfs", RUN_TMPFS,
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
        """``docker rm -f``; when that fails, an error log if the container is still there."""
        try:
            r = self._docker("rm", "-f", name, timeout=60)
            if r.returncode == 0:
                return
            still = self._docker("container", "inspect", "--format", "{{.State.Status}}", name)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.error("could not remove container %s (%s); remove it with `docker rm -f %s`",
                      name, e, name)  # fmt: skip
            return
        if still.returncode == 0:
            log.error(
                "container %s is still there (%s) after `docker rm -f`: %s; remove it by hand "
                "before using that GPU", name, still.stdout.strip(), r.stderr.strip(),
            )  # fmt: skip
        else:
            log.debug("container %s was already gone: %s", name, r.stderr.strip())

    def reap_stale(self) -> list[str]:
        """Remove this backend's containers whose fukikae process (the pid in the name) is
        gone, as a killed run can leave behind; returns their names. A live pid, ours or not,
        is left alone."""
        try:
            r = self._docker("ps", "-a", "--filter", f"name=^{CONTAINER_PREFIX}",
                             "--format", "{{.Names}}")  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("could not list %s containers: %s", self.name, e)
            return []
        if r.returncode != 0:
            log.warning("could not list %s containers: %s", self.name, r.stderr.strip())
            return []
        reaped = []
        for name in r.stdout.split():
            m = _CONTAINER_RE.fullmatch(name)
            if m is None:
                continue
            pid = int(m.group(1))
            if pid == os.getpid() or _alive(pid):
                continue
            log.warning("removing container %s, left by pid %d, which is gone", name, pid)
            self._remove_container(name)
            reaped.append(name)
        return reaped

    def load(self) -> None:
        if self._proc is None and self.device.startswith("cuda"):
            self.reap_stale()  # before the lock: a stale one would be named as a leftover
        super().load()

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

    def _subdir(self, sub: str) -> Path:
        """``<stage>/<sub>``, refused unless it is a real directory: the container can write
        the stage, and a symlink it put there would point into the host."""
        assert self._cwd is not None
        folder = self._cwd / sub
        with contextlib.suppress(FileExistsError):
            folder.mkdir()
        if not stat.S_ISDIR(os.lstat(folder).st_mode):
            raise BackendError(f"{self.name}: the worker replaced {folder} with a non-directory")
        return folder

    def _stage(self, src: Path, sub: str, *, keep: bool) -> Path:
        """A copy of ``src`` in the worker's directory: by content when ``keep`` (reused by
        later lines), else under a fresh name (removed after the line). Written with
        ``O_EXCL | O_NOFOLLOW``, never through a link the container left."""
        src = Path(src).resolve()
        folder = self._subdir(sub)
        if keep:
            known = self._staged.get(str(src))
            if known is not None and _regular(known):
                return known
            digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
            dest = folder / f"{digest}{src.suffix}"
            if not _regular(dest):
                _copy_in(src, dest)
            self._staged[str(src)] = dest
            return dest
        self._serial += 1
        dest = folder / f"{self._serial:06d}{src.suffix}"
        _copy_in(src, dest)
        return dest

    def _take_out(self, staged: Path, dest: Path) -> None:
        """Copy the worker's render at ``staged`` to ``dest``: opened ``O_NOFOLLOW`` relative to
        a directory opened ``O_NOFOLLOW``, and only if it is a regular file, so a symlink the
        container swapped in can't make fukikae copy a host file into the render."""
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        dir_fd = os.open(staged.parent, flags | os.O_DIRECTORY)
        try:
            try:
                fd = os.open(staged.name, flags, dir_fd=dir_fd)
            except OSError as e:
                raise BackendError(f"{self.name}: cannot open the render {staged}: {e}") from e
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise BackendError(f"{self.name}: the render {staged} is not a regular file")
                tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.part")
                try:
                    with tmp.open("xb") as w:
                        shutil.copyfileobj(f, w)
                    os.replace(tmp, dest)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        tmp.unlink()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staged.name, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)

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
        self._serial += 1
        staged_out = self._subdir("out") / f"{self._serial:06d}.wav"
        try:
            seconds = super().synthesize(
                text, voice, language, staged_out,
                resolve=lambda ref: self._stage(resolve(ref), "refs", keep=True),
                emotion_audio=emo, **kw,
            )  # fmt: skip
            self._take_out(staged_out, Path(os.path.abspath(out)))
            return seconds
        finally:
            for p in (emo, staged_out):
                if p is not None:
                    with contextlib.suppress(FileNotFoundError):
                        p.unlink()

    # -- setup ------------------------------------------------------------------------------

    def preflight(self, *, image: bool, engines: bool) -> None:
        """``BackendUnavailable`` unless docker answers, has the NVIDIA runtime, and (``image``:
        the image is to be built; ``engines``: the engines are) has the disk that needs."""
        root, runtimes = self.docker_info()
        if "nvidia" not in runtimes and not any(
            shutil.which(tool) for tool in ("nvidia-container-cli", "nvidia-container-runtime")
        ):
            raise self._unavailable(
                "docker has no nvidia runtime (`docker info` lists none, and there is no "
                "nvidia-container-cli): install the NVIDIA Container Toolkit and run "
                "`sudo nvidia-ctk runtime configure --runtime=docker`"
            )
        if image and root and Path(root).exists():
            free = _free_bytes(Path(root))
            if free < IMAGE_BYTES:
                raise self._unavailable(
                    f"the image needs about {IMAGE_BYTES // 10**9} GB free under docker's root "
                    f"{root}, which has {free / 10**9:.1f} GB"
                )
        if engines:
            where = self.engine_dir().parent
            free = _free_bytes(where)
            if free < ENGINE_BYTES + INTERMEDIATE_BYTES:
                raise self._unavailable(
                    f"the engines need about {(ENGINE_BYTES + INTERMEDIATE_BYTES) // 10**9} GB "
                    f"free at {where} ({ENGINE_BYTES // 10**9} GB of engines plus about "
                    f"{INTERMEDIATE_BYTES // 10**9} GB of ONNX graphs and the converted GPT "
                    f"checkpoint while they build), which has {free / 10**9:.1f} GB"
                )

    def engines_current(self, gpu: int) -> bool:
        """Engines for this project digest exist, built for this GPU's compute capability."""
        manifest = self.manifest()
        have = self.gpu_compute_capability(gpu)
        return bool(manifest and have and manifest.get("compute_capability") == have)

    @classmethod
    def setup(
        cls, *, fetch: bool = True, force: bool = False, run: Callable[..., Any] = subprocess.run
    ) -> None:
        """Build the image, fetch the pinned models (online, the only download) and build the
        engines for this machine's GPU (the first CUDA device, under its lock). Engines already
        built for this project digest and this GPU's compute capability are kept unless
        ``force``. Docker missing or down, no NVIDIA runtime or too little disk is
        ``BackendUnavailable`` before anything starts."""
        from fukikae.gpulock import gpu_lock

        inst = cls({}, "cuda:0")
        if inst.project_digest() is None:
            raise BackendUnavailable(f"{inst.project_dir()} has no uv.lock and Dockerfile")
        gpu = nvml_index("cuda:0")
        build_engines = force or not inst.engines_current(gpu)
        inst.preflight(image=not inst.image_exists(), engines=build_engines)
        client = inst.worker_env()
        log.info("building %s from %s", inst.image(), inst.project_dir())
        run([DOCKER, "build", "-t", inst.image(), str(inst.project_dir())], env=client,
            check=True)  # fmt: skip
        hub = _hub_cache()
        hub.mkdir(parents=True, exist_ok=True)
        user = f"{os.getuid()}:{os.getgid()}"
        base = [DOCKER, "run", "--rm", "--init", "--user", user, *HARDENING,
                "-w", "/tmp",  # noqa: S108
                "-v", f"{inst.checkout() / 'src'}:{SRC_MOUNT}:ro",
                "-e", "HOME=/tmp", "-e", f"HF_HUB_CACHE={hub}",
                "-e", f"PYTHONPATH={SRC_MOUNT}:{_impl.INDEX_TTS}"]  # fmt: skip
        for k, v in TELEMETRY_OFF.items():
            base += ["-e", f"{k}={v}"]
        if fetch:
            log.info("fetching the %s model files", cls.name)
            online = [*base, "-v", f"{hub}:{hub}", "-e", "HF_HUB_OFFLINE=0",
                      "-e", "TRANSFORMERS_OFFLINE=0"]  # fmt: skip
            fetch_client = dict(client)
            # the operator's proxy and CA, as the PyTorch worker's fetch gets them: docker
            # reads each value from its own environment; a CA file or directory is mounted
            for k in sorted(FETCH_NET):
                if k in os.environ:
                    online += ["-e", k]
                    fetch_client[k] = os.environ[k]
                    if k in CA_PATHS and Path(os.environ[k]).exists():
                        ca = os.environ[k]
                        online += ["-v", f"{ca}:{ca}:ro"]
            if os.environ.get("HF_TOKEN"):
                online += ["-e", "HF_TOKEN"]  # docker reads the value from its own environment
                fetch_client["HF_TOKEN"] = os.environ["HF_TOKEN"]
            run([*online, "--entrypoint", "python", inst.image(), "-m", WORKER_MODULE, cls.name,
                 "--fetch"],
                env=fetch_client, check=True)  # fmt: skip
        if not build_engines:
            log.info("engines for %s on compute capability %s are at %s; --force rebuilds them",
                     inst.project_digest(), inst.manifest().get("compute_capability"),  # type: ignore[union-attr]
                     inst.engine_dir())  # fmt: skip
            return
        engines = inst.engine_dir()
        # the build writes a sibling of the engines and renames it into place: mount the parent
        engines.parent.mkdir(parents=True, exist_ok=True)
        log.info("building the TensorRT engines on GPU %d into %s", gpu, engines)
        name = inst.container_name()
        with gpu_lock(gpu, f"{cls.name} engine build"):
            try:
                run([*base, "--name", name, "--gpus", f"device={gpu}", "--network", "none",
                     "--shm-size", "2g",
                     "-v", f"{hub}:{hub}:ro", "-v", f"{engines.parent}:{engines.parent}",
                     "-e", f"FUKIKAE_HUB_REAL={hub}", "-e", "HF_HUB_OFFLINE=1",
                     "-e", f"{GPU_INDEX_ENV}={gpu}",
                     "--entrypoint", "python", inst.image(), "-m",
                     "fukikae.backends.tts.impl_indextts2_trt",
                     "build", str(engines)],
                    env=client, check=True)  # fmt: skip
            finally:
                # a docker client killed under the build leaves the container on the card
                inst._remove_container(name)


def _docker_trouble(stderr: str) -> str:
    """What a failed docker command's stderr says is wrong, in a line."""
    line = next((ln.strip() for ln in stderr.splitlines() if ln.strip()), "no error message")
    low = stderr.lower()
    if "permission denied" in low:
        return (
            f"permission denied on the docker socket ({line}); add this user to the docker "
            "group or point DOCKER_HOST at a daemon it may use"
        )
    if "cannot connect to the docker daemon" in low or "is the docker daemon running" in low:
        return f"the docker daemon is unreachable ({line})"
    return f"docker failed: {line}"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (or its nearest existing ancestor)."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _copy_in(src: Path, dest: Path) -> None:
    """Copy ``src`` to a new file ``dest``, never following or replacing what is there."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(dest, flags, 0o644)
    except FileExistsError as e:
        raise BackendError(f"{dest} is already there and is not a regular file") from e
    with os.fdopen(fd, "wb") as w, src.open("rb") as r:
        shutil.copyfileobj(r, w)
