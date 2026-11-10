"""Cloning TTS backends that run out of process, each in its own environment.

A ``WorkerSynthesizer`` is a ``Synthesizer`` whose model lives in a worker process
(``fukikae.backends.tts.worker``) started with the interpreter of ``<checkout>/.venvs/<extra>``.
Each worker environment is a uv project of its own under ``<checkout>/workers/<extra>/`` (its
own Python version and lock), so a model's pins never meet fukikae's: IndexTTS2, for one, needs
Python < 3.12 and exact torch, transformers and numpy pins. Set one up with
``fukikae backends setup <name>`` (``WorkerSynthesizer.setup``): ``uv sync --locked`` into
``.venvs/<extra>``, the lock's sha256 recorded there (``LOCK_HASH_FILE``), then the worker's
``--fetch`` step, the only time it downloads (the model's pinned snapshots). ``load`` refuses an
environment whose recorded hash doesn't match ``workers/<extra>/uv.lock``.

The worker's environment is an allowlist (``worker_env``: PATH, HOME, locale, temp, the Hugging
Face and torch caches, ``CUDA_*``), never fukikae's whole environment, which holds the secrets
mise decrypts; at runtime it also sets ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.
``uv sync`` in the setup step gets the same allowlist plus uv's own settings (``sync_env``).
The tests' fake impl hook (``FUKIKAE_TTS_FAKE_IMPL``) is not on it; the tests extend it.

``<checkout>`` is the repository this module was loaded from, or ``FUKIKAE_WORKER_CHECKOUT``.

Lifecycle. On a GPU, ``load`` first takes the card's lock (``fukikae.gpulock``, ruling CD15),
which waits for another fukikae GPU job and refuses when a leftover worker or llama-server from
a run that died still runs there; the worker inherits the lock's descriptor. It starts the
worker with ``CUDA_VISIBLE_DEVICES`` set to that one card (numbered as nvidia-smi does), checks
the protocol version (``hello``) and sends ``load``. The VRAM check comes before ``load`` (the
tts stage's ``LazySynth``, from ``vram_mb``). ``unload`` asks the worker to unload, closes its
stdin, waits ``stop_wait_s``, then terminates and kills it, and always reaps it; a timeout, a
protocol error or the worker's death does the same at once. On Linux the worker also gets
SIGTERM if fukikae itself dies (``PR_SET_PDEATHSIG``), and a finalizer reaps it if the object
is dropped without ``unload``.

Trust. The worker's answers are parsed as JSON objects of at most ``MAX_ANSWER`` bytes; any
other output is a protocol error. Nothing the worker says is used as a path or a command: the
output file is the one fukikae named, checked after the fact (a regular file, not a link, at
most ``MAX_OUTPUT`` bytes, a mono wav at the backend's ``sample_rate``), and its length is read
from the file, not taken from the answer.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, ClassVar, NoReturn

from fukikae.backends.base import Synthesizer
from fukikae.backends.loader import BackendUnavailable
from fukikae.cards.card import CardError
from fukikae.config import ConfigError
from fukikae.gpulock import gpu_lock, held_fds
from fukikae.privacy import TELEMETRY_OFF

if TYPE_CHECKING:
    from fukikae.cards.card import Voice

log = logging.getLogger(__name__)

WORKER_MODULE = "fukikae.backends.tts.worker"
PROTOCOL = 1
MAX_ANSWER = 1 << 20
MAX_OUTPUT = 256 << 20
MAX_RENDER_S = 600.0
SECONDS_SLACK = 0.05
STDERR_LINES = 30
STDERR_KEEP = 2000  # lines of worker stderr held in memory, newest last
STDERR_LINE_MAX = 4096  # bytes per held line; a longer one is split
HELLO_TIMEOUT_S = 60.0
KILL_WAIT_S = 10.0
# A worker that closed its output is given this long to exit before it is killed (B10).
EOF_WAIT_S = 5.0
LOCK_HASH_FILE = "fukikae-uv-lock.sha256"
ENV_ALLOW = frozenset(
    {
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TERM",
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
        "LD_LIBRARY_PATH", "PYTHONPATH",
    }
)  # fmt: skip
ENV_ALLOW_PREFIXES = ("LC_", "CUDA_")
# Network settings the online fetch step gets (never the offline worker).
FETCH_NET = frozenset(
    {
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    }
)  # fmt: skip
# What `uv sync` gets beyond ENV_ALLOW (outside PYTHONPATH): uv's cache, config, Python
# installs, index and network settings. Its sdist builds run build scripts with this
# environment, so it holds no credential uv doesn't need (an index URL may carry one).
SYNC_ALLOW = frozenset(
    {
        "XDG_DATA_HOME", "XDG_CONFIG_HOME",
        "UV_CACHE_DIR", "UV_NO_CACHE", "UV_CONFIG_FILE", "UV_NO_CONFIG",
        "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_DOWNLOADS", "UV_PYTHON_PREFERENCE",
        "UV_INDEX", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL",
        "UV_INDEX_STRATEGY", "UV_KEYRING_PROVIDER", "UV_INSECURE_HOST", "UV_NATIVE_TLS",
        "UV_OFFLINE", "UV_HTTP_TIMEOUT", "UV_LINK_MODE", "UV_CONCURRENT_DOWNLOADS",
        "UV_CONCURRENT_BUILDS", "UV_CONCURRENT_INSTALLS",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    }
)  # fmt: skip


def lock_digest(lock: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(lock).read_bytes()).hexdigest()


def record_lock(project_dir: Path, venv: Path) -> None:
    """Write the sha256 of ``project_dir/uv.lock`` into the environment synced from it."""
    (Path(venv) / LOCK_HASH_FILE).write_text(lock_digest(Path(project_dir) / "uv.lock") + "\n")


class BackendError(RuntimeError):
    """A worker that failed, broke the protocol, timed out or died."""


class _StderrTail:
    """The worker's stderr, drained by a thread into a ring of ``STDERR_KEEP`` lines, so a
    worker that floods it costs bounded memory and no disk."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._lines: collections.deque[str] = collections.deque(maxlen=STDERR_KEEP)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="tts-worker-stderr", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            for raw in iter(lambda: self._stream.readline(STDERR_LINE_MAX), b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                with self._lock:
                    self._lines.append(line)

    def tail(self, n: int, wait: float = 0.0) -> str:
        if wait:
            self._thread.join(wait)
        with self._lock:
            return "\n".join(list(self._lines)[-n:])

    def close(self) -> None:
        self._thread.join(5.0)
        with contextlib.suppress(OSError):
            self._stream.close()


_DEATH_HOOK: list[Callable[[], None] | None] = []


def death_signal_hook() -> Callable[[], None] | None:
    """The worker's ``preexec_fn``: ``prctl(PR_SET_PDEATHSIG, SIGTERM)``, so the worker gets
    SIGTERM when fukikae dies. ctypes resolves ``prctl`` here, in the parent, once (B8): the
    hook runs between fork and exec, where loading a library can deadlock, so it only calls
    the function. None off Linux or when libc can't be loaded."""
    if not _DEATH_HOOK:
        prctl: Any = None
        if sys.platform.startswith("linux"):
            import ctypes

            with contextlib.suppress(OSError, AttributeError):
                prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl

        def call() -> None:  # pragma: no cover - runs in the child
            prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG

        _DEATH_HOOK.append(call if prctl is not None else None)
    return _DEATH_HOOK[0]


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """``sig`` to the worker's whole process group (it leads its own session, B9), so
    anything it started goes with it. Only while the worker is unreaped: the group id is its
    pid, which the kernel may give to another process once the worker is reaped."""
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def _exit_status(proc: subprocess.Popen[bytes]) -> int | None:
    """The worker's exit status once it has exited, else None, without reaping it (unlike
    ``Popen.poll``): the exited worker stays a zombie, keeping its pid and so its group id,
    until ``_reap`` has signalled the group."""
    if proc.returncode is not None:
        return proc.returncode
    try:
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return proc.poll()
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _wait_exit(proc: subprocess.Popen[bytes], timeout: float) -> int | None:
    """``_exit_status`` once the worker exits, or None after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        code = _exit_status(proc)
        if code is not None or time.monotonic() >= deadline:
            return code
        time.sleep(0.02)


def nvml_index(device: str) -> int:
    """The NVML (nvidia-smi) index of a torch ``cuda:<n>`` device in this process."""
    n = int(device.partition(":")[2] or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [int(p) for p in visible.split(",") if p.strip()]
        return ids[n]
    return n


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM, then SIGKILL, to the worker's group, then reap the worker. The group is
    signalled even when the worker has already exited, since it may have left children in
    its group, and before the wait, while the group id can't belong to anyone else (B9)."""
    _signal_group(proc, signal.SIGTERM)
    _wait_exit(proc, KILL_WAIT_S)
    _signal_group(proc, signal.SIGKILL)
    proc.wait()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class WorkerSynthesizer(Synthesizer):
    """Subclasses set ``name``, ``extra``, ``sources``, ``voice_keys``, ``capabilities``,
    ``sample_rate``, ``languages``, ``vram_mb``, ``speech_cps``, ``model``, ``revision`` and
    ``model_revision``, and list the ``[tts]`` options they accept in ``option_keys``."""

    kind: ClassVar[str] = "tts"
    extra: ClassVar[str]
    model: ClassVar[str]
    revision: ClassVar[str]
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    default_device: ClassVar[str | None] = "cuda"  # ruling CD14
    option_keys: ClassVar[frozenset[str]] = frozenset()
    load_timeout_s: ClassVar[float] = 900.0
    synth_timeout_s: ClassVar[float] = 300.0
    stop_wait_s: ClassVar[float] = 30.0
    last_stripped: int = 0  # control markup the worker took out of the last request (CD20)

    def __init__(self, options: dict[str, Any], device: str) -> None:
        unknown = sorted(set(options) - self.option_keys)
        if unknown:
            allowed = ", ".join(sorted(self.option_keys)) or "none"
            raise ConfigError(
                f"the {self.name} TTS backend has no option {', '.join(unknown)} "
                f"(it takes: {allowed})"
            )
        self.options = dict(options)
        self.device = device
        self.pid: int | None = None
        self.gpu_index: int | None = None  # the card's nvml index while the worker runs on one
        self._proc: subprocess.Popen[bytes] | None = None
        self._err: _StderrTail | None = None
        self._cwd: Path | None = None
        self._buf = b""
        self._finalizer: weakref.finalize[Any, Any] | None = None
        self._lock = contextlib.ExitStack()

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        if options:
            raise CardError(
                f"{cls.name} takes no [voice.options] (a {source or 'clone'} card), "
                f"got {', '.join(sorted(options))}"
            )

    # -- where the worker lives ------------------------------------------------------------

    @classmethod
    def checkout(cls) -> Path:
        env = os.environ.get("FUKIKAE_WORKER_CHECKOUT")
        if env:
            return Path(env).expanduser()
        return Path(__file__).resolve().parents[4]

    def project_dir(self) -> Path:
        return self.checkout() / "workers" / self.extra

    def venv(self) -> Path:
        return self.checkout() / ".venvs" / self.extra

    def interpreter(self) -> Path:
        return self.venv() / "bin" / "python"

    def command(self) -> list[str]:
        return [str(self.interpreter()), "-m", WORKER_MODULE, self.name]

    def setup_hint(self) -> str:
        return f"fukikae backends setup {self.name}"

    def lock_revision(self) -> str | None:
        """The first 12 hex of the sha256 of the worker project's ``uv.lock``, for the
        backend's identity in cache keys (B3); None when there is no lock file."""
        lock = self.project_dir() / "uv.lock"
        return lock_digest(lock)[:12] if lock.is_file() else None

    def lock_problem(self) -> str | None:
        """Why the environment doesn't match its project's lock, or None when it does."""
        lock = self.project_dir() / "uv.lock"
        recorded = self.venv() / LOCK_HASH_FILE
        if not lock.is_file():
            return f"{lock} is missing"
        if not recorded.is_file():
            return (
                f"{self.venv()} records no lock hash (it was not set up by `{self.setup_hint()}`)"
            )
        if recorded.read_text().strip() != lock_digest(lock):
            return f"{self.venv()} was synced from another version of {lock}"
        return None

    def environment_location(self) -> Path | str:
        """Where the worker's environment lives, for ``fukikae doctor``: its venv here."""
        return self.venv()

    def environment_problem(self) -> str | None:
        """Why the worker can't start here (its environment missing, or synced from another
        lock), or None when it can."""
        if not self.interpreter().exists():
            return f"no worker environment at {self.venv()}"
        return self.lock_problem()

    def check_environment(self) -> None:
        """``BackendUnavailable`` naming the setup command when the worker can't start."""
        if not self.interpreter().exists():
            raise BackendUnavailable(
                f"tts backend {self.name!r} runs in its own environment, and "
                f"{self.venv()} does not exist; set it up with `{self.setup_hint()}` "
                '(README, "Cloning backends")'
            )
        problem = self.lock_problem()
        if problem:
            raise BackendUnavailable(
                f"tts backend {self.name!r}: {problem}; set it up again with `{self.setup_hint()}`"
            )

    def worker_env(self, *, online: bool = False) -> dict[str, str]:
        """The allowlisted environment the worker gets (never fukikae's secrets); offline
        unless ``online`` (the setup step's download, which may also use ``HF_TOKEN``)."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ENV_ALLOW or k.startswith(ENV_ALLOW_PREFIXES)
        }
        src = str(self.checkout() / "src")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        # no .pyc beside pinned code: bytecode nobody checked must never be imported
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(TELEMETRY_OFF)  # ruling CD24, online and offline alike
        if online:
            if os.environ.get("HF_TOKEN"):
                env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            # the fetch step's downloads go through the proxy and CA the operator set
            env.update({k: v for k, v in os.environ.items() if k in FETCH_NET})
        else:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    @staticmethod
    def sync_env() -> dict[str, str]:
        """The environment ``uv sync`` runs with: the worker allowlist (outside PYTHONPATH)
        and ``SYNC_ALLOW``. fukikae's own venv (VIRTUAL_ENV) and its frozen-sync setting
        (UV_FROZEN) are not the worker's, and fukikae's secrets reach no sdist build script."""
        return {
            k: v
            for k, v in os.environ.items()
            if (k in ENV_ALLOW and k != "PYTHONPATH")
            or k in SYNC_ALLOW
            or k.startswith(ENV_ALLOW_PREFIXES)
        }

    @classmethod
    def setup(
        cls,
        *,
        fetch: bool = True,
        force: bool = False,
        run: Callable[..., Any] = subprocess.run,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Set the worker environment up: ``uv sync --locked`` into ``.venvs/<extra>``, record
        the lock's hash, then (``fetch``) run the worker's ``--fetch`` step online. ``force``
        and ``options`` (the workspace's ``[tts.<backend>]``) change nothing here (the sync is
        idempotent and the same for every option); a subclass with a costly step that depends
        on them (indextts2_trt's engine build) honours them."""
        del force, options
        inst = cls({}, "cpu")
        env = cls.sync_env()
        env["UV_PROJECT_ENVIRONMENT"] = str(inst.venv())
        log.info("syncing %s from %s", inst.venv(), inst.project_dir() / "uv.lock")
        run(["uv", "sync", "--locked", "--directory", str(inst.project_dir())], env=env, check=True)
        record_lock(inst.project_dir(), inst.venv())
        if fetch:
            log.info("fetching the %s model files", cls.name)
            with tempfile.TemporaryDirectory(prefix=f"fukikae-{cls.name}-setup-") as cwd:
                run([*inst.command(), "--fetch"], env=inst.worker_env(online=True), cwd=cwd,
                    check=True)  # fmt: skip

    # -- lifecycle ---------------------------------------------------------------------------

    def load(self) -> None:
        if self._proc is not None:
            return
        self.check_environment()
        env = self.worker_env()
        if self.device.startswith("cuda"):
            gpu = nvml_index(self.device)
            self.gpu_index = gpu
            # Ruling CD15: the card's lock, held until the worker is reaped (the worker gets
            # the descriptor too); taking it names a leftover worker or llama-server.
            self._lock.enter_context(gpu_lock(gpu, f"tts worker {self.name}"))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            worker_device = "cuda:0"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            worker_device = "cpu"
        self._cwd = Path(tempfile.mkdtemp(prefix=f"fukikae-{self.name}-"))
        self._buf = b""
        log.info("starting the %s worker on %s", self.name, self.device)
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - our interpreter and module, no shell
                self.command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self._cwd,
                bufsize=0,
                pass_fds=held_fds(),
                start_new_session=True,  # its own group, killed whole (B9)
                preexec_fn=death_signal_hook(),  # noqa: PLW1509 - resolved in the parent (B8)
            )
        except OSError as e:
            self._stop()
            raise BackendError(f"cannot start the {self.name} worker: {e}") from e
        self.pid = self._proc.pid
        assert self._proc.stderr is not None
        self._err = _StderrTail(self._proc.stderr)
        self._finalizer = weakref.finalize(self, _reap, self._proc)
        try:
            hello = self._request({"op": "hello"}, HELLO_TIMEOUT_S)
            if hello.get("protocol") != PROTOCOL:
                self._fail(
                    f"speaks protocol {hello.get('protocol')!r}, this fukikae speaks {PROTOCOL}"
                )
            started = time.monotonic()
            self._request(
                {"op": "load", "options": _json_safe(self.options), "device": worker_device},
                self.load_timeout_s,
            )
            log.info("%s loaded in %.1f s", self.name, time.monotonic() - started)
        except BaseException:
            self._stop()
            raise

    def unload(self) -> None:
        if self._proc is None:
            self._stop()
            return
        if _exit_status(self._proc) is None:
            try:
                self._request({"op": "unload"}, 60.0)
            except BackendError as e:
                log.warning("%s worker did not unload cleanly: %s", self.name, e)
        self._stop()

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        self.pid = None
        self.gpu_index = None
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    with contextlib.suppress(OSError):
                        proc.stdin.close()
                if _wait_exit(proc, self.stop_wait_s) is None:
                    log.warning("%s worker still running after %g s; stopping it",
                                self.name, self.stop_wait_s)  # fmt: skip
                _reap(proc)
            finally:
                if proc.stdout is not None:
                    proc.stdout.close()
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._err is not None:
            self._err.close()
            self._err = None
        if self._cwd is not None:
            shutil.rmtree(self._cwd, ignore_errors=True)
            self._cwd = None
        self._lock.close()

    # -- the protocol -----------------------------------------------------------------------

    def _stderr_tail(self) -> str:
        if self._err is None:
            return ""
        # a worker that has exited has written all it will; let the drain catch up
        done = self._proc is not None and _exit_status(self._proc) is not None
        return self._err.tail(STDERR_LINES, wait=2.0 if done else 0.2)

    def _fail(self, what: str, *, kill: bool = True) -> NoReturn:
        tail = self._stderr_tail()
        if kill and self._proc is not None:
            _signal_group(self._proc, signal.SIGKILL)
            self._stop()
        message = f"{self.name} worker {what}"
        if tail:
            message += f"\n--- worker stderr (last lines) ---\n{tail}"
        raise BackendError(message)

    def _read_line(self, timeout: float) -> bytes:
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while b"\n" not in self._buf:
                if len(self._buf) > MAX_ANSWER:
                    self._fail(f"sent an answer longer than {MAX_ANSWER} bytes (protocol error)")
                left = deadline - time.monotonic()
                if left <= 0:
                    self._fail(f"did not answer within {timeout:g} s")
                if not sel.select(left):
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    code = _wait_exit(self._proc, EOF_WAIT_S)
                    if code is None:
                        self._fail(f"closed its output and did not exit within {EOF_WAIT_S:g} s")
                    self._fail(f"exited with status {code}")
                self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        if len(line) > MAX_ANSWER:
            self._fail(f"sent an answer longer than {MAX_ANSWER} bytes (protocol error)")
        return line

    def _request(self, req: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None:
            raise BackendError(f"{self.name} worker is not running; load() first")
        try:
            self._proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            code = _exit_status(self._proc)
            self._fail(f"is gone (exit status {code})")
        line = self._read_line(timeout)
        try:
            answer = json.loads(line)
        except ValueError:
            self._fail(f"sent a line that is not JSON (protocol error): {line[:200]!r}")
        if not isinstance(answer, dict) or not isinstance(answer.get("ok"), bool):
            self._fail(f"sent an answer that is not a protocol object: {line[:200]!r}")
        if not answer["ok"]:
            error = answer.get("error")
            self._fail(
                f"failed: {str(error)[:4000] if error is not None else 'no reason given'}",
                kill=False,
            )
        return answer

    # -- synthesis --------------------------------------------------------------------------

    @staticmethod
    def voice_payload(voice: Voice, resolve: Callable[[str], Path]) -> dict[str, Any]:
        """The card's voice as the worker gets it: sidecar refs resolved to absolute paths."""
        d = asdict(voice)
        d.pop("fingerprint", None)
        d["references"] = [
            {
                "clip": str(Path(resolve(r.clip)).resolve()),
                "transcript": r.transcript,
                "language": r.language,
            }
            for r in voice.references
        ]
        if voice.latent:
            d["latent"] = str(Path(resolve(voice.latent)).resolve())
        return _json_safe(d)

    def synthesize(
        self,
        text: str,
        voice: Voice,
        language: str,
        out: Path,
        *,
        resolve: Callable[[str], Path],
        emotion_audio: Path | None = None,
        emotion_vector: dict[str, float] | None = None,
        instruction: str | None = None,
        emotion_strength: float = 1.0,
        target_seconds: float | None = None,
    ) -> float:
        if self._proc is None:
            raise BackendError(f"{self.name}: load() before synthesize()")
        dest = Path(os.path.abspath(out))
        if os.path.lexists(dest):
            dest.unlink()
        answer = self._request(
            {
                "op": "synthesize",
                "text": text,
                "voice": self.voice_payload(voice, resolve),
                "language": language,
                "out": str(dest),
                "emotion_audio": (
                    None if emotion_audio is None else str(Path(emotion_audio).resolve())
                ),
                "emotion_vector": emotion_vector,
                "instruction": instruction,
                "emotion_strength": float(emotion_strength),
                "target_seconds": target_seconds,
            },
            self.synth_timeout_s,
        )
        stripped = answer.get("stripped", 0)
        self.last_stripped = stripped if isinstance(stripped, int) else 0
        seconds = answer.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            self._fail("answered without seconds (protocol error)")
        return self._check_output(dest, float(seconds))

    def _check_output(self, dest: Path, reported: float) -> float:
        """The render's length, read from the file fukikae named, after checking it is a
        regular mono wav at ``sample_rate`` of a sane size that matches the answer."""
        from scipy.io import wavfile

        try:
            st = os.lstat(dest)
        except FileNotFoundError:
            self._fail(f"reported success but wrote no {dest}", kill=False)
        if not stat.S_ISREG(st.st_mode):
            self._fail(f"left something other than a regular file at {dest}", kill=False)
        if st.st_size > MAX_OUTPUT:
            self._fail(f"wrote {st.st_size} bytes to {dest}, over {MAX_OUTPUT}", kill=False)
        try:
            rate, data = wavfile.read(dest, mmap=True)
        except (ValueError, OSError) as e:
            self._fail(f"wrote an unreadable wav to {dest} ({e})", kill=False)
        shape = tuple(data.shape)
        if len(shape) > 1 and shape[1] != 1:
            self._fail(f"wrote {shape[1]} channels to {dest}, not mono", kill=False)
        if rate != self.sample_rate:
            self._fail(f"wrote sample rate {rate} to {dest}, not {self.sample_rate}", kill=False)
        seconds = len(data) / rate
        if not 0 < seconds <= MAX_RENDER_S:
            self._fail(f"wrote {seconds:.2f} seconds of audio to {dest}", kill=False)
        if abs(seconds - reported) > SECONDS_SLACK:
            self._fail(
                f"reported {reported:.3f} seconds, but {dest} holds {seconds:.3f} seconds",
                kill=False,
            )
        return seconds
