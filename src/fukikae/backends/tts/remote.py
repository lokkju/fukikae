"""Cloning TTS backends that run out of process, each in its own environment.

A ``WorkerSynthesizer`` is a ``Synthesizer`` whose model lives in a worker process
(``fukikae.backends.tts.worker``) started with the interpreter of ``<checkout>/.venvs/<extra>``.
Each worker environment is a uv project of its own under ``<checkout>/workers/<extra>/`` (its
own Python version and lock), so a model's pins never meet fukikae's: IndexTTS2, for one, needs
Python < 3.12 and exact torch, transformers and numpy pins. Set one up with::

    UV_PROJECT_ENVIRONMENT=<checkout>/.venvs/<extra> \
        uv sync --directory <checkout>/workers/<extra> --frozen

``<checkout>`` is the repository this module was loaded from, or ``FUKIKAE_WORKER_CHECKOUT``.

Lifecycle. ``load`` refuses to start when another fukikae TTS worker already runs on the same
GPU (a leftover from a run that died), starts the worker with ``CUDA_VISIBLE_DEVICES`` set to
that one card (numbered as nvidia-smi does), checks the protocol version (``hello``) and sends
``load``. The VRAM check comes before ``load`` (the tts stage's ``LazySynth``, from
``vram_mb``). ``unload`` asks the worker to unload, closes its stdin, waits ``stop_wait_s``,
then terminates and kills it, and always reaps it; a timeout, a protocol error or the worker's
death does the same at once. On Linux the worker also gets SIGTERM if fukikae itself dies
(``PR_SET_PDEATHSIG``), and a finalizer reaps it if the object is dropped without ``unload``.

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
from fukikae.gpu import gpu_processes

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
HELLO_TIMEOUT_S = 60.0
KILL_WAIT_S = 10.0


class BackendError(RuntimeError):
    """A worker that failed, broke the protocol, timed out or died."""


def _die_with_parent() -> None:  # pragma: no cover - runs in the child between fork and exec
    import ctypes

    with contextlib.suppress(Exception):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def nvml_index(device: str) -> int:
    """The NVML (nvidia-smi) index of a torch ``cuda:<n>`` device in this process."""
    n = int(device.partition(":")[2] or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [int(p) for p in visible.split(",") if p.strip()]
        return ids[n]
    return n


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def leftover_workers(gpu: int) -> list[tuple[int, str]]:
    """(pid, command line) of fukikae TTS workers holding ``gpu`` (an NVML index)."""
    return [(pid, cmd) for pid, _ in gpu_processes(gpu) if WORKER_MODULE in (cmd := _cmdline(pid))]


def _reap(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    else:
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
    option_keys: ClassVar[frozenset[str]] = frozenset()
    load_timeout_s: ClassVar[float] = 900.0
    synth_timeout_s: ClassVar[float] = 300.0
    stop_wait_s: ClassVar[float] = 30.0

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
        self._proc: subprocess.Popen[bytes] | None = None
        self._err: IO[bytes] | None = None
        self._cwd: Path | None = None
        self._buf = b""
        self._finalizer: weakref.finalize[Any, Any] | None = None

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
        return (
            f"UV_PROJECT_ENVIRONMENT={self.venv()} uv sync --directory {self.project_dir()} "
            "--frozen"
        )

    # -- lifecycle ---------------------------------------------------------------------------

    def load(self) -> None:
        if self._proc is not None:
            return
        if not self.interpreter().exists():
            raise BackendUnavailable(
                f"tts backend {self.name!r} runs in its own environment, and "
                f"{self.venv()} does not exist; set it up with `{self.setup_hint()}` "
                '(README, "Cloning backends")'
            )
        env = dict(os.environ)
        src = str(self.checkout() / "src")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
        env["PYTHONUNBUFFERED"] = "1"
        if self.device.startswith("cuda"):
            gpu = nvml_index(self.device)
            found = leftover_workers(gpu)
            if found:
                listed = ", ".join(f"pid {pid} ({cmd})" for pid, cmd in found)
                raise BackendError(
                    f"GPU {gpu} already runs a fukikae TTS worker: {listed}, probably left over "
                    "from an earlier run; stop it (`kill <pid>`) and run again"
                )
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            worker_device = "cuda:0"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            worker_device = "cpu"
        self._err = tempfile.TemporaryFile()  # noqa: SIM115 - closed in _stop
        self._cwd = Path(tempfile.mkdtemp(prefix=f"fukikae-{self.name}-"))
        self._buf = b""
        log.info("starting the %s worker on %s", self.name, self.device)
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - our interpreter and module, no shell
                self.command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._err,
                env=env,
                cwd=self._cwd,
                bufsize=0,
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,  # noqa: PLW1509
            )
        except OSError as e:
            self._stop()
            raise BackendError(f"cannot start the {self.name} worker: {e}") from e
        self.pid = self._proc.pid
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
        if self._proc.poll() is None:
            try:
                self._request({"op": "unload"}, 60.0)
            except BackendError as e:
                log.warning("%s worker did not unload cleanly: %s", self.name, e)
        self._stop()

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        self.pid = None
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    with contextlib.suppress(OSError):
                        proc.stdin.close()
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=self.stop_wait_s)
                    except subprocess.TimeoutExpired:
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

    # -- the protocol -----------------------------------------------------------------------

    def _stderr_tail(self) -> str:
        if self._err is None:
            return ""
        self._err.flush()
        self._err.seek(0)
        lines = collections.deque(
            self._err.read().decode("utf-8", errors="replace").splitlines(), STDERR_LINES
        )
        self._err.seek(0, os.SEEK_END)
        return "\n".join(lines)

    def _fail(self, what: str, *, kill: bool = True) -> NoReturn:
        tail = self._stderr_tail()
        if kill and self._proc is not None:
            proc = self._proc
            if proc.poll() is None:
                proc.kill()
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
                    code = self._proc.wait()
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
            code = self._proc.poll()
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
