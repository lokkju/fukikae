"""A llama-server process that lives exactly as long as a ``with`` block.

The server runs on one GPU (``CUDA_VISIBLE_DEVICES``, numbered as nvidia-smi does) and listens
on 127.0.0.1 only. Leaving the block, by any exception included, terminates it, waits 10 s,
then kills it, and always reaps it, so no orphan keeps holding the GPU. On Linux the child
also gets SIGTERM if fukikae itself dies (``PR_SET_PDEATHSIG``).

Each run makes a random API key (in a 0600 file, never on the command line), so only this
process can use the server; after ``/health`` answers, ``/props`` must accept that key and
name our GGUF, which proves the server on the port is the one we started. A port another
process takes between choosing it and the server binding it is retried once.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from fukikae.gpulock import held_fds

log = logging.getLogger(__name__)

POLL_S = 0.25
STOP_WAIT_S = 10.0
STDERR_LINES = 20


class LlamaServerError(RuntimeError):
    """llama-server could not be started, exited early, or never became healthy."""


class _PortTaken(Exception):  # noqa: N818 - internal signal, never escapes
    """llama-server exited because its port was taken."""


def _http_get(url: str, timeout: float) -> int:
    """Status code of a GET; raises OSError when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - our own localhost URL
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _die_with_parent() -> None:  # pragma: no cover - runs in the child between fork and exec
    with contextlib.suppress(Exception):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class LlamaServer:
    def __init__(
        self,
        binary: str,
        gguf: Path,
        gpu_index: int,
        *,
        ctx: int = 8192,
        port: int | None = None,
        startup_timeout_s: float = 300.0,
        popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        http_get: Callable[[str, float], int] = _http_get,
    ) -> None:
        self.binary = binary
        self.gguf = Path(gguf)
        self.gpu_index = gpu_index
        self.ctx = ctx
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self._popen = popen
        self._http_get = http_get
        self._proc: subprocess.Popen[Any] | None = None
        self._log: IO[bytes] | None = None
        self._key_file: Path | None = None
        self.api_key: str | None = None
        self.pid: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise LlamaServerError("llama-server is not running")
        return f"http://127.0.0.1:{self.port}"

    def args(self) -> list[str]:
        assert self._key_file is not None
        return [
            self.binary, "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.ctx), "-ngl", "999", "--jinja", "--api-key-file", str(self._key_file),
        ]  # fmt: skip

    def _stderr_tail(self) -> str:
        if self._log is None:
            return ""
        self._log.flush()
        self._log.seek(0)
        lines = collections.deque(
            self._log.read().decode("utf-8", errors="replace").splitlines(), STDERR_LINES
        )
        return "\n".join(lines)

    def __enter__(self) -> LlamaServer:
        if shutil.which(self.binary) is None:
            raise LlamaServerError(
                f"llama-server binary {self.binary!r} not found; install it with mise "
                "(`mise install` in the fukikae checkout) or set [translate] binary to its path"
            )
        # A per-run key, in a 0600 file rather than on the command line, so only this process
        # can use the server, and /props (which needs it) proves we reached our own server.
        self.api_key = secrets.token_hex(24)
        fd, name = tempfile.mkstemp(prefix="fukikae-llama-key-")
        with os.fdopen(fd, "w") as f:
            f.write(self.api_key + "\n")
        self._key_file = Path(name)
        try:
            pick_port = self.port is None
            for attempt in (1, 2):
                if pick_port:
                    self.port = _free_port()
                self._start()
                try:
                    self._wait_healthy()
                    self._check_identity()
                    return self
                except _PortTaken:
                    # another process took the free port between our probe and the bind
                    self._stop_process()
                    if attempt == 2 or not pick_port:
                        raise LlamaServerError(
                            f"llama-server could not bind port {self.port}:\n" + self._stderr_tail()
                        ) from None
                    log.warning(
                        "port %d was taken before llama-server bound it; retrying", self.port
                    )
        except BaseException:
            self._stop()
            raise
        raise AssertionError("unreachable")

    def _start(self) -> None:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        self._close_log()
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in _stop
        log.info(
            "starting %s on GPU %d, port %d", Path(self.binary).name, self.gpu_index, self.port
        )
        try:
            self._proc = self._popen(
                self.args(),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=env,
                # the GPU lock (ruling CD15) stays held while the server lives
                pass_fds=held_fds(),
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,  # noqa: PLW1509
            )
        except OSError as e:
            raise LlamaServerError(f"cannot start {self.binary!r}: {e}") from e
        self.pid = self._proc.pid

    def _wait_healthy(self) -> None:
        assert self._proc is not None
        start = time.monotonic()
        url = f"{self.base_url}/health"
        while True:
            code = self._proc.poll()
            if code is not None:
                tail = self._stderr_tail()
                if "bind" in tail.lower():
                    raise _PortTaken
                raise LlamaServerError(
                    f"llama-server exited with status {code} before it was ready:\n" + tail
                )
            with contextlib.suppress(OSError):
                if self._http_get(url, 2.0) == 200:
                    log.info("llama-server ready after %.1f s", time.monotonic() - start)
                    return
            waited = time.monotonic() - start
            if waited >= self.startup_timeout_s:
                raise LlamaServerError(
                    f"llama-server not healthy after {self.startup_timeout_s:g} s:\n"
                    + self._stderr_tail()
                )
            time.sleep(POLL_S)

    def _check_identity(self) -> None:
        """/props must accept our key and name our model file: the answering server is ours."""
        req = urllib.request.Request(  # noqa: S310 - our own localhost URL
            f"{self.base_url}/props", headers={"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
                props = json.loads(r.read())
        except (OSError, ValueError) as e:
            raise LlamaServerError(
                f"the server on port {self.port} did not accept this run's key on /props "
                f"({type(e).__name__}); it is not the llama-server fukikae started"
            ) from None
        found = str(props.get("model_path", ""))
        if not found or Path(found).resolve() != self.gguf.resolve():
            raise LlamaServerError(
                f"the server on port {self.port} serves {found or 'no model'!r}, not "
                f"{self.gguf}; it is not the llama-server fukikae started"
            )

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                log.warning("llama-server ignored SIGTERM for %g s; killing it", STOP_WAIT_S)
                proc.kill()
                proc.wait()
        else:
            proc.wait()

    def _stop(self) -> None:
        try:
            self._stop_process()
        finally:
            self._close_log()
            if self._key_file is not None:
                self._key_file.unlink(missing_ok=True)
                self._key_file = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop()
