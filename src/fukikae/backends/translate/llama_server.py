"""A llama-server process that lives exactly as long as a ``with`` block.

The server runs on one GPU (``CUDA_VISIBLE_DEVICES``, numbered as nvidia-smi does) and listens
on 127.0.0.1 only. Leaving the block, by any exception included, terminates it, waits 10 s,
then kills it, and always reaps it, so no orphan keeps holding the GPU. On Linux the child
also gets SIGTERM if fukikae itself dies (``PR_SET_PDEATHSIG``).
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import logging
import os
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

log = logging.getLogger(__name__)

POLL_S = 0.25
STOP_WAIT_S = 10.0
STDERR_LINES = 20


class LlamaServerError(RuntimeError):
    """llama-server could not be started, exited early, or never became healthy."""


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
        self.pid: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise LlamaServerError("llama-server is not running")
        return f"http://127.0.0.1:{self.port}"

    def args(self) -> list[str]:
        return [
            self.binary, "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.ctx), "-ngl", "999", "--jinja",
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
        if self.port is None:
            self.port = _free_port()
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - closed in _stop
        args = self.args()
        log.info(
            "starting %s on GPU %d, port %d", Path(self.binary).name, self.gpu_index, self.port
        )
        try:
            self._proc = self._popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,  # noqa: PLW1509
            )
        except OSError as e:
            self._close_log()
            raise LlamaServerError(f"cannot start {self.binary!r}: {e}") from e
        self.pid = self._proc.pid
        try:
            self._wait_healthy()
        except BaseException:
            self._stop()
            raise
        return self

    def _wait_healthy(self) -> None:
        assert self._proc is not None
        start = time.monotonic()
        url = f"{self.base_url}/health"
        while True:
            code = self._proc.poll()
            if code is not None:
                raise LlamaServerError(
                    f"llama-server exited with status {code} before it was ready:\n"
                    + self._stderr_tail()
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

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    log.warning("llama-server ignored SIGTERM for %g s; killing it", STOP_WAIT_S)
                    proc.kill()
                    proc.wait()
            elif proc is not None:
                proc.wait()
        finally:
            self._close_log()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop()
