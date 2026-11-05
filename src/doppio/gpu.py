"""VRAM gate: wait for free GPU memory before a stage loads a model."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from doppio.config import Config


class VramTimeout(Exception):  # noqa: N818
    """Raised when a device never reaches the free memory a stage needs."""


def free_mb(device: int) -> int:
    """Free memory on an NVIDIA device in MiB. Imports pynvml lazily."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        free_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)  # type: ignore[arg-type]
        return free_bytes // (1024 * 1024)
    finally:
        pynvml.nvmlShutdown()


def _stderr_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def wait_for_vram(
    device: int,
    need_mb: int,
    timeout_s: float | None,
    poll_s: float = 15.0,
    query: Callable[[int], int] = free_mb,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = _stderr_log,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until `device` has `need_mb` free; None timeout waits forever, 0 fails at once."""
    start = clock()
    logged = False
    while True:
        free = query(device)
        if free >= need_mb:
            return
        waited = clock() - start
        if timeout_s is not None and waited >= timeout_s:
            raise VramTimeout(
                f"GPU {device}: need {need_mb} MiB free, last saw {free} MiB after {waited:.0f} s"
            )
        if not logged:
            log(f"waiting for GPU {device}: need {need_mb} MiB free, {free} MiB free now")
            logged = True
        pause = poll_s
        if timeout_s is not None:
            pause = min(pause, timeout_s - waited)
        sleep(pause)


def device_string(cfg: Config) -> str:
    """`cuda:<n>` when NVML sees the configured device, else `cpu`. Never raises."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if 0 <= cfg.gpu_device < pynvml.nvmlDeviceGetCount():
                pynvml.nvmlDeviceGetHandleByIndex(cfg.gpu_device)
                return f"cuda:{cfg.gpu_device}"
            return "cpu"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return "cpu"
