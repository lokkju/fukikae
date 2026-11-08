"""VRAM gate: wait for free GPU memory before a stage loads a model."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable

from fukikae.config import Config


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


def gpu_processes(device: int) -> list[tuple[int, str]]:
    """(pid, process name) of the compute processes on an NVIDIA device, as nvidia-smi lists
    them; a name this host can't read is "?". Empty when NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            pids = [int(p.pid) for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []
    out: list[tuple[int, str]] = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                out.append((pid, f.read().strip()))
        except OSError:
            out.append((pid, "?"))
    return out


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


def _torch_index(nvml_index: int) -> int | None:
    """Torch's index for an NVML (PCI bus order) card; None when CUDA_VISIBLE_DEVICES hides it."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return nvml_index
    try:
        ids = [int(p) for p in visible.split(",") if p.strip()]
    except ValueError:
        _stderr_log(f"CUDA_VISIBLE_DEVICES={visible!r} is not a list of indices; using cpu")
        return None
    if nvml_index not in ids:
        _stderr_log(f"GPU {nvml_index} is not in CUDA_VISIBLE_DEVICES={visible}; using cpu")
        return None
    return ids.index(nvml_index)


def device_string(cfg: Config) -> str:
    """`cuda:<n>` (torch's index for the configured NVML card), else `cpu`. Never raises."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if 0 <= cfg.gpu_device < pynvml.nvmlDeviceGetCount():
                pynvml.nvmlDeviceGetHandleByIndex(cfg.gpu_device)
                idx = _torch_index(cfg.gpu_device)
                return "cpu" if idx is None else f"cuda:{idx}"
            return "cpu"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return "cpu"
