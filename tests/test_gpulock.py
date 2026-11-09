"""The per-GPU lock every fukikae GPU job holds (ruling CD15)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fukikae import gpulock
from fukikae.gpulock import GpuBusy, gpu_lock

SRC = str(Path(__file__).resolve().parents[1] / "src")

_HOLDER = """
import sys, time
from fukikae.gpulock import gpu_lock
with gpu_lock(0, "test holder", timeout_s=5):
    print("locked", flush=True)
    time.sleep(float(sys.argv[1]))
print("released", flush=True)
"""


@pytest.fixture(autouse=True)
def _same_lock_dir(tmp_path, monkeypatch):
    """The test and the processes it starts share one lock dir."""
    monkeypatch.setenv("FUKIKAE_LOCK_DIR", str(tmp_path))


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": SRC, "FUKIKAE_LOCK_DIR": str(tmp_path), **extra}


def test_two_concurrent_acquirers_serialize(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, "1.5"], stdout=subprocess.PIPE, text=True,
        env=_env(tmp_path),
    )  # fmt: skip
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        started = time.monotonic()
        with gpu_lock(0, "test waiter", timeout_s=20, poll_s=0.05):
            waited = time.monotonic() - started
            # the holder let go before we got it
            assert holder.wait(timeout=5) == 0
        assert waited >= 1.0
    finally:
        holder.kill()
        holder.wait()


def test_a_busy_lock_times_out_naming_the_holder(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, "10"], stdout=subprocess.PIPE, text=True,
        env=_env(tmp_path),
    )  # fmt: skip
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with (
            pytest.raises(GpuBusy, match=rf"pid {holder.pid}.*test holder"),
            gpu_lock(0, "test waiter", timeout_s=0.3, poll_s=0.05),
        ):
            pass
    finally:
        holder.kill()
        holder.wait()


def test_the_lock_is_reentrant_within_a_process_and_exposes_its_fd():
    assert gpulock.held_fds() == ()
    with gpu_lock(0, "outer", timeout_s=0) as fd:
        with gpu_lock(0, "inner", timeout_s=0) as fd2:
            assert fd == fd2
            assert gpulock.held_fds() == (fd,)
        assert gpulock.held_fds() == (fd,)
    assert gpulock.held_fds() == ()


def test_another_gpu_has_its_own_lock(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, "5"], stdout=subprocess.PIPE, text=True,
        env=_env(tmp_path),
    )  # fmt: skip
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with gpu_lock(1, "other card", timeout_s=0):
            pass
    finally:
        holder.kill()
        holder.wait()


def _detached(tmp_path: Path, argv_mark: list[str], cuda: str | None) -> int:
    """Start a sleeper that is not our descendant (its parent exits), with ``argv_mark`` on
    its command line, and return its pid."""
    env = _env(tmp_path)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    launcher = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', *sys.argv[1:]],"
        " start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,"
        " stderr=subprocess.DEVNULL); print(p.pid)"
    )
    out = subprocess.run(
        [sys.executable, "-c", launcher, *argv_mark], capture_output=True, text=True, env=env,
        check=True, timeout=30,
    )  # fmt: skip
    pid = int(out.stdout.strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not gpulock._cmdline(pid):
        time.sleep(0.02)
    return pid


@pytest.mark.parametrize(
    "mark",
    [
        ["-m", "fukikae.backends.tts.worker", "indextts2"],
        ["llama-server", "--api-key-file", "/run/user/0/fukikae-llama-key-abc"],
    ],
)
def test_a_leftover_is_found_by_its_command_line_and_named(tmp_path, mark):
    pid = _detached(tmp_path, mark, "0")
    try:
        found = gpulock.leftover_jobs(0)
        assert pid in [p for p, _ in found]
        with pytest.raises(GpuBusy, match=rf"pid {pid}"), gpu_lock(0, "a new job", timeout_s=0):
            pass
        # the lock was let go when the leftover check failed
        assert gpulock.held_fds() == ()
    finally:
        os.kill(pid, signal.SIGKILL)


def test_leftovers_on_another_card_or_the_cpu_are_not_ours(tmp_path):
    other = _detached(tmp_path, ["-m", "fukikae.backends.tts.worker", "x"], "1")
    cpu = _detached(tmp_path, ["-m", "fukikae.backends.tts.worker", "x"], "")
    try:
        found = [p for p, _ in gpulock.leftover_jobs(0)]
        assert other not in found and cpu not in found
    finally:
        os.kill(other, signal.SIGKILL)
        os.kill(cpu, signal.SIGKILL)


def test_our_own_children_are_not_leftovers(tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "fukikae.backends.tts.worker"],
        env=_env(tmp_path, CUDA_VISIBLE_DEVICES="0"),
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "worker" not in gpulock._cmdline(child.pid):
            time.sleep(0.02)
        assert child.pid not in [p for p, _ in gpulock.leftover_jobs(0)]
    finally:
        child.kill()
        child.wait()


def test_lock_dir_falls_back_when_no_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("FUKIKAE_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gpulock.tempfile, "gettempdir", lambda: str(tmp_path))
    d = gpulock.lock_dir()
    assert d == tmp_path / f"fukikae-{os.getuid()}"
    assert d.is_dir() and (d.stat().st_mode & 0o777) == 0o700
    assert gpulock.lock_path(3) == d / "fukikae-gpu3.lock"
