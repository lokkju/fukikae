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


def test_lock_dir_is_fixed_per_user_whatever_the_session(monkeypatch, tmp_path):
    """B7: an ssh session, a systemd user unit and cron see different XDG_RUNTIME_DIR and
    TMPDIR values; the lock directory must be the same for all of them."""
    monkeypatch.delenv("FUKIKAE_LOCK_DIR", raising=False)
    monkeypatch.setattr(gpulock, "LOCK_ROOT", tmp_path / "root")
    (tmp_path / "root").mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    seen = set()
    other_tmp = tmp_path / "tmpdir"
    for xdg, tmpdir in ((str(runtime), str(other_tmp)), (None, None)):
        for key, value in (("XDG_RUNTIME_DIR", xdg), ("TMPDIR", tmpdir)):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        seen.add(gpulock.lock_dir())
    (d,) = seen
    assert d == tmp_path / "root" / f"fukikae-{os.getuid()}"
    assert d.is_dir() and (d.stat().st_mode & 0o777) == 0o700
    assert gpulock.lock_path(3) == d / "fukikae-gpu3.lock"


def test_a_lock_file_replaced_under_us_is_taken_again(monkeypatch):
    """B7: a tmp cleaner may delete the lock file while it is held; a new job then creates
    another file and flocks that. The acquirer checks after flock that its descriptor is still
    the file at the path, and starts over when it isn't."""
    real = gpulock.fcntl.flock
    swapped = []

    def flock(fd, op):
        real(fd, op)
        if not swapped and op & gpulock.fcntl.LOCK_EX:
            path = gpulock.lock_path(0)
            path.unlink()
            path.touch()
            swapped.append(True)

    monkeypatch.setattr(gpulock.fcntl, "flock", flock)
    with gpu_lock(0, "after a swap", timeout_s=0) as fd:
        assert swapped
        assert os.fstat(fd).st_ino == os.stat(gpulock.lock_path(0)).st_ino


def test_another_thread_waits_for_the_lock():
    """C7/B6: the lock is reentrant per thread, not per process: a second thread (a future
    GPU step in the CPU tail) blocks until the first lets go, or times out."""
    import threading

    taken = threading.Event()
    release = threading.Event()

    def hold():
        with gpu_lock(0, "head", timeout_s=0):
            taken.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    assert taken.wait(5)
    try:
        assert gpulock.held_fds() == ()  # this thread holds nothing
        with pytest.raises(GpuBusy), gpu_lock(0, "tail", timeout_s=0.2, poll_s=0.05):
            pass
        threading.Timer(0.3, release.set).start()
        started = time.monotonic()
        with gpu_lock(0, "tail", timeout_s=5, poll_s=0.05) as fd:
            assert time.monotonic() - started >= 0.25
            assert gpulock.held_fds() == (fd,)
    finally:
        release.set()
        t.join(5)


def test_a_process_with_unreadable_environ_is_not_on_every_card(monkeypatch, caplog):
    """Another user's worker hides its environment; it is skipped (logged), not taken as
    being on every card. A readable environment without CUDA_VISIBLE_DEVICES sees them all."""
    import logging

    def denied(_pid):
        raise PermissionError("denied")

    monkeypatch.setattr(gpulock, "_read_environ", denied)
    with caplog.at_level(logging.DEBUG, logger="fukikae.gpulock"):
        assert gpulock._on_card(4242, 0) is False
    assert any("4242" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: [b"PATH=/bin"])
    assert gpulock._on_card(4242, 0) is True
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: [b"CUDA_VISIBLE_DEVICES=1"])
    assert gpulock._on_card(4242, 0) is False and gpulock._on_card(4242, 1) is True
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: [b"CUDA_VISIBLE_DEVICES="])
    assert gpulock._on_card(4242, 0) is False


def test_unreadable_environs_are_warned_about_once_per_scan(monkeypatch, caplog):
    import logging

    def denied(_pid):
        raise PermissionError("denied")

    monkeypatch.setattr(gpulock, "_read_environ", denied)
    monkeypatch.setattr(gpulock.os, "listdir", lambda _p: ["101", "102", "self"])
    monkeypatch.setattr(gpulock, "_cmdline", lambda _pid: f"python -m {gpulock.WORKER_MARK} x")
    monkeypatch.setattr(gpulock, "_ours", lambda _pid, _me: False)
    with caplog.at_level(logging.WARNING, logger="fukikae.gpulock"):
        assert gpulock.leftover_jobs(0) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "101" in warnings[0].getMessage() and "102" in warnings[0].getMessage()
