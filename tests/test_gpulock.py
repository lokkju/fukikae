"""The per-GPU lock every fukikae GPU job holds (ruling CD15)."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
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
        # giving up leaves the holder's line alone
        assert gpulock.lock_path(0).read_text() == f"{holder.pid} test holder\n"
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


def _detached(
    tmp_path: Path,
    started: set[int],
    argv: list[str],
    cuda: str | None,
    executable: str | None = None,
) -> int:
    """Start ``argv`` so that it is not our descendant (its parent exits), add its pid to the
    pids the leftover scan may see (``started``), and return it. ``executable`` runs in place
    of ``argv[0]``, which then is only the name the process shows."""
    env = _env(tmp_path)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    launcher = (
        "import json, subprocess, sys; a = json.loads(sys.argv[1]); "
        "p = subprocess.Popen(a['argv'], executable=a['exe'], start_new_session=True,"
        " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);"
        " print(p.pid)"
    )
    spec = json.dumps({"argv": argv, "exe": executable})
    out = subprocess.run(
        [sys.executable, "-c", launcher, spec], capture_output=True, text=True, env=env,
        check=True, timeout=30,
    )  # fmt: skip
    pid = int(out.stdout.strip())
    started.add(pid)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not gpulock._argv(pid):
        time.sleep(0.02)
    return pid


_SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.mark.parametrize(
    ("argv", "exe"),
    [
        ([*_SLEEP, "-m", "fukikae.backends.tts.worker", "indextts2"], None),
        (
            ["llama-server", "-c", "import time; time.sleep(60)", "--api-key-file",
             "/run/user/0/fukikae-llama-key-abc"],
            sys.executable,
        ),
    ],
)  # fmt: skip
def test_a_leftover_is_found_by_its_command_line_and_named(tmp_path, started_pids, argv, exe):
    pid = _detached(tmp_path, started_pids, argv, "0", exe)
    try:
        found = gpulock.leftover_jobs(0)
        assert pid in [p for p, _ in found]
        with pytest.raises(GpuBusy, match=rf"pid {pid}"), gpu_lock(0, "a new job", timeout_s=0):
            pass
        # the lock was let go when the leftover check failed
        assert gpulock.held_fds() == ()
    finally:
        os.kill(pid, signal.SIGKILL)


def test_a_shell_that_only_mentions_the_worker_is_not_a_leftover(tmp_path, started_pids):
    """The scan reads the argv array: a ``bash -c "... fukikae.backends.tts.worker ..."`` (or a
    grep, an editor) names the module inside one argument and is not a worker."""
    pid = _detached(
        tmp_path, started_pids,
        ["bash", "-c", "sleep 60; : python -m fukikae.backends.tts.worker x"], "0",
    )  # fmt: skip
    try:
        assert pid not in [p for p, _ in gpulock.leftover_jobs(0)]
    finally:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize(
    ("argv", "job"),
    [
        (["/usr/bin/python3", "-m", "fukikae.backends.tts.worker", "indextts2"], True),
        (["python", "-u", "-m", "fukikae.backends.tts.worker"], True),
        (["python", "-m", "fukikae.backends.tts.worker.extra"], False),
        (["python", "fukikae.backends.tts.worker"], False),
        (["bash", "-c", "python -m fukikae.backends.tts.worker x"], False),
        (["/opt/bin/llama-server", "--api-key-file", "/run/k/fukikae-llama-key-ab"], True),
        (["llama-server", "--api-key-file=/run/k/fukikae-llama-key-ab"], True),
        (["llama-server", "--api-key-file", "/run/k/other-key"], False),
        (["llama-server", "/run/k/fukikae-llama-key-ab"], False),
        (["vim", "llama-server", "--api-key-file", "/run/k/fukikae-llama-key-ab"], False),
        ([], False),
    ],
)
def test_a_job_is_recognised_by_its_argv(argv, job):
    assert gpulock._is_job(argv) is job


def test_the_holder_pid_is_written_before_the_leftover_scan(monkeypatch):
    """B7: a waiter that finds the lock taken reads who holds it, so the pid goes in at once,
    not after the /proc scan."""
    seen: list[str] = []

    def scan(index):
        seen.append(gpulock.lock_path(index).read_text())
        return []

    monkeypatch.setattr(gpulock, "leftover_jobs", scan)
    with gpu_lock(0, "early bird", timeout_s=0):
        pass
    assert seen == [f"{os.getpid()} early bird\n"]


def test_a_failed_leftover_check_leaves_no_holder_behind(monkeypatch):
    monkeypatch.setattr(gpulock, "leftover_jobs", lambda _i: [(4242, "llama-server x")])
    with pytest.raises(GpuBusy), gpu_lock(0, "refused", timeout_s=0):
        pass
    assert gpulock.lock_path(0).read_text() == ""


def test_leftovers_on_another_card_or_the_cpu_are_not_ours(tmp_path, started_pids):
    worker = [*_SLEEP, "-m", "fukikae.backends.tts.worker", "x"]
    other = _detached(tmp_path, started_pids, worker, "1")
    cpu = _detached(tmp_path, started_pids, worker, "")
    try:
        found = [p for p, _ in gpulock.leftover_jobs(0)]
        assert other not in found and cpu not in found
    finally:
        os.kill(other, signal.SIGKILL)
        os.kill(cpu, signal.SIGKILL)


def test_our_own_children_are_not_leftovers(tmp_path, started_pids):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "-m", "fukikae.backends.tts.worker"],
        env=_env(tmp_path, CUDA_VISIBLE_DEVICES="0"),
    )  # fmt: skip
    started_pids.add(child.pid)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and gpulock.WORKER_MARK not in gpulock._argv(child.pid):
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
    monkeypatch.setattr(gpulock, "_proc_pids", lambda: [101, 102])
    monkeypatch.setattr(gpulock, "_argv", lambda _pid: ["python", "-m", gpulock.WORKER_MARK])
    monkeypatch.setattr(gpulock, "_ours", lambda _pid, _me: False)
    with caplog.at_level(logging.WARNING, logger="fukikae.gpulock"):
        assert gpulock.leftover_jobs(0) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "101" in warnings[0].getMessage() and "102" in warnings[0].getMessage()


def test_fukikae_gpu_index_names_the_card_before_cuda_visible_devices(monkeypatch):
    """A worker in a container (indextts2_trt) sees its one card as device 0 and has no
    CUDA_VISIBLE_DEVICES at all; FUKIKAE_GPU_INDEX names the host's card for it."""
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: [b"FUKIKAE_GPU_INDEX=1"])
    assert gpulock._on_card(4242, 1) is True and gpulock._on_card(4242, 0) is False
    both = [b"CUDA_VISIBLE_DEVICES=0", b"FUKIKAE_GPU_INDEX=2"]
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: both)
    assert gpulock._on_card(4242, 2) is True and gpulock._on_card(4242, 0) is False
    monkeypatch.setattr(gpulock, "_read_environ", lambda _pid: [b"FUKIKAE_GPU_INDEX=x"])
    assert gpulock._on_card(4242, 0) is True  # unreadable value: on every card, as before


def test_a_waiter_names_the_holder_and_how_long_it_has_held_the_card(monkeypatch, caplog, tmp_path):
    """#29: "waiting for GPU N, held by pid P (<what>) for <dur>" at once and every
    WAIT_LOG_S, and the wait shows in status.json while it lasts."""
    import fcntl
    import io
    import json
    import logging

    from fukikae import progress

    path = gpulock.lock_path(0)
    path.write_text("4242 tts indextts2\n")
    os.utime(path, (time.time() - 240, time.time() - 240))  # taken 4 minutes ago
    fake_holder = os.open(path, os.O_RDWR)
    fcntl.flock(fake_holder, fcntl.LOCK_EX)
    monkeypatch.setattr(gpulock, "_alive", lambda pid: pid == 4242 or pid == os.getpid())
    monkeypatch.setattr(gpulock, "WAIT_LOG_S", 0.2)
    caplog.set_level(logging.WARNING, logger="fukikae.gpulock")
    statuses: list[list[dict]] = []

    def watch():
        deadline = time.monotonic() + 0.7
        while time.monotonic() < deadline:
            data = progress.read_status(tmp_path)
            if data and data["gpu"]:
                statuses.append(data["gpu"])
            time.sleep(0.05)
        os.close(fake_holder)  # lets go of the lock

    try:
        with progress.run(tmp_path, ["S01E12"], stream=io.StringIO(), tty=False, status_every=0):
            t = threading.Thread(target=watch)
            t.start()
            with gpu_lock(0, "asr", timeout_s=5, poll_s=0.05):
                held = json.loads((tmp_path / ".fukikae" / "status.json").read_text())["gpu"]
            t.join(5)
    finally:
        with contextlib.suppress(OSError):
            os.close(fake_holder)
    waits = [r.getMessage() for r in caplog.records if r.getMessage().startswith("waiting for GPU")]
    assert waits[0] == "waiting for GPU 0, held by pid 4242 (tts indextts2) for 4 min 0 s"
    assert 2 <= len(waits) <= 6  # repeated every WAIT_LOG_S, not every poll
    assert statuses and statuses[0][0]["state"] == "waiting"
    assert statuses[0][0]["holder_pid"] == 4242 and statuses[0][0]["holder"] == "tts indextts2"
    assert held == [{"gpu": 0, "thread": "head", "state": "held", "holder_pid": os.getpid(),
                     "holder": "asr", "held_since": held[0]["held_since"]}]  # fmt: skip
