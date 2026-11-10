"""One fukikae GPU job per card at a time (ruling CD15).

Every fukikae job that puts a model on a GPU (a GPU stage, the managed llama-server, a TTS
worker, an eval model) holds ``gpu_lock(<nvml index>)`` for as long as the model lives: an
``flock`` on ``<dir>/fukikae-gpu<N>.lock``, where ``<dir>`` is ``FUKIKAE_LOCK_DIR``, else
``/tmp/fukikae-<uid>`` (mode 0700): a fixed path, never ``$XDG_RUNTIME_DIR`` or ``$TMPDIR``,
which differ between an ssh session, a systemd user unit and cron, so two runs would lock
different files (B7). After each ``flock`` the descriptor is checked to still be the file at
that path (a tmp cleaner may have replaced it), else the lock is taken again. A second job on
the same card waits for the first (``timeout_s``: the run's ``--no-wait``/VRAM timeout,
``runctx``) instead of checking free memory, starting, and finding out 70 s later that both
loaded.

The lock is reentrant per thread: a stage that holds it may start a worker that takes it
again, while another thread of the same process (a GPU step in the CPU tail of a
multi-episode run) waits for it like another process would (C7). Child processes that hold
the model (the TTS worker, llama-server) get the lock's file descriptor (``held_fds``, passed
with ``pass_fds``), so the lock stays held until the last of them exits, even when fukikae
itself is killed and a child outlives it.

Once the lock is taken, ``leftover_jobs`` scans ``/proc`` for fukikae TTS workers and
fukikae-started llama-servers still running on that card (by their argv array, whole arguments
only, and the card from
the process's ``CUDA_VISIBLE_DEVICES`` when it can be read), so a leftover from a run that died
is named in the error. It doesn't depend on NVML, which can't see processes in another pid
namespace; our own descendants are never leftovers.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)

LOCK_DIR_ENV = "FUKIKAE_LOCK_DIR"
LOCK_ROOT = Path("/tmp")  # noqa: S108 - a fixed, per-user subdirectory, checked below
WORKER_MARK = "fukikae.backends.tts.worker"
LLAMA_MARK = "fukikae-llama-key-"  # the per-run API key file llama_server passes
POLL_S = 2.0


class GpuBusy(RuntimeError):  # noqa: N818 - reads as the state, like VramTimeout
    """The card is held by another fukikae job, or a leftover one still runs on it."""


_mutex = threading.Lock()
_held: dict[tuple[int, int], list[int]] = {}  # (index, thread ident) -> [fd, depth]


def lock_dir() -> Path:
    env = os.environ.get(LOCK_DIR_ENV)
    if env:
        d = Path(env)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = LOCK_ROOT / f"fukikae-{os.getuid()}"
    with contextlib.suppress(FileExistsError):
        d.mkdir(mode=0o700)
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise GpuBusy(f"{d} is not a directory owned by this user; set {LOCK_DIR_ENV}")
    if st.st_mode & 0o077:
        os.chmod(d, 0o700)
    return d


def lock_path(index: int) -> Path:
    return lock_dir() / f"fukikae-gpu{index}.lock"


def held_fds() -> tuple[int, ...]:
    """File descriptors of the GPU locks this thread holds (for a child's ``pass_fds``)."""
    me = threading.get_ident()
    with _mutex:
        return tuple(fd for (_, tid), (fd, _) in _held.items() if tid == me)


# -- /proc -------------------------------------------------------------------------------------


def _argv(pid: int) -> list[str]:
    """The process's argv array (empty when it can't be read)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return []
    return [a.decode("utf-8", errors="replace") for a in raw.split(b"\0")[:-1]] if raw else []


def _is_job(argv: list[str]) -> bool:
    """A fukikae TTS worker (``-m`` followed by exactly ``WORKER_MARK``) or a fukikae-started
    llama-server (``argv[0]``'s basename is ``llama-server`` and its ``--api-key-file`` is one
    of our per-run key files). Whole arguments only: a ``bash -c "... <module> ..."``, a grep
    or an editor that mentions either one is not a job."""
    if any(a == "-m" and b == WORKER_MARK for a, b in zip(argv, argv[1:], strict=False)):
        return True
    if not argv or os.path.basename(argv[0]) != "llama-server":
        return False
    keys = [b for a, b in zip(argv, argv[1:], strict=False) if a == "--api-key-file"]
    keys += [a.partition("=")[2] for a in argv if a.startswith("--api-key-file=")]
    return any(os.path.basename(k).startswith(LLAMA_MARK) for k in keys)


def _proc_pids() -> list[int]:
    try:
        return sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    except OSError:
        return []


def _ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    # the command name may hold spaces and parentheses; the fields after the last ')' don't
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[1]) if len(fields) > 1 else None


def _read_environ(pid: int) -> list[bytes]:
    """The process's environment entries; OSError when it can't be read (another user's)."""
    with open(f"/proc/{pid}/environ", "rb") as f:
        return f.read().split(b"\0")


def _cuda_devices(pid: int) -> tuple[bool, str | None]:
    """(readable, CUDA_VISIBLE_DEVICES): ``(True, None)`` when the variable is unset (every
    card is visible), ``(False, None)`` when the environment can't be read."""
    try:
        env = _read_environ(pid)
    except OSError:
        return False, None
    for item in env:
        if item.startswith(b"CUDA_VISIBLE_DEVICES="):
            return True, item.partition(b"=")[2].decode("utf-8", errors="replace")
    return True, None


def _ours(pid: int, me: int) -> bool:
    """``pid`` is this process or one of its descendants."""
    seen = 0
    while pid and pid > 1 and seen < 4096:
        if pid == me:
            return True
        nxt = _ppid(pid)
        if nxt is None:
            return False
        pid, seen = nxt, seen + 1
    return False


_UNREADABLE: list[int] = []  # pids _on_card skipped during the current scan


def _on_card(pid: int, index: int) -> bool:
    """Whether ``pid`` can use card ``index``. A process whose environment can't be read
    (another user's) is skipped, with a debug log, rather than taken to be on every card."""
    readable, visible = _cuda_devices(pid)
    if not readable:
        log.debug("leftover scan: cannot read /proc/%d/environ; skipping that process", pid)
        _UNREADABLE.append(pid)
        return False
    if visible is None:
        return True
    if not visible.strip():
        return False  # hidden from every card: a CPU run
    try:
        return index in [int(p) for p in visible.split(",") if p.strip()]
    except ValueError:
        return True


def leftover_jobs(index: int) -> list[tuple[int, str]]:
    """(pid, command line) of fukikae TTS workers and fukikae llama-servers on card ``index``
    that are not this process's own."""
    me = os.getpid()
    out: list[tuple[int, str]] = []
    _UNREADABLE.clear()
    for pid in _proc_pids():
        argv = _argv(pid)
        if not _is_job(argv):
            continue
        cmd = " ".join(argv)
        if _ours(pid, me) or not _on_card(pid, index):
            continue
        out.append((pid, cmd))
    if _UNREADABLE:
        log.warning(
            "leftover scan: cannot read the environment of fukikae job(s) %s (another user's?); "
            "not counted as on GPU %d",
            ", ".join(map(str, _UNREADABLE)), index,
        )  # fmt: skip
        _UNREADABLE.clear()
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder(fd: int) -> tuple[int | None, str]:
    try:
        text = os.pread(fd, 4096, 0).decode("utf-8", errors="replace").strip()
    except OSError:
        return None, ""
    pid, _, what = text.partition(" ")
    return (int(pid) if pid.isdigit() else None), what


def _listed(found: list[tuple[int, str]]) -> str:
    return ", ".join(f"pid {pid} ({cmd[:200]})" for pid, cmd in found)


# -- the lock ----------------------------------------------------------------------------------


def _same_file(fd: int, path: Path) -> bool:
    """``fd`` is still the file at ``path`` (not unlinked or replaced since it was opened)."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    mine = os.fstat(fd)
    return (st.st_dev, st.st_ino) == (mine.st_dev, mine.st_ino)


def _open_lock(path: Path) -> int:
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)


def _acquire(index: int, what: str, timeout_s: float | None, poll_s: float) -> int:
    path = lock_path(index)
    fd = _open_lock(path)
    try:
        started = time.monotonic()
        logged = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _same_file(fd, path):
                    break
                # the file was replaced while we waited or as we took it: take the new one
                os.close(fd)
                fd = -1
                fd = _open_lock(path)
                continue
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
            pid, held_for = _holder(fd)
            if pid is not None and not _alive(pid):
                # the job that took it is gone and a child of it still holds the lock
                found = leftover_jobs(index)
                raise GpuBusy(
                    f"GPU {index} is locked by what is left of pid {pid} ({held_for}), which "
                    f"exited: {_listed(found) or 'a child of it still runs'}; stop it "
                    "(`kill <pid>`) and run again"
                )
            waited = time.monotonic() - started
            holder = f"pid {pid} ({held_for})" if pid is not None else "another fukikae job"
            if timeout_s is not None and waited >= timeout_s:
                raise GpuBusy(
                    f"GPU {index} is held by {holder}; gave up after {waited:.0f} s (lock {path})"
                )
            if not logged:
                log.warning("waiting for GPU %d: %s holds it", index, holder)
                logged = True
            pause = poll_s if timeout_s is None else max(0.0, min(poll_s, timeout_s - waited))
            time.sleep(pause)
        found = leftover_jobs(index)
        if found:
            raise GpuBusy(
                f"GPU {index} already runs {_listed(found)}, probably left over from an earlier "
                "run; stop it (`kill <pid>`) and run again"
            )
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} {what}\n".encode(), 0)
        return fd
    except BaseException:
        if fd >= 0:
            os.close(fd)  # closing the descriptor lets go of the lock too
        raise


@contextlib.contextmanager
def gpu_lock(
    index: int, what: str, *, timeout_s: float | None = None, poll_s: float = POLL_S
) -> Iterator[int]:
    """Hold card ``index`` (an NVML index) for the block; yields the lock's descriptor.
    ``timeout_s`` None takes the run's VRAM timeout (``runctx.vram_timeout_s``: None waits
    forever, 0 fails at once)."""
    if timeout_s is None:
        from fukikae import runctx

        timeout_s = runctx.vram_timeout_s
    key = (index, threading.get_ident())
    with _mutex:
        entry = _held.get(key)
        if entry is not None:
            entry[1] += 1
            fd = entry[0]
        else:
            fd = -1
    if fd < 0:
        # another thread of this process holding the card makes this flock wait, as another
        # process would: flock locks belong to the open file, and this is a new one
        fd = _acquire(index, what, timeout_s, poll_s)
        with _mutex:
            _held[key] = [fd, 1]
    try:
        yield fd
    finally:
        with _mutex:
            entry = _held[key]
            entry[1] -= 1
            done = entry[1] == 0
            if done:
                del _held[key]
        if done:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
            os.close(fd)


def gpu_job(gpu_device: int, device: str, what: str) -> contextlib.AbstractContextManager[object]:
    """``gpu_lock(gpu_device)`` when ``device`` is a cuda device, else nothing."""
    if device.startswith("cuda"):
        return gpu_lock(gpu_device, what)
    return contextlib.nullcontext()
