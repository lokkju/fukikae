"""Progress during a run (#29): stage lines, a live bar or periodic log lines, a status file.

``fukikae run`` opens a ``Reporter`` for the workspace (``run``); ``job.run_stages`` wraps each
stage it runs in ``stage(episode, name)``, and the long stages report what they count with the
module functions ``start(total, unit)``, ``advance(n)``, ``set_done(done)`` and ``finish()``.
With no reporter open (tests, eval, a stage run directly) every function does nothing.

A reporter failure never fails a run or a log call: every module function, ``stage``'s begin
and end, the ticker and the stream wrapper catch any exception, log it once at debug, and
degrade (a TTY reporter stops drawing; a second failure, or one without a TTY, turns the
reporter off for the rest of the run).

Each thread has its own slot, named by its role: ``head`` (the thread that runs the GPU stages)
or ``tail`` (D26's CPU tail thread, ``fukikae-tail``). On a TTY (not ``TERM=dumb``) the slots
share one carriage-return line on stderr, redrawn at most every ``render_every`` seconds; the
log handlers writing to the same stream get a wrapper that clears the line and writes under the
reporter's lock, so a log line never lands on the bar. Elsewhere each slot logs a plain line at
most every ``line_every`` seconds (``tts S01E12: 120/325 cues, 3.4 s/cue, 6 min 48 s elapsed,
ETA 11 min``; a phase that can't count logs its elapsed time and ETA). A daemon ticker redraws,
logs and rewrites the status file on those intervals while nothing is counted.

``<workspace>/.fukikae/runs/<host>-<pid>.json`` is rewritten atomically at most every
``status_every`` seconds and at every stage boundary: the run, its host and pid, each
episode's position, each slot's stage and counts, the GPU lock waits and holds, and the last
log line. ``fukikae status`` reads them all (``read_runs``, ``describe_runs``): a run on this
host is live while its pid is (and, where /proc exists, is the same process by its start time;
elsewhere a reused pid reads as live), a run on another host while its file was updated in the
last ``STALE_S`` seconds.

A stage's ETA before its first count comes from earlier runs: ``stage-history.json`` beside
the run files keeps the last ``HISTORY_KEEP`` durations and counts of each stage (written
under a file lock, so two runs don't drop each other's entries), and without any, the
episodes' manifests give the durations of the stages they recorded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import socket
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

log = logging.getLogger(__name__)

RUNS_DIR = "runs"
HISTORY_NAME = "stage-history.json"
HISTORY_KEEP = 20
STALE_S = 120.0  # a run on another host whose file is older than this has stopped
KEEP_DEAD_S = 86400.0  # `fukikae status` deletes dead runs' files older than this
TAIL_PREFIX = "fukikae-tail"  # job.run_episodes' tail thread
TICK = "progress_tick"  # a LogRecord attribute marking the periodic lines


def role() -> str:
    """``tail`` on D26's tail thread, else ``head``."""
    return "tail" if threading.current_thread().name.startswith(TAIL_PREFIX) else "head"


def _host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


# -- formatting --------------------------------------------------------------------------------


def _num(x: Any) -> float | None:
    """``x`` as a finite float, or None (bools, strings and nulls from a corrupt file)."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def fmt_duration(seconds: float) -> str:
    s = max(0, round(seconds))
    if s < 60:
        return f"{s} s"
    if s < 600:
        return f"{s // 60} min {s % 60} s"
    if s < 3600:
        return f"{round(s / 60)} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def _singular(unit: str) -> str:
    return unit[:-1] if unit.endswith("s") and " " not in unit else unit


def _count(x: float) -> str:
    return str(round(x))


def fmt_rate(seconds_per_unit: float, unit: str) -> str:
    if seconds_per_unit >= 1:
        return f"{seconds_per_unit:.1f} s/{_singular(unit)}"
    return f"{1 / seconds_per_unit:.1f} {unit}/s"


def slot_line(s: dict[str, Any]) -> str:
    """``<stage> <episode>[ (<phase>)]: <done>/<total> <unit>, <rate>, <elapsed> elapsed,
    ETA <eta>`` from a slot's status dict; parts that aren't known are left out."""
    head = f"{s.get('stage', '?')} {s.get('episode', '?')}"
    if s.get("phase"):
        head += f" ({s['phase']})"
    unit = str(s.get("unit") or "")
    parts: list[str] = []
    total, rate = _num(s.get("total")), _num(s.get("rate"))
    if total:
        parts.append(f"{_count(_num(s.get('done')) or 0)}/{_count(total)} {unit}".strip())
    if rate:
        parts.append(fmt_rate(rate, unit))
    parts.append(f"{fmt_duration(_num(s.get('elapsed')) or 0)} elapsed")
    eta = _num(s.get("eta"))
    if eta is not None:
        parts.append(
            f"ETA {fmt_duration(eta)}"
            + (" (earlier runs)" if s.get("eta_from") == "history" else "")
        )
    return f"{head}: {', '.join(parts)}"


# -- history -----------------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass
class _Hint:
    seconds: float | None = None  # the stage's usual duration
    per_unit: dict[str, float] = field(default_factory=dict)  # unit -> usual seconds per unit


def _manifest_durations(state_dir: Path, stage: str) -> list[float]:
    out: list[float] = []
    for path in sorted((state_dir / "episodes").glob("*/manifest.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8")).get("stages", {}).get(stage)
            if not rec or rec.get("status") != "done" or not rec.get("finished"):
                continue
            started = datetime.fromisoformat(rec["started"])
            finished = datetime.fromisoformat(rec["finished"])
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
        out.append((finished - started).total_seconds())
    return out


def _hint_from(entries: Any) -> _Hint:
    """The usual duration and seconds per unit from a stage's history entries, skipping any
    entry that isn't a dict or whose numbers don't parse; a non-list is no history."""
    hint = _Hint()
    if not isinstance(entries, list):
        return hint
    seconds: list[float] = []
    units: dict[str, list[float]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        s = _num(e.get("seconds"))
        if s is not None and s >= 0:
            seconds.append(s)
        total, counted, unit = _num(e.get("total")), _num(e.get("counted")), e.get("unit")
        if total and total > 0 and counted is not None and counted >= 0 and isinstance(unit, str):
            units.setdefault(unit, []).append(counted / total)
    hint.seconds = _median(seconds)
    hint.per_unit = {u: statistics.median(v) for u, v in units.items() if v}
    return hint


def _jsonable(o: Any) -> Any:
    """json.dumps' fallback: numpy scalars as their Python value, anything else as text."""
    item = getattr(o, "item", None)
    if callable(item):
        with contextlib.suppress(Exception):
            return item()
    return str(o)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """An exclusive flock on ``path`` (created if missing) for the block; no lock where fcntl
    doesn't exist."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not Linux
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing drops the lock


# -- the reporter ------------------------------------------------------------------------------


@dataclass
class _Slot:
    episode: str
    stage: str
    started: float  # monotonic
    started_wall: datetime
    hint: _Hint
    phase: str | None = None
    total: float | None = None
    done: float = 0.0
    unit: str = ""
    counted_from: float = 0.0  # monotonic time of the last start()
    last_line: float = 0.0  # monotonic time of the last periodic line
    record: bool = True  # False: the stage's time doesn't go into the history (mostly cached)


def _proc_start(pid: int) -> str | None:
    """The process's start time in clock ticks (field 22 of /proc/<pid>/stat), which tells a
    reused pid from the run's own; None where /proc can't say (not Linux)."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = text[text.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


class _LockedStream:
    """A log handler's stream while a TTY reporter draws on it: each write clears the bar and
    writes under the reporter's lock, so no redraw lands between the two. A failure to clear
    degrades the reporter; the write itself goes ahead (its errors are the handler's)."""

    def __init__(self, reporter: Reporter, stream: TextIO) -> None:
        self._reporter = reporter
        self._stream = stream

    def write(self, text: str) -> int:
        r = self._reporter
        error: Exception | None = None
        with r._lock:  # noqa: SLF001
            try:
                r.clear_line()
            except Exception as e:  # noqa: BLE001
                error = e
            written = self._stream.write(text)
        if error is not None:
            r.fail("clear", error)  # outside the lock: it may log
        return written

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _LastLine(logging.Handler):
    def __init__(self, reporter: Reporter) -> None:
        super().__init__(logging.INFO)
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, TICK, False):
            return
        with contextlib.suppress(Exception):
            self.reporter.last_log = record.getMessage().splitlines()[0][:500]


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty()) and os.environ.get("TERM") != "dumb"
    except Exception:  # noqa: BLE001
        return False


class Reporter:
    def __init__(
        self,
        workspace: Path,
        episodes: list[str],
        *,
        stream: TextIO | None = None,
        tty: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now(UTC),
        line_every: float = 30.0,
        status_every: float = 2.0,
        render_every: float = 0.2,
    ) -> None:
        self.state_dir = Path(workspace) / ".fukikae"
        self.host = _host()
        safe_host = re.sub(r"[^A-Za-z0-9._-]", "_", self.host) or "host"
        self.status_path = self.state_dir / RUNS_DIR / f"{safe_host}-{os.getpid()}.json"
        self.history_path = self.state_dir / HISTORY_NAME
        self.stream = stream if stream is not None else sys.stderr
        self.tty = _is_tty(self.stream) if tty is None else tty
        self.clock, self.wall = clock, wall
        self.line_every, self.status_every, self.render_every = (
            line_every, status_every, render_every,
        )  # fmt: skip
        self.run_id = uuid.uuid4().hex[:12]
        self.started_wall = wall()
        self.episodes = list(episodes)
        self.positions: dict[str, str] = dict.fromkeys(self.episodes, "pending")
        self.slots: dict[str, _Slot] = {}
        self.gpu: dict[tuple[int, str], dict[str, Any]] = {}
        self.last_log: str | None = None
        self.state = "running"
        self.ended: datetime | None = None
        self.disabled = False
        self._failures = 0
        self._debugged: set[str] = set()
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._seq = 0  # status snapshots taken
        self._written = 0  # the newest snapshot on disk
        self._last_status = float("-inf")
        self._last_render = float("-inf")
        self._drawn = 0  # characters of the bar on screen (TTY)
        self._wrapped: list[tuple[logging.StreamHandler[Any], TextIO]] = []
        self._capture = _LastLine(self)
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    # -- failures ----------------------------------------------------------------------------

    def _debug_once(self, what: str, e: BaseException) -> None:
        if what in self._debugged:
            return
        self._debugged.add(what)
        with contextlib.suppress(Exception):
            log.debug("progress: %s failed (%s: %s); the run goes on", what, type(e).__name__, e)

    def fail(self, what: str, e: BaseException) -> None:
        """A reporter call raised: stop drawing (TTY), or on a second failure or without a TTY,
        turn the reporter off. Never raises."""
        self._failures += 1
        if self.tty and self._failures == 1:
            self.tty = False
            self._drawn = 0
        else:
            self.disabled = True
            self._stop.set()
        self._debug_once(f"reporter ({what})", e)

    # -- logging -----------------------------------------------------------------------------

    def install(self) -> None:
        logging.getLogger("fukikae").addHandler(self._capture)
        if not self.tty:
            return
        for name in ("fukikae", None):
            for h in logging.getLogger(name).handlers:
                if isinstance(h, logging.StreamHandler) and h.stream is self.stream:
                    h.setStream(cast(TextIO, _LockedStream(self, self.stream)))
                    self._wrapped.append((h, self.stream))

    def uninstall(self) -> None:
        logging.getLogger("fukikae").removeHandler(self._capture)
        for h, stream in self._wrapped:
            if isinstance(h.stream, _LockedStream):
                with contextlib.suppress(Exception):
                    h.setStream(stream)
        self._wrapped.clear()

    # -- the ticker --------------------------------------------------------------------------

    def tick_every(self) -> float:
        every = [self.status_every, self.line_every]
        if self.tty:
            every.append(self.render_every)
        return max(0.1, min(e for e in every if e > 0) if any(e > 0 for e in every) else 1.0)

    def start_ticker(self, every: float | None = None) -> None:
        """Redraw, log and rewrite the status file on their intervals from a daemon thread,
        so a phase that counts nothing still moves; stops at close()."""
        period = self.tick_every() if every is None else every
        if period <= 0:
            return

        def loop() -> None:
            while not self._stop.wait(period):
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001
                    self.fail("tick", e)

        self._ticker = threading.Thread(target=loop, name="fukikae-progress", daemon=True)
        self._ticker.start()

    def tick(self) -> None:
        lines: list[str] = []
        with self._lock:
            if self.disabled:
                return
            now = self.clock()
            if not self.tty:
                for slot in self.slots.values():
                    if now - slot.last_line >= self.line_every:
                        slot.last_line = now
                        lines.append(slot_line(self._slot_dict(slot, now)))
            self._render()
        self._write_status()
        for line in lines:
            log.info("%s", line, extra={TICK: True})

    # -- slots -------------------------------------------------------------------------------

    def _hint(self, stage: str) -> _Hint:
        hint = _hint_from(self._history().get(stage))
        if hint.seconds is None:
            hint.seconds = _median(_manifest_durations(self.state_dir, stage))
        return hint

    def stage_begin(self, episode: str, stage: str) -> None:
        now = self.clock()
        try:
            hint = self._hint(stage)
        except Exception as e:  # noqa: BLE001
            self._debug_once("stage history", e)
            hint = _Hint()
        with self._lock:
            self.slots[role()] = _Slot(
                episode, stage, now, self.wall(), hint, counted_from=now, last_line=now
            )
            self.positions[episode] = role()
        self._write_status(force=True)
        eta = f" (about {fmt_duration(hint.seconds)} on earlier runs)" if hint.seconds else ""
        log.info("%s %s: started%s", stage, episode, eta)

    def stage_end(self, ok: bool) -> None:
        now = self.clock()
        with self._lock:
            slot = self.slots.pop(role(), None)
            if slot is None:
                return
            seconds = now - slot.started
            counted = now - slot.counted_from
            self._render(force=True)
        if ok and slot.record:
            self._record(slot, seconds, counted)
        self._write_status(force=True)
        word = "done in" if ok else "failed after"
        log.info("%s %s: %s %s", slot.stage, slot.episode, word, fmt_duration(seconds))

    def start(self, total: float | None, unit: str, phase: str | None, cached: int = 0) -> None:
        total = None if total is None else float(total)
        with self._lock:
            slot = self.slots.get(role())
            if slot is None:
                return
            slot.total, slot.unit, slot.phase, slot.done = total, unit, phase, 0.0
            slot.counted_from = self.clock()
            self._render()
            name = (slot.stage, slot.episode)
        self._write_status()
        if cached:
            log.info("%s %s: %d %s cached", *name, cached, unit)

    def skip_history(self) -> None:
        with self._lock:
            slot = self.slots.get(role())
            if slot is not None:
                slot.record = False

    def advance(self, n: float) -> None:
        n = float(n)
        self._set(lambda s: s.done + n)

    def set_done(self, done: float) -> None:
        done = float(done)
        self._set(lambda _: done)

    def finish(self) -> None:
        self._set(lambda s: s.total if s.total is not None else s.done)

    def _set(self, value: Callable[[_Slot], float]) -> None:
        line = None
        with self._lock:
            slot = self.slots.get(role())
            if slot is None:
                return
            slot.done = value(slot)
            if slot.total is not None and slot.done > slot.total:
                slot.total = slot.done  # a count that ran past its estimate
            now = self.clock()
            if not self.tty and slot.total and now - slot.last_line >= self.line_every:
                slot.last_line = now
                line = slot_line(self._slot_dict(slot, now))
            self._render()
        self._write_status()
        if line is not None:
            log.info("%s", line, extra={TICK: True})

    def episode(self, episode: str, position: str) -> None:
        with self._lock:
            self.positions[episode] = position
        self._write_status(force=True)

    # -- GPU lock ----------------------------------------------------------------------------

    def gpu_waiting(self, index: int, pid: int | None, what: str, since: float | None) -> None:
        with self._lock:
            self.gpu[(index, role())] = {
                "gpu": index,
                "thread": role(),
                "state": "waiting",
                "holder_pid": pid,
                "holder": what,
                "held_since": _iso_epoch(since),
                "waiting_since": self.gpu.get((index, role()), {}).get("waiting_since")
                or self.wall().isoformat(timespec="seconds"),
            }
        self._write_status()

    def gpu_held(self, index: int, what: str) -> None:
        with self._lock:
            self.gpu[(index, role())] = {
                "gpu": index,
                "thread": role(),
                "state": "held",
                "holder_pid": os.getpid(),
                "holder": what,
                "held_since": self.wall().isoformat(timespec="seconds"),
            }
        self._write_status(force=True)

    def gpu_released(self, index: int) -> None:
        with self._lock:
            released = self.gpu.pop((index, role()), None) is not None
        if released:
            self._write_status(force=True)

    # -- output ------------------------------------------------------------------------------

    def _slot_dict(self, slot: _Slot, now: float) -> dict[str, Any]:
        elapsed = now - slot.started
        counting = now - slot.counted_from
        rate = eta = None
        eta_from = None
        if slot.total and slot.done > 0:
            rate = counting / slot.done
            eta, eta_from = max(0.0, (slot.total - slot.done) * rate), "rate"
        elif slot.total and slot.unit in slot.hint.per_unit:
            eta = max(0.0, slot.total * slot.hint.per_unit[slot.unit] - counting)
            eta_from = "history"
        elif slot.hint.seconds is not None:
            eta, eta_from = max(0.0, slot.hint.seconds - elapsed), "history"
        return {
            "episode": slot.episode,
            "stage": slot.stage,
            "phase": slot.phase,
            "stage_started": slot.started_wall.isoformat(timespec="seconds"),
            "done": slot.done,
            "total": slot.total,
            "unit": slot.unit,
            "rate": rate,
            "elapsed": round(elapsed, 1),
            "eta": None if eta is None else round(eta, 1),
            "eta_from": eta_from,
        }

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            return {
                "version": 2,
                "run_id": self.run_id,
                "host": self.host,
                "pid": os.getpid(),
                "pid_start": _proc_start(os.getpid()),
                "started": self.started_wall.isoformat(timespec="seconds"),
                "updated": self.wall().isoformat(timespec="seconds"),
                "state": self.state,
                "ended": self.ended.isoformat(timespec="seconds") if self.ended else None,
                "workspace": str(self.state_dir.parent),
                "episodes": [{"id": e, "position": self.positions[e]} for e in self.positions],
                "threads": {r: self._slot_dict(s, now) for r, s in sorted(self.slots.items())},
                "gpu": [self.gpu[k] for k in sorted(self.gpu)],
                "last_log": self.last_log,
            }

    def _write_status(self, force: bool = False) -> None:
        """Take a snapshot under the lock (at most every ``status_every`` seconds unless
        ``force``), then write it after releasing it; an older snapshot never replaces a newer
        one. Never raises."""
        error: BaseException | None = None
        text = None
        with self._lock:
            now = self.clock()
            if not force and now - self._last_status < self.status_every:
                return
            self._last_status = now
            self._seq += 1
            seq = self._seq
            try:
                text = json.dumps(self.snapshot(), indent=2, default=_jsonable)
            except Exception as e:  # noqa: BLE001
                error = e
        if text is None:
            self._debug_once("status snapshot", error or RuntimeError("no snapshot"))
            return
        with self._write_lock:
            if seq <= self._written:
                return
            try:
                self.status_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(self.status_path, text)
                self._written = seq
            except Exception as e:  # noqa: BLE001 - a full disk never stops the run
                self._debug_once("status file", e)

    def clear_line(self) -> None:
        with self._lock:
            if self._drawn:
                self._drawn = 0
                self.stream.write("\r\x1b[K")
                self.stream.flush()

    def _render(self, force: bool = False) -> None:
        if not self.tty:
            return
        now = self.clock()
        if not force and now - self._last_render < self.render_every:
            return
        self._last_render = now
        text = " | ".join(
            (f"[{r}] " if len(self.slots) > 1 else "") + slot_line(self._slot_dict(s, now))
            for r, s in sorted(self.slots.items())
        )
        width = _width(self.stream)
        text = text[: max(10, width - 1)]
        self._drawn = 0
        self.stream.write("\r" + text + "\x1b[K")
        self.stream.flush()
        self._drawn = len(text)

    # -- history -----------------------------------------------------------------------------

    def _history(self) -> dict[str, Any]:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _record(self, slot: _Slot, seconds: float, counted: float) -> None:
        """Append the stage's time (and count) to the history, read-modify-write under a file
        lock. Never raises."""
        entry: dict[str, Any] = {
            "episode": slot.episode,
            "finished": self.wall().isoformat(timespec="seconds"),
            "seconds": round(seconds, 2),
        }
        if slot.total and slot.done > 0:
            entry |= {"total": slot.done, "unit": slot.unit, "counted": round(counted, 2)}
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with _file_lock(self.state_dir / f".{HISTORY_NAME}.lock"):
                history = self._history()
                entries = history.get(slot.stage)
                entries = [e for e in entries if isinstance(e, dict)] if isinstance(
                    entries, list
                ) else []  # fmt: skip
                history[slot.stage] = [*entries, entry][-HISTORY_KEEP:]
                _atomic_write(self.history_path, json.dumps(history, indent=1, default=_jsonable))
        except Exception as e:  # noqa: BLE001
            self._debug_once("stage history", e)

    def close(self, state: str) -> None:
        self._stop.set()
        if self._ticker is not None and self._ticker is not threading.current_thread():
            self._ticker.join(5)
        with self._lock:
            with contextlib.suppress(Exception):
                self.clear_line()
            if state == "finished" and any(p == "failed" for p in self.positions.values()):
                state = "failed"
            self.state = state
            self.ended = self.wall()
            self.slots.clear()
            self.gpu.clear()
        self._write_status(force=True)


def _iso_epoch(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


def _width(stream: TextIO) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return 120


def _atomic_write(path: Path, text: str) -> None:
    from fukikae.artifacts import atomic_path

    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


# -- the module API ----------------------------------------------------------------------------

_active: Reporter | None = None


def active() -> Reporter | None:
    return _active


def _call(method: str, *args: Any) -> None:
    """``method`` of the open reporter, if any; a failure degrades the reporter, never the
    caller."""
    r = _active
    if r is None or r.disabled:
        return
    try:
        getattr(r, method)(*args)
    except Exception as e:  # noqa: BLE001
        r.fail(method, e)


@contextlib.contextmanager
def run(workspace: Path, episodes: list[str], **options: Any) -> Iterator[Reporter | None]:
    """Open the run's reporter for the block (None if it can't be opened); the run file ends
    as finished, failed or interrupted."""
    global _active
    tick = options.pop("tick", None)
    reporter: Reporter | None = None
    try:
        reporter = Reporter(workspace, episodes, **options)
        reporter.install()
        reporter._write_status(force=True)  # noqa: SLF001
        reporter.start_ticker(tick)
    except Exception as e:  # noqa: BLE001
        log.debug("progress: no reporter for this run (%s: %s)", type(e).__name__, e)
        if reporter is not None:
            with contextlib.suppress(Exception):
                reporter.close("failed")
                reporter.uninstall()
        reporter = None
    _active = reporter
    state = "failed"
    try:
        yield reporter
        state = "finished"
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    finally:
        _active = None
        if reporter is not None:
            try:
                reporter.close(state)
            except Exception as e:  # noqa: BLE001
                reporter._debug_once("close", e)  # noqa: SLF001
            try:
                reporter.uninstall()
            except Exception as e:  # noqa: BLE001
                reporter._debug_once("uninstall", e)  # noqa: SLF001


@contextlib.contextmanager
def stage(episode: str, name: str) -> Iterator[None]:
    """One stage's run: a start line with the earlier runs' ETA, a finish line with its time."""
    r = _active
    if r is None or r.disabled:
        yield
        return
    _call("stage_begin", episode, name)
    ok = False
    try:
        yield
        ok = True
    finally:
        _call("stage_end", ok)


def start(
    total: float | None, unit: str = "", *, phase: str | None = None, cached: int = 0
) -> None:
    """The current stage (or a phase of it) has ``total`` ``unit`` to do; None: a phase that
    can't count (its ETA comes from earlier runs). ``cached``: how many more were found done
    already, logged once and never counted."""
    _call("start", total, unit, phase, cached)


def skip_history() -> None:
    """Keep the current stage's time out of the ETA history (it was mostly cached)."""
    _call("skip_history")


def advance(n: float = 1) -> None:
    _call("advance", n)


def set_done(done: float) -> None:
    _call("set_done", done)


def finish() -> None:
    _call("finish")


def episode(ep: str, position: str) -> None:
    _call("episode", ep, position)


def gpu_waiting(index: int, pid: int | None, what: str, since: float | None) -> None:
    _call("gpu_waiting", index, pid, what, since)


def gpu_held(index: int, what: str) -> None:
    _call("gpu_held", index, what)


def gpu_released(index: int) -> None:
    _call("gpu_released", index)


# -- fukikae status ----------------------------------------------------------------------------


def _parse(iso: Any) -> datetime | None:
    if not isinstance(iso, str):
        return None
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _ago(iso: Any, now: datetime) -> float:
    t = _parse(iso)
    return 0.0 if t is None else max(0.0, (now - t).total_seconds())


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _when(status: dict[str, Any]) -> datetime:
    """When the run last did anything: its end, else its last update."""
    return (
        _parse(status.get("ended"))
        or _parse(status.get("updated"))
        or datetime.min.replace(tzinfo=UTC)
    )


def read_runs(
    workspace: Path, *, clean: bool = False, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Every run recorded in the workspace, newest first. ``clean``: delete the files of runs
    that are no longer live and last did anything over ``KEEP_DEAD_S`` ago, except the newest
    of them (and unreadable files that old)."""
    now = now or datetime.now(UTC)
    runs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((Path(workspace) / ".fukikae" / RUNS_DIR).glob("*.json")):
        data = _read(path)
        if data is None:
            if clean:
                with contextlib.suppress(OSError):
                    if time.time() - path.stat().st_mtime > KEEP_DEAD_S:
                        path.unlink()
            continue
        runs.append((path, data))
    runs.sort(key=lambda pr: _when(pr[1]), reverse=True)
    if clean:
        kept_dead = False
        for path, data in list(runs):
            if is_live(data, now):
                continue
            if kept_dead and (now - _when(data)).total_seconds() > KEEP_DEAD_S:
                with contextlib.suppress(OSError):
                    path.unlink()
                runs.remove((path, data))
            kept_dead = True
    return [d for _, d in runs]


def read_status(workspace: Path) -> dict[str, Any] | None:
    """The run to show for the workspace: the newest live one, else the newest."""
    runs = read_runs(workspace)
    live = [r for r in runs if is_live(r)]
    return (live or runs or [None])[0]


def shown_runs(runs: list[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    """What ``fukikae status`` shows of ``runs`` (newest first): every live run, then the most
    recent one that isn't."""
    live = [r for r in runs if is_live(r, now)]
    ended = [r for r in runs if not is_live(r, now)]
    return live + ended[:1]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remote(status: dict[str, Any]) -> bool:
    host = status.get("host")
    return isinstance(host, str) and host != _host()


def is_live(status: dict[str, Any], now: datetime | None = None) -> bool:
    """The run is still going: recorded as running, and on this host its pid is alive and is
    the same process (its start time matches, where /proc can tell); on another host, where
    the pid can't be checked, its file was updated within ``STALE_S``."""
    if status.get("state") != "running":
        return False
    if _remote(status):
        return _ago(status.get("updated"), now or datetime.now(UTC)) < STALE_S
    pid = status.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not _pid_alive(pid):
        return False
    recorded = status.get("pid_start")
    return recorded is None or _proc_start(pid) in (None, recorded)


def _episodes(status: dict[str, Any]) -> str:
    eps = status.get("episodes")
    if not isinstance(eps, list):
        return ""
    return ", ".join(
        f"{e.get('id', '?')} {e.get('position', '?')}" for e in eps if isinstance(e, dict)
    )


def _ended(status: dict[str, Any]) -> str:
    """``<how> at <time>; episodes: ...`` for a run that isn't live."""
    where = f" on {status['host']}" if _remote(status) else ""
    if status.get("state") == "running":
        how = "stopped updating" if _remote(status) else "died"
        when = f"{how} at {_local(status.get('updated'))}{where}"
    else:
        ended = status.get("ended") or status.get("updated")
        when = f"{status.get('state') or 'ended'} at {_local(ended)}{where}"
    return f"{when}; episodes: {_episodes(status) or 'none'}"


def describe(status: dict[str, Any] | None, now: datetime | None = None) -> str:
    """What ``fukikae status`` prints for one run (a run file's contents, or None)."""
    now = now or datetime.now(UTC)
    if status is None:
        return "no run in progress (no run recorded in this workspace)"
    if not is_live(status, now):
        return f"no run in progress (last run {_ended(status)})"
    local = _local
    remote = _remote(status)
    since = _ago(status.get("updated"), now)
    pid = "" if remote else f", pid {status.get('pid')}"
    lines = [
        f"run {status.get('run_id')}{pid}, started "
        f"{local(status.get('started'))} ({fmt_duration(_ago(status.get('started'), now))} ago)",
        f"episodes: {_episodes(status)}",
    ]
    threads = status.get("threads")
    threads = threads if isinstance(threads, dict) else {}
    for name, s in threads.items():
        if not isinstance(s, dict):
            continue
        s = dict(s)
        s["elapsed"] = _ago(s.get("stage_started"), now)
        eta = _num(s.get("eta"))
        s["eta"] = None if eta is None else max(0.0, eta - since)
        lines.append(f"{name}: {slot_line(s)}")
    if not threads:
        lines.append("between stages")
    gpus = status.get("gpu")
    for g in gpus if isinstance(gpus, list) else []:
        if not isinstance(g, dict):
            continue
        gpu = f"GPU {g.get('gpu', '?')} ({g.get('thread', '?')})"
        holder = f"pid {g.get('holder_pid')} ({g.get('holder')})"
        if g.get("state") == "waiting":
            since_held = g.get("held_since")
            held = f" for {fmt_duration(_ago(since_held, now))}" if since_held else ""
            waited = fmt_duration(_ago(g.get("waiting_since"), now))
            lines.append(f"{gpu}: waiting {waited}, held by {holder}{held}")
        else:
            lines.append(f"{gpu}: held by this run ({g.get('holder')})")
    if status.get("last_log"):
        lines.append(f"last log: {status['last_log']}")
    if remote:
        lines.append(
            f"running on {status['host']} (pid not checkable here), "
            f"updated {fmt_duration(since)} ago"
        )
    else:
        lines.append(f"updated {fmt_duration(since)} ago")
    return "\n".join(lines)


def describe_runs(runs: list[dict[str, Any]], now: datetime | None = None) -> str:
    """What ``fukikae status`` prints for the workspace's runs (read_runs' result): each live
    run, then the most recent run that has ended."""
    now = now or datetime.now(UTC)
    shown = shown_runs(runs, now)
    live = [r for r in shown if is_live(r, now)]
    ended = [r for r in shown if not is_live(r, now)]
    if not live:
        return describe(ended[0] if ended else None, now)
    parts = [describe(r, now) for r in live]
    if len(live) > 1:
        parts.insert(0, f"{len(live)} runs in progress")
    if ended:
        parts.append(f"last finished run: {_ended(ended[0])}")
    return "\n\n".join(parts)


def _local(iso: Any) -> str:
    t = _parse(iso)
    if t is None:
        return "an unknown time" if not iso else str(iso)
    return t.astimezone().strftime("%Y-%m-%d %H:%M:%S")
