"""Progress during a run (#29): stage lines, a live bar or periodic log lines, status.json.

``fukikae run`` opens a ``Reporter`` for the workspace (``run``); ``job.run_stages`` wraps each
stage it runs in ``stage(episode, name)``, and the long stages report what they count with the
module functions ``start(total, unit)``, ``advance(n)``, ``set_done(done)`` and ``finish()``.
With no reporter open (tests, eval, a stage run directly) every function does nothing.

Each thread has its own slot, named by its role: ``head`` (the thread that runs the GPU stages)
or ``tail`` (D26's CPU tail thread, ``fukikae-tail``). On a TTY the slots share one
carriage-return line on stderr, redrawn at most every ``render_every`` seconds and cleared
before any log line; elsewhere each slot logs a plain line at most every ``line_every``
seconds (``tts S01E12: 120/325 cues, 3.4 s/cue, 6 min 48 s elapsed, ETA 11 min``).

``<workspace>/.fukikae/status.json`` is rewritten atomically at most every ``status_every``
seconds and at every stage boundary: the run, its pid, each episode's position, each slot's
stage and counts, the GPU lock waits and holds, and the last log line. ``fukikae status``
reads it (``read_status``, ``describe``) and tells a dead run from a live one by its pid.

A stage's ETA before its first count comes from earlier runs: ``stage-history.json`` beside
status.json keeps the last ``HISTORY_KEEP`` durations and counts of each stage, and without
any, the episodes' manifests give the durations of the stages they recorded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger(__name__)

STATUS_NAME = "status.json"
HISTORY_NAME = "stage-history.json"
HISTORY_KEEP = 20
TAIL_PREFIX = "fukikae-tail"  # job.run_episodes' tail thread
TICK = "progress_tick"  # a LogRecord attribute marking the periodic lines


def role() -> str:
    """``tail`` on D26's tail thread, else ``head``."""
    return "tail" if threading.current_thread().name.startswith(TAIL_PREFIX) else "head"


# -- formatting --------------------------------------------------------------------------------


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
    head = f"{s['stage']} {s['episode']}"
    if s.get("phase"):
        head += f" ({s['phase']})"
    parts: list[str] = []
    if s.get("total"):
        parts.append(f"{_count(s.get('done', 0))}/{_count(s['total'])} {s.get('unit', '')}".strip())
    if s.get("rate"):
        parts.append(fmt_rate(s["rate"], s.get("unit", "")))
    parts.append(f"{fmt_duration(s.get('elapsed', 0))} elapsed")
    eta = s.get("eta")
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


def _proc_start(pid: int) -> str | None:
    """The process's start time in clock ticks (field 22 of /proc/<pid>/stat), which tells a
    reused pid from the run's own; None where /proc can't say."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = text[text.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


class _ClearLine(logging.Filter):
    def __init__(self, reporter: Reporter) -> None:
        super().__init__()
        self.reporter = reporter

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002
        self.reporter.clear_line()
        return True


class _LastLine(logging.Handler):
    def __init__(self, reporter: Reporter) -> None:
        super().__init__(logging.INFO)
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, TICK, False):
            return
        with contextlib.suppress(Exception):
            self.reporter.last_log = record.getMessage().splitlines()[0][:500]


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
        self.status_path = self.state_dir / STATUS_NAME
        self.history_path = self.state_dir / HISTORY_NAME
        self.stream = stream if stream is not None else sys.stderr
        self.tty = self.stream.isatty() if tty is None else tty
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
        self._lock = threading.RLock()
        self._last_status = float("-inf")
        self._last_render = float("-inf")
        self._drawn = 0  # characters of the bar on screen (TTY)
        self._filters: list[tuple[logging.Handler, logging.Filter]] = []
        self._capture = _LastLine(self)

    # -- logging -----------------------------------------------------------------------------

    def install(self) -> None:
        logging.getLogger("fukikae").addHandler(self._capture)
        if not self.tty:
            return
        for name in ("fukikae", None):
            for h in logging.getLogger(name).handlers:
                if getattr(h, "stream", None) is self.stream:
                    f = _ClearLine(self)
                    h.addFilter(f)
                    self._filters.append((h, f))

    def uninstall(self) -> None:
        logging.getLogger("fukikae").removeHandler(self._capture)
        for h, f in self._filters:
            h.removeFilter(f)
        self._filters.clear()

    # -- slots -------------------------------------------------------------------------------

    def _hint(self, stage: str) -> _Hint:
        hint = _Hint()
        entries = self._history().get(stage, [])
        hint.seconds = _median([float(e["seconds"]) for e in entries if "seconds" in e])
        units: dict[str, list[float]] = {}
        for e in entries:
            if e.get("total") and e.get("unit") and e.get("counted"):
                units.setdefault(e["unit"], []).append(float(e["counted"]) / float(e["total"]))
        hint.per_unit = {u: statistics.median(v) for u, v in units.items()}
        if hint.seconds is None:
            hint.seconds = _median(_manifest_durations(self.state_dir, stage))
        return hint

    def stage_begin(self, episode: str, stage: str) -> None:
        now = self.clock()
        hint = self._hint(stage)
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
            if ok:
                self._record(slot, seconds, now)
            self._write_status(force=True)
            self._render(force=True)
        word = "done in" if ok else "failed after"
        log.info("%s %s: %s %s", slot.stage, slot.episode, word, fmt_duration(seconds))

    def start(self, total: float | None, unit: str, phase: str | None) -> None:
        with self._lock:
            slot = self.slots.get(role())
            if slot is None:
                return
            slot.total, slot.unit, slot.phase, slot.done = total, unit, phase, 0.0
            slot.counted_from = self.clock()
            self._changed()

    def advance(self, n: float) -> None:
        self._set(lambda s: s.done + n)

    def set_done(self, done: float) -> None:
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
            now = self.clock()
            if not self.tty and slot.total and now - slot.last_line >= self.line_every:
                slot.last_line = now
                line = slot_line(self._slot_dict(slot, now))
            self._changed()
        if line is not None:
            log.info("%s", line, extra={TICK: True})

    def _changed(self) -> None:
        self._write_status()
        self._render()

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
            if self.gpu.pop((index, role()), None) is not None:
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
                "version": 1,
                "run_id": self.run_id,
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
        now = self.clock()
        if not force and now - self._last_status < self.status_every:
            return
        self._last_status = now
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.status_path, json.dumps(self.snapshot(), indent=2))
        except OSError as e:  # a full disk never stops the run
            log.debug("status.json: %s", e)

    def clear_line(self) -> None:
        with self._lock:
            if self._drawn:
                self.stream.write("\r\x1b[K")
                self.stream.flush()
                self._drawn = 0

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
        self.stream.write("\r" + text + "\x1b[K")
        self.stream.flush()
        self._drawn = len(text)
        if not text:
            self._drawn = 0

    # -- history -----------------------------------------------------------------------------

    def _history(self) -> dict[str, list[dict[str, Any]]]:
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _record(self, slot: _Slot, seconds: float, now: float) -> None:
        history = self._history()
        entry: dict[str, Any] = {
            "episode": slot.episode,
            "finished": self.wall().isoformat(timespec="seconds"),
            "seconds": round(seconds, 2),
        }
        if slot.total:
            entry |= {
                "total": slot.total,
                "unit": slot.unit,
                "counted": round(now - slot.counted_from, 2),
            }
        entries = history.get(slot.stage, [])
        history[slot.stage] = [*entries, entry][-HISTORY_KEEP:]
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.history_path, json.dumps(history, indent=1))
        except OSError as e:
            log.debug("stage history: %s", e)

    def close(self, state: str) -> None:
        with self._lock:
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


@contextlib.contextmanager
def run(workspace: Path, episodes: list[str], **options: Any) -> Iterator[Reporter]:
    """Open the run's reporter for the block; the status file ends as finished, failed or
    interrupted."""
    global _active
    reporter = Reporter(workspace, episodes, **options)
    reporter.install()
    _active = reporter
    reporter._write_status(force=True)  # noqa: SLF001
    state = "failed"
    try:
        yield reporter
        state = "finished"
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    finally:
        _active = None
        reporter.close(state)
        reporter.uninstall()


@contextlib.contextmanager
def stage(episode: str, name: str) -> Iterator[None]:
    """One stage's run: a start line with the earlier runs' ETA, a finish line with its time."""
    r = _active
    if r is None:
        yield
        return
    r.stage_begin(episode, name)
    ok = False
    try:
        yield
        ok = True
    finally:
        r.stage_end(ok)


def start(total: float | None, unit: str = "", *, phase: str | None = None) -> None:
    """The current stage (or a phase of it) has ``total`` ``unit`` to do; None: a phase that
    can't count (its ETA comes from earlier runs)."""
    if _active is not None:
        _active.start(total, unit, phase)


def advance(n: float = 1) -> None:
    if _active is not None:
        _active.advance(n)


def set_done(done: float) -> None:
    if _active is not None:
        _active.set_done(done)


def finish() -> None:
    if _active is not None:
        _active.finish()


def episode(ep: str, position: str) -> None:
    if _active is not None:
        _active.episode(ep, position)


def gpu_waiting(index: int, pid: int | None, what: str, since: float | None) -> None:
    if _active is not None:
        _active.gpu_waiting(index, pid, what, since)


def gpu_held(index: int, what: str) -> None:
    if _active is not None:
        _active.gpu_held(index, what)


def gpu_released(index: int) -> None:
    if _active is not None:
        _active.gpu_released(index)


# -- fukikae status ----------------------------------------------------------------------------


def read_status(workspace: Path) -> dict[str, Any] | None:
    path = Path(workspace) / ".fukikae" / STATUS_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_live(status: dict[str, Any]) -> bool:
    """The run is still going: recorded as running, and its pid is alive and is the same
    process (its start time matches, where /proc can tell)."""
    pid = status.get("pid")
    if status.get("state") != "running" or not isinstance(pid, int) or not _pid_alive(pid):
        return False
    recorded = status.get("pid_start")
    return recorded is None or _proc_start(pid) in (None, recorded)


def _ago(iso: str | None, now: datetime) -> float:
    if not iso:
        return 0.0
    try:
        return max(0.0, (now - datetime.fromisoformat(iso)).total_seconds())
    except ValueError:
        return 0.0


def describe(status: dict[str, Any] | None, now: datetime | None = None) -> str:
    """What ``fukikae status`` prints for ``status`` (read_status's result)."""
    now = now or datetime.now(UTC)
    if status is None:
        return "no run in progress (no run recorded in this workspace)"
    local = _local
    if not is_live(status):
        if status.get("state") == "running":
            when = f"died at {local(status.get('updated'))}"
        else:
            ended = status.get("ended") or status.get("updated")
            when = f"{status.get('state', 'ended')} at {local(ended)}"
        eps = ", ".join(f"{e['id']} {e['position']}" for e in status.get("episodes", []))
        return f"no run in progress (last run {when}; episodes: {eps or 'none'})"
    since = _ago(status.get("updated"), now)
    lines = [
        f"run {status.get('run_id')}, pid {status.get('pid')}, started "
        f"{local(status.get('started'))} ({fmt_duration(_ago(status.get('started'), now))} ago)",
        "episodes: " + ", ".join(f"{e['id']} {e['position']}" for e in status.get("episodes", [])),
    ]
    threads = status.get("threads", {})
    for name, s in threads.items():
        s = dict(s)
        s["elapsed"] = _ago(s.get("stage_started"), now)
        if s.get("eta") is not None:
            s["eta"] = max(0.0, s["eta"] - since)
        lines.append(f"{name}: {slot_line(s)}")
    if not threads:
        lines.append("between stages")
    for g in status.get("gpu", []):
        holder = f"pid {g.get('holder_pid')} ({g.get('holder')})"
        if g.get("state") == "waiting":
            held = f" for {fmt_duration(_ago(g['held_since'], now))}" if g.get("held_since") else ""
            waited = fmt_duration(_ago(g.get("waiting_since"), now))
            lines.append(
                f"GPU {g['gpu']} ({g['thread']}): waiting {waited}, held by {holder}{held}"
            )
        else:
            lines.append(f"GPU {g['gpu']} ({g['thread']}): held by this run ({g.get('holder')})")
    if status.get("last_log"):
        lines.append(f"last log: {status['last_log']}")
    lines.append(f"updated {fmt_duration(since)} ago")
    return "\n".join(lines)


def _local(iso: str | None) -> str:
    if not iso:
        return "an unknown time"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso
