"""The run reporter (#29): periodic lines, the TTY bar, status.json, history ETAs, status."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fukikae import progress
from fukikae.progress import Reporter


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.wall0 = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return self.wall0 + timedelta(seconds=self.t - 1000.0)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _reporter(tmp_path: Path, clock: FakeClock, **kw) -> Reporter:
    kw.setdefault("stream", io.StringIO())
    kw.setdefault("tty", False)
    return Reporter(tmp_path, ["S01E12"], clock=clock, wall=clock.wall, **kw)


def _run(tmp_path: Path, clock: FakeClock | None = None, **kw):
    if clock is not None:
        kw |= {"clock": clock, "wall": clock.wall}
    return progress.run(tmp_path, ["S01E12"], stream=io.StringIO(), tty=False, **kw)


def _status(tmp_path: Path) -> dict:
    data = progress.read_status(tmp_path)
    assert data is not None
    return data


def _ticks(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if getattr(r, progress.TICK, False)]


def test_non_tty_lines_are_throttled_and_say_count_rate_elapsed_and_eta(tmp_path, clock, caplog):
    caplog.set_level(logging.INFO, logger="fukikae")
    with _run(tmp_path, clock), progress.stage("S01E12", "tts"):
        progress.start(325, "cues")
        for _ in range(120):
            clock.t += 3.4
            progress.advance()
    lines = _ticks(caplog)
    # 408 s of work at one line per 30 s at most: 13 lines, never two within 30 s
    assert 10 <= len(lines) <= 14
    assert lines[0].startswith("tts S01E12: 9/325 cues, 3.4 s/cue, 31 s elapsed, ETA 18 min")
    assert lines[-1] == "tts S01E12: 117/325 cues, 3.4 s/cue, 6 min 38 s elapsed, ETA 12 min"
    messages = [r.getMessage() for r in caplog.records]
    assert "tts S01E12: started" in messages
    assert "tts S01E12: done in 6 min 48 s" in messages


def test_non_tty_writes_nothing_to_the_stream_and_no_line_without_a_total(tmp_path, clock, caplog):
    caplog.set_level(logging.INFO, logger="fukikae")
    stream = io.StringIO()
    r = _reporter(tmp_path, clock, stream=stream)
    r.stage_begin("S01E12", "separate")
    r.start(None, "", "demucs")
    clock.t += 300
    r.advance(1)
    r.stage_end(True)
    assert stream.getvalue() == "" and _ticks(caplog) == []


def test_tty_draws_a_bar_and_clears_it_before_log_lines(tmp_path, clock):
    stream = io.StringIO()
    r = _reporter(tmp_path, clock, stream=stream, tty=True)
    r.stage_begin("S01E12", "tts")
    r.start(10, "cues", None)
    clock.t += 2
    r.advance(1)
    assert stream.getvalue().endswith(
        "\rtts S01E12: 1/10 cues, 2.0 s/cue, 2 s elapsed, ETA 18 s\x1b[K"
    )
    clock.t += 0.05
    before = stream.getvalue()
    r.advance(1)  # within render_every: not redrawn
    assert stream.getvalue() == before
    r.clear_line()
    assert stream.getvalue().endswith("\r\x1b[K")


def test_tty_bar_is_cleared_before_a_log_line_from_the_handler(tmp_path, clock):
    stream = io.StringIO()
    logger = logging.getLogger("fukikae")
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    old = logger.level
    logger.setLevel(logging.INFO)
    try:
        r = _reporter(tmp_path, clock, stream=stream, tty=True)
        r.install()
        r.stage_begin("S01E12", "tts")
        r.start(10, "cues", None)
        clock.t += 1
        r.advance(1)
        logging.getLogger("fukikae.x").warning("hello")
        r.uninstall()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
    assert "ETA 9 s\x1b[K\r\x1b[Khello\n" in stream.getvalue()
    assert r.last_log == "hello"


def test_two_threads_report_into_their_own_slots(tmp_path, clock):
    r = _reporter(tmp_path, clock, status_every=0)
    errors: list[BaseException] = []
    seen: dict[str, float] = {}

    def work(name: str, ep: str, stage: str, n: int) -> None:
        try:
            threading.current_thread().name = name
            r.stage_begin(ep, stage)
            r.start(n, "cues", None)
            for _ in range(n):
                r.advance(1)
            seen[name] = r.slots[progress.role()].done
            r.stage_end(True)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=work, args=("main-head", "S01E13", "tts", 2000)),
        threading.Thread(target=work, args=("fukikae-tail_0", "S01E12", "mix", 2000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert seen == {"main-head": 2000, "fukikae-tail_0": 2000}
    json.loads(r.status_path.read_text())  # never torn
    assert r.positions == {"S01E12": "tail", "S01E13": "head"}


def test_status_json_is_written_atomically_with_the_run_and_its_slots(tmp_path, clock, monkeypatch):
    replaced: list[tuple[str, str]] = []
    real = os.replace

    def spy(src, dst):
        replaced.append((str(src), str(dst)))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    r = _reporter(tmp_path, clock, status_every=5)
    r.stage_begin("S01E12", "tts")
    r.start(100, "cues", None)
    n_after_begin = len(replaced)
    clock.t += 1
    r.advance(10)  # within status_every: no write
    assert len(replaced) == n_after_begin
    clock.t += 5
    r.advance(10)
    assert len(replaced) == n_after_begin + 1
    src, dst = replaced[-1]
    assert dst == str(tmp_path / ".fukikae" / "status.json") and src != dst
    assert Path(src).parent == Path(dst).parent and not Path(src).exists()
    data = json.loads(Path(dst).read_text())
    assert data["pid"] == os.getpid() and data["state"] == "running"
    assert data["episodes"] == [{"id": "S01E12", "position": "head"}]
    head = data["threads"]["head"]
    assert (head["stage"], head["done"], head["total"], head["unit"]) == ("tts", 20, 100, "cues")
    assert head["rate"] == pytest.approx(0.3) and head["eta"] == pytest.approx(24.0)
    assert head["eta_from"] == "rate"
    assert not list((tmp_path / ".fukikae").glob(".*.tmp*"))


def test_gpu_waits_and_holds_show_in_status(tmp_path, clock):
    r = _reporter(tmp_path, clock, status_every=0)
    r.gpu_waiting(0, 4242, "tts indextts2", since=clock.wall0.timestamp() - 240)
    data = json.loads(r.status_path.read_text())
    assert data["gpu"] == [
        {"gpu": 0, "thread": "head", "state": "waiting", "holder_pid": 4242,
         "holder": "tts indextts2", "held_since": "2026-11-11T11:56:00+00:00",
         "waiting_since": "2026-11-11T12:00:00+00:00"},
    ]  # fmt: skip
    r.gpu_held(0, "asr")
    assert json.loads(r.status_path.read_text())["gpu"][0]["state"] == "held"
    r.gpu_released(0)
    assert json.loads(r.status_path.read_text())["gpu"] == []


def test_eta_before_the_first_tick_comes_from_earlier_runs(tmp_path, clock, caplog):
    caplog.set_level(logging.INFO, logger="fukikae")
    r = _reporter(tmp_path, clock)
    for seconds in (600, 700, 800):
        r.stage_begin("S01E01", "tts")
        r.start(100, "cues", None)
        clock.t += seconds
        r.advance(100)
        r.stage_end(True)
    r.stage_begin("S01E12", "tts")
    assert "tts S01E12: started (about 12 min on earlier runs)" in caplog.messages
    r.start(200, "cues", None)  # twice the cues: 2 x 7 s per cue
    snap = r.snapshot()["threads"]["head"]
    assert snap["eta"] == pytest.approx(1400) and snap["eta_from"] == "history"


def test_eta_falls_back_to_manifest_durations(tmp_path, clock):
    m = tmp_path / ".fukikae" / "episodes" / "S01E01"
    m.mkdir(parents=True)
    (m / "manifest.json").write_text(json.dumps({"stages": {"separate": {
        "key": "k", "status": "done", "started": "2026-11-10T10:00:00+00:00",
        "finished": "2026-11-10T10:03:00+00:00"}}}))  # fmt: skip
    r = _reporter(tmp_path, clock)
    r.stage_begin("S01E12", "separate")
    clock.t += 60
    snap = r.snapshot()["threads"]["head"]
    assert snap["eta"] == pytest.approx(120) and snap["eta_from"] == "history"


def test_functions_do_nothing_without_a_reporter(tmp_path):
    assert progress.active() is None
    with progress.stage("S01E12", "tts"):
        progress.start(3, "cues")
        progress.advance()
        progress.finish()
    assert not (tmp_path / ".fukikae").exists()


def test_run_ends_finished_failed_or_interrupted(tmp_path, clock):
    with _run(tmp_path, clock):
        pass
    assert _status(tmp_path)["state"] == "finished"
    with pytest.raises(RuntimeError), _run(tmp_path):
        raise RuntimeError("boom")
    assert _status(tmp_path)["state"] == "failed"
    with pytest.raises(KeyboardInterrupt), _run(tmp_path):
        raise KeyboardInterrupt
    data = _status(tmp_path)
    assert data["state"] == "interrupted" and data["ended"] and data["threads"] == {}


# -- fukikae status ----------------------------------------------------------------------------


def _live_status(tmp_path: Path, clock: FakeClock) -> Reporter:
    r = _reporter(tmp_path, clock, status_every=0)
    r.last_log = "tts: 3 cards checked"
    r.stage_begin("S01E12", "tts")
    r.start(325, "cues", None)
    clock.t += 408
    r.advance(120)
    r.gpu_held(0, "tts indextts2")
    return r


def test_status_of_a_live_run(tmp_path, clock):
    _live_status(tmp_path, clock)
    status = progress.read_status(tmp_path)
    assert status is not None and progress.is_live(status)
    now = clock.wall() + timedelta(seconds=10)
    text = progress.describe(status, now=now)
    assert "pid " + str(os.getpid()) in text
    assert "episodes: S01E12 head" in text
    # elapsed and ETA move on with the clock between writes
    assert "head: tts S01E12: 120/325 cues, 3.4 s/cue, 6 min 58 s elapsed, ETA 11 min" in text
    assert "GPU 0 (head): held by this run (tts indextts2)" in text
    assert "last log: tts: 3 cards checked" in text
    assert "updated 10 s ago" in text


def test_status_of_a_dead_run_says_it_died(tmp_path, clock):
    _live_status(tmp_path, clock)
    path = tmp_path / ".fukikae" / "status.json"
    data = json.loads(path.read_text())
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    data["pid"] = dead.pid
    data["pid_start"] = None
    path.write_text(json.dumps(data))
    status = progress.read_status(tmp_path)
    assert status is not None and not progress.is_live(status)
    text = progress.describe(status)
    assert text.startswith("no run in progress (last run died at ")
    assert "episodes: S01E12 head" in text


def test_status_of_a_reused_pid_is_not_live(tmp_path, clock):
    _live_status(tmp_path, clock)
    data = progress.read_status(tmp_path)
    assert data is not None
    data["pid_start"] = "1"  # our pid, but not the process that wrote it
    assert not progress.is_live(data)


def test_status_of_a_finished_run_and_of_none(tmp_path, clock):
    with _run(tmp_path, clock):
        progress.episode("S01E12", "done")
    text = progress.describe(progress.read_status(tmp_path))
    assert text == (
        "no run in progress (last run finished at "
        f"{clock.wall0.astimezone().strftime('%Y-%m-%d %H:%M:%S')}; episodes: S01E12 done)"
    )
    assert progress.describe(progress.read_status(tmp_path / "nowhere")) == (
        "no run in progress (no run recorded in this workspace)"
    )


def test_fukikae_status_command(tmp_path, clock):
    from typer.testing import CliRunner

    from fukikae.cli import app

    _live_status(tmp_path, clock)
    res = CliRunner().invoke(app, ["status", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "tts S01E12: 120/325 cues" in res.output
    res = CliRunner().invoke(app, ["status", str(tmp_path), "--json"])
    data = json.loads(res.output)
    assert data["live"] is True and data["threads"]["head"]["done"] == 120
