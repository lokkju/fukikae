"""The studio lock against `registry resolve` and `registry cluster`, runs that wait out a short
decision, and the lock file's own safety (final whole-branch review, Part B)."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae import job
from fukikae.cli import app
from fukikae.registry.store import Registry
from fukikae.studio import apply as apply_mod
from fukikae.studio.apply import DecisionRefused, apply_decision
from fukikae.workspace import Episode, Workspace
from tests.registry import test_cluster, test_resolve
from tests.studio.test_apply import decision_for
from tests.studio.test_records import _Stage

UNFINISHED = {
    "id": "c0ffee", "time": "2026-11-17T00:00:00+00:00", "op": "merge", "completes": "abc",
    "incomplete": {"left": [{"do": "stale", "episode": "S01E01", "stage": "characters"}],
                   "errors": ["a remap failed"]},
}  # fmt: skip


def _unfinished(ws: Workspace) -> None:
    log = apply_mod.decisions_log(ws)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(UNFINISHED) + "\n")


def _ep(ws: Workspace, eid: str) -> Episode:
    return Episode(source=Path("x"), id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


# --- 1. resolve and cluster take the studio lock and refuse unfinished decisions ----------


def _resolve(ws: Workspace):
    return test_resolve.resolve(ws, "S01E01,S01E02")


def _cluster(ws: Workspace):
    return CliRunner().invoke(
        app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )


def _assignments(ws: Workspace) -> list[Path]:
    return sorted(ws.root.glob(".fukikae/episodes/*/05-characters/assignments.json"))


@pytest.mark.parametrize(("make", "run"), [(test_resolve.make_ws, _resolve),
                                           (test_cluster.make_ws, _cluster)])  # fmt: skip
def test_resolve_and_cluster_are_refused_while_a_writer_holds_the_studio_lock(tmp_path, make, run):
    ws = make(tmp_path)
    with apply_mod.studio_lock(ws, "probe: a studio decision mid-steps"):
        r = run(ws)
    assert r.exit_code == 1, r.output
    assert "probe: a studio decision mid-steps" in r.output
    assert _assignments(ws) == []
    with Registry(ws.registry_path) as reg:
        assert reg.actors() == []


@pytest.mark.parametrize(("make", "run"), [(test_resolve.make_ws, _resolve),
                                           (test_cluster.make_ws, _cluster)])  # fmt: skip
def test_resolve_and_cluster_are_refused_while_a_run_holds_the_lock(tmp_path, make, run):
    ws = make(tmp_path)
    with job_lock(ws):
        r = run(ws)
    assert r.exit_code == 1 and "a `fukikae run` is in progress" in r.output, r.output
    assert _assignments(ws) == []


@pytest.mark.parametrize(("make", "run"), [(test_resolve.make_ws, _resolve),
                                           (test_cluster.make_ws, _cluster)])  # fmt: skip
def test_resolve_and_cluster_refuse_an_unfinished_decision(tmp_path, make, run):
    ws = make(tmp_path)
    _unfinished(ws)
    r = run(ws)
    assert r.exit_code == 1, r.output
    assert "fukikae studio resume" in r.output and "incomplete" in r.output
    assert _assignments(ws) == []
    with Registry(ws.registry_path) as reg:
        assert reg.actors() == []


def test_resolve_still_runs_with_no_studio_activity(tmp_path):
    ws = test_resolve.make_ws(tmp_path)
    r = _resolve(ws)
    assert r.exit_code == 0, r.output
    assert len(_assignments(ws)) == 2


class job_lock:  # noqa: N801 (a context manager named for what it stands in for)
    """``studio/.lock`` held shared, as `fukikae run` holds it."""

    def __init__(self, ws: Workspace) -> None:
        self.path = apply_mod.studio_dir(ws) / ".lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")
        fcntl.flock(self.f, fcntl.LOCK_SH)
        return self

    def __exit__(self, *exc):
        self.f.close()


# --- 2. a run waits a bounded time for a writer ----------------------------------------------


class _Hold:
    """Takes ``studio/.lock`` exclusively, as a studio decision does, on another thread, and
    holds it for ``hold_s`` seconds or until `release`."""

    def __init__(self, ws: Workspace, hold_s: float | None) -> None:
        self.ws = ws
        self.hold_s = hold_s
        self.taken = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with apply_mod.studio_lock(self.ws, "`fukikae studio` decision (test)"):
            self.taken.set()
            self.done.wait(self.hold_s)

    def start(self) -> None:
        self.thread.start()
        assert self.taken.wait(5)

    def release(self) -> None:
        self.done.set()
        self.thread.join(5)


def _hold_before(monkeypatch, ws: Workspace, eid: str, hold_s: float | None) -> list[_Hold]:
    """Before ``eid``'s head starts, a decision takes the lock (held ``hold_s`` seconds)."""
    real = job.progress.episode
    holds: list[_Hold] = []

    def episode(e: str, state: str) -> None:
        if e == eid and state == "head" and not holds:
            h = _Hold(ws, hold_s)
            h.start()
            holds.append(h)
        real(e, state)

    monkeypatch.setattr(job.progress, "episode", episode)
    return holds


def test_a_run_waits_for_a_short_decision_and_goes_on(ws, monkeypatch):
    stage = _Stage("characters")
    holds = _hold_before(monkeypatch, ws, "S01E02", 1.0)
    t0 = time.monotonic()
    res = job.run_episodes([_ep(ws, "S01E01"), _ep(ws, "S01E02")], ws.config, [stage],
                           tail=(), from_stage="characters")  # fmt: skip
    waited = time.monotonic() - t0
    assert holds and res["S01E02"].error is None, res
    assert res["S01E01"].ran == ["characters"] and res["S01E02"].ran == ["characters"]
    assert stage.ran == 2 and waited >= 0.9


def test_a_run_that_cant_get_the_lock_stops_that_episode_and_tries_the_next(ws, monkeypatch):
    monkeypatch.setattr(apply_mod, "RUN_LOCK_WAIT_S", 0.3)
    monkeypatch.setattr(apply_mod, "RUN_LOCK_POLL_S", 0.05)
    stage = _Stage("characters")
    holds = _hold_before(monkeypatch, ws, "S01E02", None)
    real = job.progress.episode

    def episode(e: str, state: str) -> None:
        real(e, state)
        if e == "S01E02" and state != "head":
            holds[0].release()  # the decision ends once S01E02 has given up

    monkeypatch.setattr(job.progress, "episode", episode)
    eps = [_ep(ws, e) for e in ("S01E01", "S01E02", "S01E03")]
    res = job.run_episodes(eps, ws.config, [stage], tail=(), from_stage="characters")
    err = res["S01E02"].error
    assert isinstance(err, apply_mod.StudioLockTimeout) and isinstance(err, ValueError)
    assert "`fukikae studio` decision (test)" in str(err) and "waited" in str(err)
    assert res["S01E02"].ran == []
    assert res["S01E03"].error is None and res["S01E03"].ran == ["characters"]


def test_the_cli_reports_a_lock_wait_as_an_error_not_a_traceback(ws, monkeypatch):
    from fukikae import cli as cli_mod

    monkeypatch.setattr(apply_mod, "RUN_LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(apply_mod, "RUN_LOCK_POLL_S", 0.05)
    monkeypatch.setattr(cli_mod, "pipeline_stages", lambda _cfg: [_Stage("characters")])
    monkeypatch.setattr("fukikae.workspace.Workspace.find_episode",
                        lambda _self, ref: _ep(ws, ref))  # fmt: skip
    with apply_mod.studio_lock(ws, "`fukikae studio` decision (test)"):
        r = CliRunner().invoke(app, ["run", str(ws.root), "S01E01", "S01E02"])
    assert r.exit_code == 1, r.output
    assert r.exception is None or isinstance(r.exception, SystemExit), r.exception
    assert "error: S01E01:" in r.output and "error: S01E02:" in r.output
    assert "Traceback" not in r.output


def test_a_single_run_waits_for_a_writer(ws):
    h = _Hold(ws, 0.5)
    h.start()
    t0 = time.monotonic()
    assert job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("characters")],
                          from_stage="characters") == ["characters"]  # fmt: skip
    assert time.monotonic() - t0 >= 0.4
    h.thread.join(5)


# --- 3. a run always makes the lock, so a decision during it is serialised -----------------


def test_a_run_makes_the_studio_lock_and_a_decision_during_it_is_refused(ws):
    assert not apply_mod.studio_dir(ws).exists()
    refused: list[str] = []

    class Deciding(_Stage):
        def run(self, ep, cfg) -> None:
            super().run(ep, cfg)
            try:
                apply_decision(ws, decision_for(ws, "lock", character=3))
            except DecisionRefused as e:
                refused.append(str(e))

    job.run_stages(_ep(ws, "S01E01"), ws.config, [Deciding("characters")],
                   from_stage="characters")  # fmt: skip
    assert refused and "a `fukikae run` is in progress" in refused[0]
    assert sorted(p.name for p in apply_mod.studio_dir(ws).iterdir()) == [".lock"]


# --- 5. the lock file is never followed through a symlink ------------------------------------


@pytest.mark.parametrize("who", ["writer", "run"])
def test_a_symlinked_lock_file_is_refused_and_its_target_left_alone(ws, tmp_path, who):
    target = tmp_path / "precious.txt"
    target.write_text("keep me\n")
    apply_mod.studio_dir(ws).mkdir()
    (apply_mod.studio_dir(ws) / ".lock").symlink_to(target)
    if who == "writer":
        with pytest.raises(DecisionRefused, match="symlink"):
            apply_decision(ws, decision_for(ws, "lock", character=3))
    else:
        with pytest.raises(ValueError, match="symlink"):
            job.run_stages(_ep(ws, "S01E01"), ws.config, [_Stage("characters")])
    assert target.read_text() == "keep me\n"


def test_the_decisions_log_is_owner_only(ws):
    apply_decision(ws, decision_for(ws, "lock", character=3))
    log = apply_mod.decisions_log(ws)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((log.parent / ".lock").stat().st_mode) & 0o077 == 0


def test_an_existing_studio_dir_is_made_owner_only_by_the_log(ws):
    d = apply_mod.studio_dir(ws)
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)  # noqa: S103 (the mode the fix must tighten)
    apply_decision(ws, decision_for(ws, "lock", character=3))
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
