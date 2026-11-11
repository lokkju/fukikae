# ruff: noqa: ARG002
from pathlib import Path

import pytest

from fukikae.artifacts import atomic_path, read_json
from fukikae.config import Config, load_config
from fukikae.job import Manifest, cache_key, run_stages
from fukikae.workspace import Episode


class FakeA:
    name = "a"

    def __init__(self, counts, write=None):
        self.counts = counts
        self.write = write  # fixed content override

    def inputs(self, ep):
        return []

    def outputs(self, ep):
        return [ep.work / "a.txt"]

    def config_key(self, cfg, ep):
        return {"value": cfg.raw["fake"]["value"]}

    def run(self, ep, cfg):
        self.counts["a"] = self.counts.get("a", 0) + 1
        ep.work.mkdir(parents=True, exist_ok=True)
        with atomic_path(ep.work / "a.txt") as tmp:
            tmp.write_text(self.write or cfg.raw["fake"]["value"])


class FakeB:
    name = "b"

    def __init__(self, counts, fail=None):
        self.counts = counts
        self.fail = fail

    def inputs(self, ep):
        return [ep.work / "a.txt"]

    def outputs(self, ep):
        return [ep.work / "b.txt"]

    def config_key(self, cfg, ep):
        return {}

    def run(self, ep, cfg):
        self.counts["b"] = self.counts.get("b", 0) + 1
        with atomic_path(ep.work / "b.txt") as tmp:
            tmp.write_text((ep.work / "a.txt").read_text() + "!")
            if self.fail:
                raise self.fail


def make_cfg(root: Path, value: str) -> Config:
    (root / "fukikae.toml").write_text(f'[fake]\nvalue = "{value}"\n')
    return load_config(root, user_config=None)


@pytest.fixture
def ep(tmp_path):
    src = tmp_path / "src.mkv"
    src.write_bytes(b"media")
    ws = tmp_path / "ws"
    ws.mkdir()
    return Episode(source=src, id="e1", work=ws / "e1", workspace_root=ws)


@pytest.fixture
def cfg(ep):
    return make_cfg(ep.workspace_root, "x")


def test_second_run_skips(ep, cfg):
    counts = {}
    stages = [FakeA(counts), FakeB(counts)]
    assert run_stages(ep, cfg, stages) == ["a", "b"]
    assert run_stages(ep, cfg, stages) == []
    assert counts == {"a": 1, "b": 1}


def test_config_change_reruns_downstream(ep, cfg):
    counts = {}
    stages = [FakeA(counts), FakeB(counts)]
    run_stages(ep, cfg, stages)
    cfg2 = make_cfg(ep.workspace_root, "y")
    assert run_stages(ep, cfg2, stages) == ["a", "b"]


def test_same_output_content_stops_propagation(ep, cfg):
    counts = {}
    run_stages(ep, cfg, [FakeA(counts, write="same"), FakeB(counts)])
    cfg2 = make_cfg(ep.workspace_root, "y")
    assert run_stages(ep, cfg2, [FakeA(counts, write="same"), FakeB(counts)]) == ["a"]


def test_hand_edit_reruns_only_downstream(ep, cfg):
    counts = {}
    stages = [FakeA(counts), FakeB(counts)]
    run_stages(ep, cfg, stages)
    (ep.work / "a.txt").write_text("edited by hand")
    assert run_stages(ep, cfg, stages) == ["b"]
    assert (ep.work / "b.txt").read_text() == "edited by hand!"


def test_missing_output_reruns(ep, cfg):
    counts = {}
    stages = [FakeA(counts), FakeB(counts)]
    run_stages(ep, cfg, stages)
    (ep.work / "b.txt").unlink()
    assert run_stages(ep, cfg, stages) == ["b"]


def test_from_stage_forces(ep, cfg):
    counts = {}
    stages = [FakeA(counts), FakeB(counts)]
    run_stages(ep, cfg, stages)
    assert run_stages(ep, cfg, stages, from_stage="b") == ["b"]


def test_to_stage_stops(ep, cfg):
    counts = {}
    assert run_stages(ep, cfg, [FakeA(counts), FakeB(counts)], to_stage="a") == ["a"]


def test_unknown_stage_names_raise(ep, cfg):
    stages = [FakeA({}), FakeB({})]
    with pytest.raises(ValueError):
        run_stages(ep, cfg, stages, from_stage="nope")
    with pytest.raises(ValueError):
        run_stages(ep, cfg, stages, to_stage="nope")


def test_failure_records_and_reraises(ep, cfg):
    counts = {}
    with pytest.raises(RuntimeError, match="boom"):
        run_stages(ep, cfg, [FakeA(counts), FakeB(counts, fail=RuntimeError("boom"))])
    m = read_json(ep.work / "manifest.json", Manifest)
    assert m.stages["a"].status == "done"
    assert m.stages["b"].status == "failed"
    assert "boom" in (m.stages["b"].error or "")
    assert run_stages(ep, cfg, [FakeA(counts), FakeB(counts)]) == ["b"]


def test_interrupted_stage_reruns(ep, cfg):
    counts = {}
    run_stages(ep, cfg, [FakeA(counts), FakeB(counts)])
    assert (ep.work / "b.txt").read_text() == "x!"
    cfg2 = make_cfg(ep.workspace_root, "y")
    with pytest.raises(KeyboardInterrupt):
        run_stages(ep, cfg2, [FakeA(counts), FakeB(counts, fail=KeyboardInterrupt())])
    # The interrupted write left the previous complete output untouched, no temp files.
    assert (ep.work / "b.txt").read_text() == "x!"
    assert list(ep.work.glob(".*.tmp*")) == []
    m = read_json(ep.work / "manifest.json", Manifest)
    assert m.stages["b"].status == "failed"
    assert run_stages(ep, cfg2, [FakeA(counts), FakeB(counts)]) == ["b"]
    assert (ep.work / "b.txt").read_text() == "y!"


def test_external_source_fingerprinted_by_stat(ep, cfg):
    class S(FakeA):
        def inputs(self, ep):
            return [ep.source]

    k1 = cache_key(S({}), ep, cfg)
    assert cache_key(S({}), ep, cfg) == k1
    ep.source.write_bytes(b"media, longer")
    assert cache_key(S({}), ep, cfg) != k1
    assert not (ep.workspace_root / ".fukikae").exists()


def test_missing_optional_input_key_does_not_depend_on_the_workspace_path(tmp_path, cfg):
    """A copied or moved workspace isn't stale because an optional input (glossary.toml) is
    absent: the missing file is named relative to the workspace (ruling CD6)."""

    class S(FakeA):
        def inputs(self, ep):
            return [ep.workspace_root / "glossary.toml", ep.work / "maybe.json"]

    keys = []
    for name in ("one", "two"):
        root = tmp_path / name / "ws"
        ep = Episode(
            source=tmp_path / "s.mkv", id="e1", work=root / ".fukikae" / "episodes" / "e1",
            workspace_root=root,
        )  # fmt: skip
        keys.append(cache_key(S({}), ep, cfg))
    assert keys[0] == keys[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / "glossary.toml").write_text("[terms]\n")
    assert cache_key(S({}), ep, cfg) != keys[1]


# ---- several episodes: the CPU tail overlaps the next episode's head (D26) ----------------

import threading  # noqa: E402

from fukikae.job import run_episodes  # noqa: E402


def _eps(tmp_path, n):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return [
        Episode(source=tmp_path / f"s{i}.mkv", id=f"e{i}", work=ws / f"e{i}", workspace_root=ws)
        for i in range(1, n + 1)
    ]


class Head(FakeA):
    def __init__(self, counts, events):
        super().__init__(counts)
        self.events = events

    def run(self, ep, cfg):
        self.events.append(f"head {ep.id}")
        if ep.id == "e2":
            self.started_e2.set()
        super().run(ep, cfg)

    started_e2 = threading.Event()


class Tail(FakeB):
    def __init__(self, counts, events, fail_on=None):
        super().__init__(counts)
        self.events = events
        self.fail_on = fail_on
        self.overlapped = False

    def run(self, ep, cfg):
        if ep.id == "e1":  # the tail of e1 waits for e2's head: they overlap
            self.overlapped = Head.started_e2.wait(timeout=10)
        self.events.append(f"tail {ep.id}")
        if ep.id == self.fail_on:
            raise RuntimeError(f"tail {ep.id} broke")
        super().run(ep, cfg)


def test_run_episodes_overlaps_a_tail_with_the_next_head(tmp_path):
    eps = _eps(tmp_path, 3)
    cfg = make_cfg(eps[0].workspace_root, "x")
    Head.started_e2 = threading.Event()
    counts: dict = {}
    events: list[str] = []
    tail = Tail(counts, events)
    results = run_episodes(eps, cfg, [Head(counts, events), tail], tail=("b",))
    assert tail.overlapped
    assert {e: r.ran for e, r in results.items()} == {e.id: ["a", "b"] for e in eps}
    assert all(r.error is None for r in results.values())
    # heads in order, tails in order; outputs are what a sequential run writes
    assert [x for x in events if x.startswith("head")] == ["head e1", "head e2", "head e3"]
    assert [x for x in events if x.startswith("tail")] == ["tail e1", "tail e2", "tail e3"]
    for e in eps:
        assert (e.work / "b.txt").read_text() == "x!"
        assert run_stages(e, cfg, [FakeA(counts), FakeB(counts)]) == []  # manifests complete


def test_run_episodes_stops_starting_heads_after_a_failure(tmp_path):
    eps = _eps(tmp_path, 3)
    cfg = make_cfg(eps[0].workspace_root, "x")
    Head.started_e2 = threading.Event()
    Head.started_e2.set()
    counts: dict = {}
    events: list[str] = []
    stages = [Head(counts, events), Tail(counts, events, "e1")]
    results = run_episodes(eps, cfg, stages, tail=("b",))
    assert isinstance(results["e1"].error, RuntimeError)
    assert "e3" not in results or results["e3"].ran == []
    assert "head e3" not in events


def test_run_episodes_honours_from_and_to(tmp_path):
    eps = _eps(tmp_path, 2)
    cfg = make_cfg(eps[0].workspace_root, "x")
    Head.started_e2 = threading.Event()
    Head.started_e2.set()
    counts: dict = {}
    events: list[str] = []
    stages = [Head(counts, events), Tail(counts, events)]
    results = run_episodes(eps, cfg, stages, tail=("b",), to_stage="a")
    assert {e: r.ran for e, r in results.items()} == {"e1": ["a"], "e2": ["a"]}
    results = run_episodes(eps, cfg, stages, tail=("b",), from_stage="b")
    assert {e: r.ran for e, r in results.items()} == {"e1": ["b"], "e2": ["b"]}
    results = run_episodes(eps, cfg, stages, tail=("b",), from_stage="a")
    assert {e: r.ran for e, r in results.items()} == {"e1": ["a", "b"], "e2": ["a", "b"]}


def test_run_episodes_shows_each_episode_and_thread_in_the_status(tmp_path):
    """#29: while e1's tail runs beside e2's head, the status names both threads' stages, and
    each episode's position (head, tail, done)."""
    import io

    from fukikae import progress

    eps = _eps(tmp_path, 2)
    cfg = make_cfg(eps[0].workspace_root, "x")
    tail_started = threading.Event()
    snaps: list[dict] = []

    class SeeingHead(FakeA):
        def run(self, ep, cfg):
            if ep.id == "e2":
                assert tail_started.wait(10)
                r = progress.active()
                assert r is not None
                snaps.append(r.snapshot())
            super().run(ep, cfg)

    class SlowTail(FakeB):
        def run(self, ep, cfg):
            tail_started.set()
            if ep.id == "e1":
                while not snaps:
                    threading.Event().wait(0.01)
            super().run(ep, cfg)

    counts: dict = {}
    root = eps[0].workspace_root
    with progress.run(root, ["e1", "e2"], stream=io.StringIO(), tty=False):
        run_episodes(eps, cfg, [SeeingHead(counts), SlowTail(counts)], tail=("b",))
    (snap,) = snaps
    threads = {k: (v["episode"], v["stage"]) for k, v in snap["threads"].items()}
    assert threads == {"head": ("e2", "a"), "tail": ("e1", "b")}
    assert snap["episodes"] == [{"id": "e1", "position": "tail"}, {"id": "e2", "position": "head"}]
    final = progress.read_status(root)
    assert final is not None and final["state"] == "finished"
    assert [e["position"] for e in final["episodes"]] == ["done", "done"]
