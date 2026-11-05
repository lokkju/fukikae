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
