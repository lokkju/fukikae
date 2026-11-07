"""The offline simulation against the pipeline itself: the characters stage (which calls
`registry.matching.match` on a real registry) and the harness's centroid matcher must make
the same decisions on the same clusters and embeddings."""

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, RegistrySettings
from fukikae.eval.identity.segments import Segment, diarization_segments
from fukikae.eval.identity.simulate import simulate
from fukikae.models import Assignments, Diarization, Turn
from fukikae.registry.matchers import CentroidMatcher, RegistryState, make_matcher
from fukikae.registry.matching import ClusterEvidence, match, mean_embedding
from fukikae.stages.characters import CharactersStage
from fukikae.stages.diarize import MAX_SPANS_PER_CLUSTER, clean_spans
from fukikae.workspace import Episode

DIM = 8
BASE = {s: np.eye(DIM, dtype=np.float32)[i] for i, s in enumerate(["A", "B", "C", "D", "E"])}
# B and E are close voices: E's centroid lands near B's.
BASE["E"] = (BASE["B"] * 0.9 + BASE["E"] * 0.45).astype(np.float32)


def embed_span(episode: str, speaker: str, start: float) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(f"{episode}/{speaker}/{start:.3f}".encode()))
    v = BASE[speaker] + rng.normal(0, 0.25, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def turns_for(spec: list[tuple[str, str, int, float]]) -> list[Turn]:
    """spec: (cluster label, true speaker, number of turns, turn length); laid end to end."""
    out, t = [], 0.0
    for label, _speaker, n, length in spec:
        for k in range(n):
            ln = length + 0.37 * (k % 5)  # varied lengths, so the 50-longest cap matters
            out.append(Turn(start=t, end=t + ln, speaker=label))
            t += ln + 0.5
    return out


# Episode -> clusters as (label, true speaker, turns, turn length).
EPISODES = {
    "S01E01": [
        ("SPEAKER_00", "A", 60, 2.0),
        ("SPEAKER_01", "B", 12, 3.0),
        ("SPEAKER_02", "C", 2, 2.5),
    ],
    "S01E02": [
        ("SPEAKER_00", "B", 10, 3.0),
        ("SPEAKER_01", "A", 15, 2.0),
        ("SPEAKER_02", "B", 4, 2.0),  # B split over two clusters: one matched, one review
        ("SPEAKER_03", "D", 10, 2.5),
    ],
    "S01E03": [
        ("SPEAKER_00", "C", 10, 2.5),
        ("SPEAKER_01", "E", 10, 2.5),
        ("SPEAKER_02", "A", 5, 2.0),
    ],
}


def speaker_of(ep: str) -> dict[str, str]:
    return {label: spk for label, spk, _, _ in EPISODES[ep]}


def episode_data(ep: str):
    turns = turns_for(EPISODES[ep])
    segs = diarization_segments(ep, turns, [])
    who = speaker_of(ep)
    embs = np.stack([embed_span(ep, who[str(s.cluster)], s.start) for s in segs])
    return turns, segs, embs


def write_stage_inputs(work: Path, ep: str, turns: list[Turn]) -> None:
    """turns.json and clusters.npz exactly as the diarize stage writes them."""
    d = work / "04-diarize"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "turns.json", Diarization(backend="fake", turns=turns))
    who = speaker_of(ep)
    labels = list(dict.fromkeys(t.speaker for t in turns))
    arrays: dict[str, Any] = {"labels": np.array(labels, dtype=str)}
    for label in labels:
        clean = clean_spans(turns, label)
        spans = sorted(clean, key=lambda s: s[1] - s[0], reverse=True)[:MAX_SPANS_PER_CLUSTER]
        rows = [embed_span(ep, who[label], s) for s, _ in spans]
        arrays[f"{label}/embeddings"] = np.array(rows, dtype=np.float32).reshape(len(spans), DIM)
        arrays[f"{label}/spans"] = np.array(spans, dtype=np.float64).reshape(len(spans), 2)
        arrays[f"{label}/clean_seconds"] = np.array(sum(e - s for s, e in clean))
    np.savez(d / "clusters.npz", **arrays)


@pytest.mark.parametrize("threshold", [0.6, 0.8])
def test_centroid_reproduces_the_characters_stage(tmp_path: Path, threshold: float):
    cfg = Config(registry=RegistrySettings(min_new_seconds=20.0, match_threshold=threshold))
    root = tmp_path / "ws"
    stage: dict[str, dict[str, Any]] = {}
    segs, embs = {}, {}
    for ep in EPISODES:
        turns, segs[ep], embs[ep] = episode_data(ep)
        work = root / ".fukikae" / "episodes" / ep
        write_stage_inputs(work, ep, turns)
        CharactersStage().run(Episode(tmp_path / f"{ep}.mkv", ep, work, root), cfg)
        got = read_json(work / "05-characters" / "assignments.json", Assignments)
        stage[ep] = {c.cluster: c for c in got.clusters}

    matcher = make_matcher("centroid", threshold=threshold, min_new_seconds=20.0)
    state = RegistryState()
    assign = simulate(list(EPISODES), matcher, segs, embs, state=state)

    for ep in EPISODES:
        for i, s in enumerate(segs[ep]):
            assert assign[(ep, i)] == stage[ep][s.cluster].character_id, (ep, s.cluster)
    statuses = {ep: {k: c.status for k, c in stage[ep].items()} for ep in EPISODES}
    # The scenario exercises every decision the matcher can make.
    assert statuses["S01E01"] == {"SPEAKER_00": "new", "SPEAKER_01": "new", "SPEAKER_02": "extra"}
    assert "review" in statuses["S01E02"].values() and "matched" in statuses["S01E02"].values()
    assert sorted(state.actors) == sorted(
        {c.actor_id for ep in stage.values() for c in ep.values() if c.status in ("new",)}
    )


def test_centroid_decisions_equal_match_on_cluster_means():
    """One episode against a fixed state: the matcher's statuses are `match`'s."""
    _, segs1, embs1 = episode_data("S01E01")
    _, segs2, embs2 = episode_data("S01E02")
    m = CentroidMatcher(threshold=0.6, min_new_seconds=20.0)
    state = RegistryState()
    m.fit_episode(state, "S01E01", embs1, segs1)
    actors, chars = state.actor_list(), state.character_list()
    evidence = []
    for label in dict.fromkeys(s.cluster for s in segs2):
        idx = [i for i, s in enumerate(segs2) if s.cluster == label]
        longest = sorted(idx, key=lambda i: segs2[i].duration, reverse=True)[:50]
        evidence.append(
            ClusterEvidence(
                str(label),
                mean_embedding(embs2[longest]),
                sum(segs2[i].duration for i in idx),
            )
        )
    want = match(evidence, actors, 0.6, 20.0, characters=chars)
    m.fit_episode(state, "S01E02", embs2, segs2)
    assert [(a.cluster, a.status, a.score) for a in m.last] == [
        (a.cluster, a.status, a.score) for a in want
    ]
    assert [a.actor_id for a in m.last if a.status != "new"] == [
        a.actor_id for a in want if a.status != "new"
    ]


def test_subtitle_segmentation_clusters_per_episode():
    """Segments without a cluster are grouped by average-linkage cosine at the threshold."""
    rng = np.random.default_rng(0)

    def episode(ep, speakers):
        segs, rows, t = [], [], 0.0
        for spk, n in speakers:
            for _ in range(n):
                segs.append(Segment(ep, t, t + 3.0, spk))
                v = BASE[spk] + rng.normal(0, 0.1, DIM)
                rows.append(v / np.linalg.norm(v))
                t += 3.5
        return segs, np.array(rows, dtype=np.float32)

    s1, e1 = episode("E1", [("A", 10), ("B", 10), ("C", 2)])
    s2, e2 = episode("E2", [("B", 10), ("A", 10)])
    assign = simulate(
        ["E1", "E2"], CentroidMatcher(0.6, 20.0), {"E1": s1, "E2": s2}, {"E1": e1, "E2": e2}
    )
    ids = {}
    for ep, segs in (("E1", s1), ("E2", s2)):
        for i, s in enumerate(segs):
            ids.setdefault(s.label, set()).add(assign[(ep, i)])
    assert len(ids["A"]) == 1 and len(ids["B"]) == 1
    assert ids["A"] != ids["B"] and None not in ids["A"] | ids["B"]
    assert ids["C"] == {None}  # 6 s: under min_new_seconds, an extra


def test_simulate_resolves_in_order_against_earlier_episodes():
    _, s1, e1 = episode_data("S01E01")
    _, s3, e3 = episode_data("S01E03")
    a = simulate(
        ["S01E01", "S01E03"],
        make_matcher("centroid"),
        {"S01E01": s1, "S01E03": s3},
        {"S01E01": e1, "S01E03": e3},
    )
    b = simulate(
        ["S01E03", "S01E01"],
        make_matcher("centroid"),
        {"S01E01": s1, "S01E03": s3},
        {"S01E01": e1, "S01E03": e3},
    )
    # A is new in E1 and matched in E3 in one order; the other order creates it from E3.
    a_e1 = {a[("S01E01", i)] for i, s in enumerate(s1) if s.cluster == "SPEAKER_00"}
    a_e3 = {a[("S01E03", i)] for i, s in enumerate(s3) if s.cluster == "SPEAKER_02"}
    assert a_e1 == a_e3 == {1}
    assert {b[("S01E03", i)] for i, s in enumerate(s3) if s.cluster == "SPEAKER_00"} == {1}


def test_unknown_matcher():
    with pytest.raises(ValueError, match="matcher"):
        make_matcher("nope")
