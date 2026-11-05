import numpy as np

from doppio.models import Assignments, ClusterAssignment, Diarization, Turn
from doppio.registry.matching import ClusterEvidence, match, pin
from doppio.registry.store import Actor, Character


def unit(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def actor(i, v):
    return Actor(id=i, name=None, external_id=None, centroid=unit(v), locked=False)


def test_match_new_extra_and_review():
    a = actor(1, [1, 0, 0])
    ev = [
        ClusterEvidence("S0", unit([1, 0.1, 0]), 60.0),
        ClusterEvidence("S1", unit([0.95, 0.2, 0]), 40.0),
        ClusterEvidence("S2", unit([0, 1, 0]), 25.0),
        ClusterEvidence("S3", unit([0, 0, 1]), 5.0),
        ClusterEvidence("S4", None, 0.0),
    ]
    got = match(ev, [a], threshold=0.8, min_new_seconds=20.0)
    assert {c.cluster: c.status for c in got} == {
        "S0": "matched",
        "S1": "review",
        "S2": "new",
        "S3": "extra",
        "S4": "extra",
    }
    by = {c.cluster: c for c in got}
    assert by["S0"].actor_id == 1 and by["S1"].actor_id == 1
    assert by["S1"].score is not None and by["S1"].score >= 0.8
    assert by["S2"].actor_id is None and by["S3"].actor_id is None
    assert by["S4"].score is None


def test_matched_uses_first_character():
    a = actor(1, [1, 0, 0])
    chars = [
        Character(id=9, actor_id=1, name=None, voice=None, locked=False),
        Character(id=4, actor_id=1, name=None, voice=None, locked=False),
    ]
    got = match([ClusterEvidence("S0", unit([1, 0, 0]), 30.0)], [a], 0.8, 20.0, characters=chars)
    assert got[0].character_id == 4


def test_empty_inputs():
    assert match([], [], 0.6, 20.0) == []
    got = match([ClusterEvidence("S0", unit([1, 0, 0]), 30.0)], [], 0.6, 20.0)
    assert got[0].status == "new" and got[0].score is None


def _ca(cluster, aid, cid):
    return ClusterAssignment(
        cluster=cluster, actor_id=aid, character_id=cid, score=0.9, clean_seconds=10.0,
        status="matched",
    )  # fmt: skip


def test_pin_inherits_by_overlap():
    prev_turns = Diarization(
        backend="x",
        turns=[Turn(start=0, end=10, speaker="A"), Turn(start=10, end=20, speaker="B")],
    )
    cur_turns = Diarization(
        backend="x",
        turns=[
            Turn(start=0.5, end=9.5, speaker="Z"),  # inside prev A
            Turn(start=9, end=19, speaker="Y"),  # 1 s in A, 9 s in B -> B
            Turn(start=30, end=40, speaker="W"),  # nothing before
        ],
    )
    prev = Assignments(episode="e", threshold=0.6, clusters=[_ca("A", 1, 11), _ca("B", 2, 22)])
    got = pin(prev, prev_turns, cur_turns)
    assert set(got) == {"Z", "Y"}
    assert got["Z"].actor_id == 1 and got["Z"].character_id == 11
    assert got["Z"].status == "pinned" and got["Z"].cluster == "Z"
    assert got["Y"].actor_id == 2


def test_pin_requires_more_than_half():
    prev_turns = Diarization(backend="x", turns=[Turn(start=0, end=5, speaker="A")])
    cur_turns = Diarization(backend="x", turns=[Turn(start=0, end=10, speaker="Z")])
    prev = Assignments(episode="e", threshold=0.6, clusters=[_ca("A", 1, 11)])
    assert pin(prev, prev_turns, cur_turns) == {}


def test_pin_skips_previous_extra_and_review():
    turns = Diarization(
        backend="x",
        turns=[Turn(start=0, end=10, speaker="A"), Turn(start=10, end=20, speaker="B")],
    )
    cur = Diarization(
        backend="x",
        turns=[Turn(start=0, end=10, speaker="Z"), Turn(start=10, end=20, speaker="Y")],
    )
    extra = ClusterAssignment(
        cluster="A", actor_id=None, character_id=None, score=0.1, clean_seconds=3.0,
        status="extra",
    )  # fmt: skip
    review = ClusterAssignment(
        cluster="B", actor_id=1, character_id=None, score=0.9, clean_seconds=30.0,
        status="review",
    )  # fmt: skip
    prev = Assignments(episode="e", threshold=0.6, clusters=[extra, review])
    assert pin(prev, turns, cur) == {}
