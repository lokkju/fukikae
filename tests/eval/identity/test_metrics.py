import numpy as np
import pytest

from fukikae.eval.groundtruth import GroundTruthLine
from fukikae.eval.identity.metrics import MIN_SPEECH_S, nearest_centroid_accuracy, score
from fukikae.eval.identity.segments import Segment

# Fairy Tail 100YQ S01E12-E13 as auto-labeled and assigned by the pipeline (results doc
# 2026-11-05-registry-fairytail.md): (name, episode, character id, labeled seconds).
FAIRYTAIL = [
    ("BANDIT A", "E13", None, 1.5),
    ("CANA", "E12", 4, 2.9),
    ("CANA", "E12", 5, 24.5),
    ("CANA", "E13", 4, 16.5),
    ("CHARLE", "E12", 2, 12.0),
    ("CHARLE", "E13", None, 2.9),
    ("ERZA", "E12", 3, 1.5),
    ("ERZA", "E12", None, 16.0),
    ("ERZA", "E13", 12, 31.9),
    ("GRAY", "E12", 11, 22.5),
    ("GRAY", "E13", 11, 3.4),
    ("HAPPY", "E12", 2, 6.9),
    ("HAPPY", "E13", 2, 32.5),
    ("HAPPY", "E13", 3, 2.2),
    ("JELLAL", "E12", 3, 2.2),
    ("JELLAL", "E12", 8, 6.6),
    ("JUVIA", "E12", 3, 2.5),
    ("JUVIA", "E12", 4, 12.8),
    ("JUVIA", "E12", 5, 3.1),
    ("JUVIA", "E13", 4, 5.1),
    ("JUVIA", "E13", None, 8.5),
    ("KIRIA", "E12", 6, 26.7),
    ("KIRIA", "E13", None, 8.8),
    ("LAXUS", "E12", 3, 2.0),
    ("LAXUS", "E12", 7, 29.5),
    ("LAXUS", "E13", 3, 4.0),
    ("LAXUS", "E13", 7, 21.1),
    ("LUCY", "E12", 4, 42.0),
    ("LUCY", "E13", 4, 17.1),
    ("LUCY", "E13", None, 11.0),
    ("MAKAROV", "E12", 10, 1.9),
    ("MAKAROV (WRAITH)", "E12", 10, 16.7),
    ("NATSU", "E12", 2, 1.6),
    ("NATSU", "E12", 3, 3.0),
    ("NATSU", "E12", 9, 32.3),
    ("NATSU", "E13", 2, 3.5),
    ("NATSU", "E13", 9, 6.2),
    ("NATSU", "E13", None, 7.2),
    ("NEBARU", "E12", 1, 11.5),
    ("TOUKA", "E12", 4, 5.2),
    ("TOUKA", "E13", 2, 3.1),
    ("TOUKA", "E13", 3, 8.7),
    ("TOUKA", "E13", None, 41.0),
    ("TOWNSPEOPLE A", "E13", None, 2.6),
    ("TOWNSPEOPLE B", "E13", 11, 2.8),
    ("WENDY", "E12", 2, 7.4),
    ("WENDY", "E13", 2, 1.8),
    ("WHITE MAGE", "E13", None, 50.9),
    ("WIZARD B", "E12", 1, 1.7),
    ("WRAITH", "E12", 3, 1.8),
    ("WRAITH", "E12", 8, 35.8),
    ("YOUNG BOB", "E12", 9, 2.1),
    ("YOUNG GOLDMINE", "E12", 11, 3.7),
    ("YOUNG MAKAROV", "E12", 1, 12.9),
    ("YOUNG PORLYUSICA", "E12", None, 2.8),
    ("YOUNG ROB", "E12", 1, 1.8),
    ("YOUNG YAJIMA", "E12", 1, 2.0),
    ("YOUNG YAJIMA", "E12", 10, 6.8),
]


def build(rows, unlabeled=()):
    """Segments laid end to end per episode, one per row, and their assignment."""
    segs: dict[str, list[Segment]] = {}
    assign: dict[tuple[str, int], int | None] = {}
    t: dict[str, float] = {}
    for name, ep, cid, secs in [*rows, *unlabeled]:
        at = t.get(ep, 0.0)
        lst = segs.setdefault(ep, [])
        assign[(ep, len(lst))] = cid
        lst.append(Segment(ep, at, at + secs, name))
        t[ep] = at + secs
    return assign, segs


def test_fairytail_numbers_reproduce():
    assign, segs = build(FAIRYTAIL)
    s = score(assign, segs)
    # The results doc: 12 names labeled in both episodes, 70% of their speech on the modal
    # id, strict 2 of 12 (Gray and Wendy).
    assert s.labeled_recurring == 12
    assert s.weighted_share == pytest.approx(0.70, abs=0.005)
    assert s.strict_fraction == pytest.approx(2 / 12)
    by = {c.name: c for c in s.characters}
    assert by["LAXUS"].share == pytest.approx(0.89, abs=0.005)
    assert by["LAXUS"].modal_id == 7
    assert by["TOUKA"].share == pytest.approx(0.15, abs=0.005)
    assert by["LUCY"].share_by_episode["E13"] == pytest.approx(0.61, abs=0.005)
    assert by["ERZA"].share_by_episode["E12"] == 0.0
    # A name with no id at all has no modal id and nothing on it.
    assert by["WHITE MAGE"].modal_id is None
    assert by["WHITE MAGE"].share == 0.0 and by["WHITE MAGE"].share_by_episode["E13"] == 0.0
    # Purity, the reverse view.
    assert s.purity_by_id[4].dominant == "LUCY"
    assert s.purity_by_id[4].purity == pytest.approx(0.58, abs=0.005)
    assert s.purity_by_id[7].purity == 1.0


def test_fairytail_bar_at_20s_has_two_recurring_characters():
    assign, segs = build(FAIRYTAIL)
    s = score(assign, segs, min_speech=20.0)
    # Only Lucy and Laxus reach 20 s in both episodes; both keep >= 80% on one id.
    assert {c.name for c in s.characters if c.recurring} == {"LAXUS", "LUCY"}
    assert (s.recurring, s.passing, s.merges) == (2, 2, [])
    assert s.verdict == "pass"


def test_passing_and_failing_characters_with_a_lower_bar():
    assign, segs = build(FAIRYTAIL)
    s = score(assign, segs, min_speech=5.0)
    by = {c.name: c for c in s.characters}
    assert by["LAXUS"].recurring and by["LAXUS"].passing
    assert by["TOUKA"].recurring and not by["TOUKA"].passing
    assert s.verdict == "fail"


def test_merges_counted_when_recurring_characters_share_a_modal_id():
    assign, segs = build(FAIRYTAIL)
    s = score(assign, segs, min_speech=1.5)
    # Happy and Wendy both sit on id 2 (and both recur at 1.5 s).
    assert ("HAPPY", "WENDY", 2) in s.merges
    assert s.verdict == "fail"
    rows = [
        ("A", "E1", 1, 30.0),
        ("A", "E2", 1, 30.0),
        ("B", "E1", 1, 30.0),
        ("B", "E2", 1, 30.0),
    ]
    s2 = score(*build(rows))
    assert s2.passing == 2 and s2.fraction == 1.0
    assert s2.merges == [("A", "B", 1)]
    assert s2.verdict == "fail"


def test_insufficient_when_nothing_recurs():
    rows = [("A", "E1", 1, 60.0), ("B", "E2", 2, 60.0), ("C", "E1", 3, 25.0), ("C", "E2", 3, 4.9)]
    s = score(*build(rows))
    assert s.recurring == 0
    assert s.fraction == 0.0
    assert s.verdict == "insufficient"
    # Nothing labeled at all, and a single episode, are insufficient too, never a crash.
    assert score({}, {}).verdict == "insufficient"
    assert score(*build([("A", "E1", 1, 60.0)])).verdict == "insufficient"


def test_pass_needs_the_share_of_passing_characters():
    good = [(n, e, i, 30.0) for i, n in enumerate("ABCDEFGHI", 1) for e in ("E1", "E2")]
    bad = [("J", "E1", 10, 30.0), ("J", "E2", 11, 30.0)]
    s = score(*build(good + bad))
    assert (s.recurring, s.passing) == (10, 9)
    assert s.verdict == "pass"  # 9/10 = 0.9 meets the 90% bar
    worse = bad + [("K", "E1", 12, 30.0), ("K", "E2", 13, 30.0)]
    assert score(*build(good + worse)).verdict == "fail"  # 9/11


def test_unlabeled_and_unassigned_speech():
    rows = [("A", "E1", 1, 24.0), ("A", "E1", None, 6.0), ("A", "E2", 1, 30.0)]
    assign, segs = build(rows, unlabeled=[(None, "E1", 1, 100.0), ("?", "E2", 1, 50.0)])
    s = score(assign, segs)
    a = next(c for c in s.characters if c.name == "A")
    # Unassigned speech counts against the character; `?` and unlabeled speech count nowhere.
    assert a.speech == pytest.approx(60.0)
    assert a.share == pytest.approx(0.9)
    assert s.purity_by_id[1].speech == pytest.approx(54.0)


def test_coverage_against_ground_truth():
    assign, segs = build([("A", "E1", 1, 10.0), ("A", "E1", None, 10.0), ("B", "E2", 2, 10.0)])
    gt = {
        "E1": [GroundTruthLine(0.0, 40.0, "A")],
        "E2": [GroundTruthLine(0.0, 5.0, "B"), GroundTruthLine(3.0, 20.0, "B")],
    }
    s = score(assign, segs, gt=gt)
    # Covered (labeled and assigned): 10 s of E1's 40, 10 s of E2's 20 (the union of its lines).
    assert s.coverage == pytest.approx(20.0 / 60.0)
    # Without ground truth: the share of labeled segment time that got an id.
    assert score(assign, segs).coverage == pytest.approx(20.0 / 30.0)


def test_default_bar_is_5s_and_flips_the_fairytail_verdict():
    """Ruling R6: at 20 s per episode the Fairy Tail baseline passes on two characters; at the
    5 s default eleven recur and it fails."""
    assert MIN_SPEECH_S == 5.0
    assign, segs = build(FAIRYTAIL)
    assert score(assign, segs, min_speech=20.0).verdict == "pass"
    s = score(assign, segs)
    assert s.recurring > 2
    assert s.verdict == "fail"
    assert s.merges  # Lucy and Juvia on id 4, Happy and Charle on id 2


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_nearest_centroid_accuracy():
    train = [
        Segment("E1", 0, 10, "A"),
        Segment("E1", 10, 20, "B"),
        Segment("E1", 20, 25, None),  # unlabeled: no centroid
        Segment("E1", 25, 30, "?"),
    ]
    train_e = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1), unit(0, 0, 1)])
    test = [
        Segment("E2", 0, 6, "A"),  # near A: right
        Segment("E2", 6, 10, "A"),  # near B: wrong
        Segment("E2", 10, 20, "b"),  # case-insensitive label, near B: right
        Segment("E2", 20, 30, "C"),  # no training speech: not scored
        Segment("E2", 30, 40, None),
    ]
    test_e = np.stack(
        [unit(1, 0.1, 0), unit(0.1, 1, 0), unit(0, 1, 0.2), unit(0, 0, 1), unit(1, 0, 0)]
    )
    got = nearest_centroid_accuracy(train, train_e, test, test_e)
    assert got["A"] == pytest.approx(0.6)
    assert got["B"] == pytest.approx(1.0)
    assert "C" not in got
    assert got["(overall)"] == pytest.approx(16 / 20)
    assert nearest_centroid_accuracy([], np.zeros((0, 3)), test, test_e) == {}
