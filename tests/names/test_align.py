import re
from typing import Any

import numpy as np
import pytest

from fukikae.models import Turn
from fukikae.names.align import (
    attribute,
    excluded_spans,
    label_for,
    label_infos,
    refine_clusters,
    refined_turns,
)
from fukikae.names.models import NamedLine
from fukikae.names.normalize import NameClass

_ids = iter(range(10_000))


def turn(start: float, end: float, speaker: str) -> Turn:
    return Turn(start=start, end=end, speaker=speaker)


def line(
    start: float,
    end: float,
    key: str | None,
    *,
    cls: NameClass | None = None,
    inner: bool = False,
    id: int | None = None,
) -> NamedLine:
    if cls is None:
        cls = NameClass.NAMED if key is not None else NameClass.IGNORED
    return NamedLine(
        id=next(_ids) if id is None else id,
        start=start,
        end=end,
        raw=(key or "").upper(),
        key=key,
        display=key.title() if key else None,
        cls=cls,
        inner=inner,
        text="words",
    )


def test_turn_covered_by_one_name_takes_it():
    pieces, _ = attribute([turn(10, 14, "SPEAKER_12")], [line(10.1, 13.9, "natsu")])
    assert [(p.label, p.key) for p in pieces] == [("SPEAKER_12~natsu", "natsu")]


def test_turn_under_share_stays_remainder():
    pieces, _ = attribute([turn(10, 20, "SPEAKER_12")], [line(10, 14, "natsu")])  # 40%
    assert pieces[0].label == "SPEAKER_12" and pieces[0].key is None


def test_two_names_in_one_turn_cut_at_gap_midpoint():
    pieces, _ = attribute(
        [turn(1300.3, 1305.9, "SPEAKER_12")],
        [line(1300.4, 1302.8, "happy"), line(1303.2, 1305.8, "natsu")],
    )
    assert [(p.key, round(p.end, 2)) for p in pieces] == [("happy", 1303.0), ("natsu", 1305.9)]
    assert pieces[0].start == 1300.3 and pieces[1].start == pytest.approx(1303.0)


def test_touching_lines_cut_at_their_boundary():
    pieces, _ = attribute([turn(0, 4, "S")], [line(0, 2, "lucy"), line(2, 4, "erza")])
    assert [(p.key, p.start, p.end) for p in pieces] == [("lucy", 0, 2), ("erza", 2, 4)]


def test_sliver_piece_joins_neighbour():
    pieces, _ = attribute([turn(0, 5, "S")], [line(0, 4.8, "lucy"), line(4.8, 5.0, "cana")])
    assert len(pieces) == 1 and pieces[0].key == "lucy"


def test_piece_under_minimum_joins_the_longer_neighbour():
    # Three keys qualify for the cut; erza's 0.6 s piece is under 0.7 s and joins lucy's (longer).
    pieces, _ = attribute(
        [turn(0, 6, "S")],
        [line(0, 3, "lucy"), line(3.0, 3.6, "erza"), line(3.6, 6, "wendy")],
        min_piece_s=0.7,
    )
    assert [(p.key, p.start, p.end) for p in pieces] == [("lucy", 0, 3.6), ("wendy", 3.6, 6)]


def test_multi_line_never_names_and_never_cuts():
    pieces, _ = attribute([turn(0, 4, "S")], [line(0, 4, None, cls=NameClass.MULTI)])
    assert pieces[0].key is None


def test_multi_line_between_two_names_leaves_its_stretch_unnamed():
    pieces, _ = attribute(
        [turn(0, 6, "S")],
        [line(0, 2, "lucy"), line(2, 4, None, cls=NameClass.MULTI), line(4, 6, "natsu")],
    )
    assert [(p.label, p.start, p.end) for p in pieces] == [
        ("S~lucy", 0, 2),
        ("S", 2, 4),
        ("S~natsu", 4, 6),
    ]


def test_overlapping_turns_one_line_names_the_larger_overlap_only():
    pieces, _ = attribute([turn(0, 4, "A"), turn(3, 5, "B")], [line(0.2, 3.9, "lucy")])
    assert {p.cluster: p.key for p in pieces} == {"A": "lucy", "B": None}


def test_unvoiced_line_reported():
    _, unvoiced = attribute(
        [turn(0, 2, "A")], [line(0.1, 1.9, "lucy", id=0), line(50, 52, "erza", id=1)]
    )
    assert unvoiced == [1]


def test_unnamed_line_is_never_unvoiced():
    _, unvoiced = attribute([turn(0, 2, "A")], [line(50, 52, None, id=7)])
    assert unvoiced == []


def test_inner_voice_line_names_its_key():
    pieces, _ = attribute([turn(0, 3, "A")], [line(0, 3, "wendy", inner=True)])
    assert pieces[0].key == "wendy"


def test_role_label_names_its_turn():
    pieces, _ = attribute([turn(0, 3, "A")], [line(0, 3, "wizard a", cls=NameClass.ROLE)])
    assert pieces[0].label == "A~wizard-a"


def test_hidden_lines_act_unnamed_and_stay_out_of_line_ids():
    pieces, _ = attribute([turn(0, 3, "A")], [line(0, 3, "wendy", id=5)], hidden={5})
    assert pieces[0].key is None and pieces[0].line_ids == []


def test_pieces_cover_every_turn_in_order_with_their_line_ids():
    turns = [turn(0, 2, "A"), turn(3, 6, "B"), turn(10, 11, "A")]
    lines = [line(0, 2, "lucy", id=0), line(3, 4.4, "natsu", id=1), line(4.6, 6, "happy", id=2)]
    pieces, _ = attribute(turns, lines)
    assert [(p.cluster, p.start, p.end, p.line_ids) for p in pieces] == [
        ("A", 0, 2, [0]),
        ("B", 3, 4.5, [1]),
        ("B", 4.5, 6, [2]),
        ("A", 10, 11, []),
    ]


def test_non_ascii_key_gets_hashed_slug():
    assert re.fullmatch(r"SPEAKER_01~n[0-9a-f]{8}", label_for("SPEAKER_01", "なつ"))


def test_ascii_slug():
    assert label_for("SPEAKER_01", "makarov (wraith)") == "SPEAKER_01~makarov-wraith"
    assert label_for("S", "young makarov") == "S~young-makarov"


def test_keys_sharing_a_slug_get_hashed_labels():
    pieces, _ = attribute(
        [turn(0, 2, "A"), turn(3, 5, "A")],
        [line(0, 2, "makarov (wraith)"), line(3, 5, "makarov wraith")],
    )
    labels = [p.label for p in pieces]
    assert len(set(labels)) == 2
    assert all(re.fullmatch(r"A~n[0-9a-f]{8}", lb) for lb in labels)


def test_label_infos_describe_named_role_and_remainder():
    lines = [line(0, 2, "lucy"), line(3, 5, "wizard a", cls=NameClass.ROLE)]
    pieces, _ = attribute([turn(0, 2, "A"), turn(3, 5, "A"), turn(8, 9, "A")], lines)
    infos = label_infos(pieces, lines)
    assert infos["A~lucy"].model_dump() == {
        "parent": "A",
        "key": "lucy",
        "display": "Lucy",
        "cls": "named",
        "slug_clash": False,
    }
    assert infos["A~wizard-a"].cls == "role"
    assert infos["A"].cls == "remainder" and infos["A"].key is None


def test_excluded_spans_are_inner_and_multi_lines():
    lines = [
        line(0, 1, "lucy"),
        line(2, 3, "wendy", inner=True),
        line(4, 5, None, cls=NameClass.MULTI),
    ]
    assert excluded_spans(lines) == [(2, 3), (4, 5)]


def _npz(path, arrays: dict[str, list[tuple[float, float]]], dim: int = 4):
    labels = list(arrays)
    out: dict[str, Any] = {"labels": np.array(labels, dtype=str)}
    rng = np.random.default_rng(0)
    for label, spans in arrays.items():
        n = len(spans)
        out[f"{label}/embeddings"] = rng.standard_normal((n, dim)).astype(np.float32)
        out[f"{label}/spans"] = np.array(spans, dtype=np.float64).reshape(n, 2)
        out[f"{label}/clean_seconds"] = np.array(sum(e - s for s, e in spans))
    np.savez(path, **out)
    return path


def test_rows_move_with_their_turn_and_cut_rows_drop(tmp_path):
    turns = [
        turn(10, 14, "SPEAKER_12"),
        turn(1300.3, 1305.9, "SPEAKER_12"),
        turn(20, 23, "SPEAKER_01"),
    ]
    lines = [
        line(10.1, 13.9, "natsu"),
        line(1300.4, 1302.8, "happy"),
        line(1303.2, 1305.8, "natsu"),
    ]
    npz = _npz(
        tmp_path / "in.npz",
        {"SPEAKER_12": [(1300.4, 1305.8), (10.1, 13.9)], "SPEAKER_01": [(20.1, 22.9)]},
    )
    pieces, _ = attribute(turns, lines)
    refined = refined_turns(pieces)
    dropped = refine_clusters(npz, pieces, refined, tmp_path / "out.npz")
    with np.load(npz) as src, np.load(tmp_path / "out.npz") as z:
        labels = [str(x) for x in z["labels"]]
        assert labels == ["SPEAKER_12~natsu", "SPEAKER_12~happy", "SPEAKER_01"]
        assert z["SPEAKER_12~natsu/embeddings"].shape == (1, 4)
        np.testing.assert_array_equal(
            z["SPEAKER_12~natsu/embeddings"][0], src["SPEAKER_12/embeddings"][1]
        )
        assert z["SPEAKER_12~natsu/embeddings"].dtype == np.float32
        assert z["SPEAKER_12~happy/embeddings"].shape == (0, 4)
        assert z["SPEAKER_12~happy/spans"].shape == (0, 2)
        np.testing.assert_array_equal(z["SPEAKER_01/spans"], src["SPEAKER_01/spans"])
        # clean seconds over the refined turns: natsu 10.1-13.9 and 1303.1-1305.8
        assert float(z["SPEAKER_12~natsu/clean_seconds"]) == pytest.approx(3.8 + 2.7)
        assert float(z["SPEAKER_12~happy/clean_seconds"]) == pytest.approx(2.5)
    assert dropped == pytest.approx(5.4, abs=0.01)


def test_refined_turns_take_piece_labels():
    pieces, _ = attribute([turn(0, 4, "S")], [line(0, 2, "lucy"), line(2, 4, "erza")])
    assert [(t.start, t.end, t.speaker) for t in refined_turns(pieces)] == [
        (0, 2, "S~lucy"),
        (2, 4, "S~erza"),
    ]


# ---- review fixes (Task 3 review) ----


def _own_key_overlap(pieces, lines) -> None:
    """Every keyed piece overlaps a line of its own key."""
    for p in pieces:
        if p.key is None:
            continue
        own = [ln for ln in lines if ln.key == p.key]
        assert any(min(p.end, ln.end) > max(p.start, ln.start) for ln in own), p


def _contiguous(pieces, t) -> None:
    assert pieces[0].start == t.start and pieces[-1].end == t.end
    for a, b in zip(pieces, pieces[1:], strict=False):
        assert a.end == b.start and a.start <= a.end


def test_line_nested_in_another_key_leaves_its_stretch_unnamed():
    t = turn(0, 14, "S")
    lines = [line(0, 10, "lucy"), line(3, 4, "natsu"), line(11, 13.5, "lucy")]
    pieces, _ = attribute([t], lines)
    _own_key_overlap(pieces, lines)
    _contiguous(pieces, t)
    assert [(p.key, p.start, p.end) for p in pieces] == [
        ("lucy", 0, 3),
        (None, 3, 4),
        ("lucy", 4, 14),
    ]


def test_nested_lines_never_hand_another_key_the_outer_line():
    t = turn(0, 12, "S")
    lines = [line(0, 10, "lucy"), line(3, 4, "natsu"), line(4.5, 6, "erza"), line(10.5, 12, "erza")]
    pieces, _ = attribute([t], lines)
    _own_key_overlap(pieces, lines)
    _contiguous(pieces, t)
    erza = [p for p in pieces if p.key == "erza"]
    assert erza and all(p.start >= 10 for p in erza)
    assert not [p for p in pieces if p.key == "natsu"]


def test_partly_overlapping_lines_of_two_keys_leave_the_overlap_unnamed():
    t = turn(0, 6, "S")
    lines = [line(0, 3, "lucy"), line(2.5, 5, "natsu")]
    pieces, _ = attribute([t], lines)
    assert [(p.key, p.start, p.end) for p in pieces] == [
        ("lucy", 0, 2.5),
        (None, 2.5, 3),
        ("natsu", 3, 6),
    ]


def test_overlapping_lines_of_one_key_count_once():
    pieces, _ = attribute(
        [turn(0, 10, "S")], [line(0, 3, "lucy"), line(0, 3, "lucy"), line(0.5, 3, "lucy")]
    )
    assert pieces[0].key is None  # 3 s of 10, not 8.5


def test_hidden_key_is_never_a_side_of_a_cut():
    pieces, _ = attribute(
        [turn(0, 4, "S")], [line(0, 2.6, "lucy"), line(2.6, 4, "erza", id=900)], hidden={900}
    )
    assert [(p.key, p.start, p.end, p.line_ids) for p in pieces] == [
        ("lucy", 0, 4, pieces[0].line_ids)
    ]
    assert 900 not in pieces[0].line_ids


def test_cut_wins_over_a_single_key_share():
    # lucy alone covers 80% of the turn, but natsu's 0.6 s makes it a two-name turn.
    pieces, _ = attribute([turn(0, 5, "S")], [line(0, 4.0, "lucy"), line(4.4, 5.0, "natsu")])
    assert [(p.key, p.start, p.end) for p in pieces] == [("lucy", 0, 4.2), ("natsu", 4.2, 5)]


def test_equal_cover_is_broken_by_key_order():
    pieces, _ = attribute(
        [turn(0, 0.8, "S")], [line(0, 0.4, "lucy"), line(0.4, 0.8, "natsu")], turn_share=0.4
    )
    assert pieces[0].key == "natsu"


def test_pieces_conserve_each_turns_time():
    cases = [
        (turn(0, 14, "A"), [line(0, 10, "lucy"), line(3, 4, "natsu"), line(11, 13.5, "lucy")]),
        (
            turn(20, 32, "B"),
            [
                line(20, 30, "lucy"),
                line(23, 24, "natsu"),
                line(24.5, 26, "erza"),
                line(30.5, 32, "erza"),
            ],
        ),
        (
            turn(40, 46, "C"),
            [
                line(40, 43, "lucy"),
                line(42.5, 43.5, None, cls=NameClass.MULTI),
                line(43.5, 46, "natsu"),
            ],
        ),
        (turn(50, 55, "D"), [line(50, 54.8, "lucy"), line(54.8, 55, "cana")]),
        (turn(60, 64, "E"), [line(60, 62, "lucy"), line(60, 63, "natsu")]),
    ]
    for t, lines in cases:
        pieces, _ = attribute([t], lines)
        assert sum(p.end - p.start for p in pieces) == pytest.approx(t.end - t.start)
        _contiguous(pieces, t)
        _own_key_overlap(pieces, lines)


def test_slug_collisions_are_judged_per_cluster():
    lines = [line(0, 2, "makarov (wraith)"), line(3, 5, "makarov wraith"), line(8, 9, "lucy")]
    pieces, _ = attribute([turn(0, 2, "A"), turn(3, 5, "B"), turn(8, 9, "A")], lines)
    assert [p.label for p in pieces] == ["A~makarov-wraith", "B~makarov-wraith", "A~lucy"]
    # a colliding key in the same cluster hashes both; label_infos records why
    lines2 = [line(0, 2, "makarov (wraith)"), line(3, 5, "makarov wraith")]
    pieces2, _ = attribute([turn(0, 2, "A"), turn(3, 5, "A")], lines2)
    infos = label_infos(pieces2, lines2)
    assert all(info.slug_clash for info in infos.values())


def test_slug_strips_accents():
    assert label_for("S", "chloé") == "S~chloe"
    assert re.fullmatch(r"S~n[0-9a-f]{8}", label_for("S", "ガジル"))


def _cut_turn_npz(tmp_path, rows):
    t = turn(0, 10, "S")
    lines = [line(0, 4.8, "lucy"), line(5.2, 10, "natsu")]
    pieces, _ = attribute([t], lines)
    assert [(p.key, p.end) for p in pieces] == [("lucy", 5.0), ("natsu", 10)]
    npz = _npz(tmp_path / "in.npz", {"S": rows})
    return pieces, npz


def test_row_inside_a_cut_piece_by_ninety_percent_moves_and_just_under_drops(tmp_path):
    pieces, npz = _cut_turn_npz(tmp_path, [(0.1, 5.5), (0.1, 5.6)])  # 4.9/5.4 and 4.9/5.5
    dropped = refine_clusters(npz, pieces, refined_turns(pieces), tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz") as z:
        assert z["S~lucy/spans"].tolist() == [[0.1, 5.5]]
    assert dropped == pytest.approx(5.5)


def test_label_rows_stay_longest_first(tmp_path):
    turns = [turn(0, 2, "A"), turn(10, 15, "A")]
    lines = [line(0, 2, "lucy"), line(10, 15, "lucy")]
    pieces, _ = attribute(turns, lines)
    npz = _npz(tmp_path / "in.npz", {"A": [(0.1, 1.9), (10.1, 14.9)]})
    refine_clusters(npz, pieces, refined_turns(pieces), tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz") as z:
        assert z["A~lucy/spans"].tolist() == [[10.1, 14.9], [0.1, 1.9]]


def test_rows_move_only_within_their_own_cluster(tmp_path):
    turns = [turn(0, 10, "A"), turn(2, 6, "B")]
    lines = [line(0, 10, "lucy")]
    pieces, _ = attribute(turns, lines)
    npz = _npz(tmp_path / "in.npz", {"A": [(6.1, 9.9)], "B": [(2.1, 5.9)]})
    refine_clusters(npz, pieces, refined_turns(pieces), tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz") as z:
        assert z["A~lucy/spans"].tolist() == [[6.1, 9.9]]
        assert z["B/spans"].tolist() == [[2.1, 5.9]]


def test_rows_must_move_to_a_label_of_turns_out(tmp_path):
    pieces, _ = attribute([turn(0, 4, "A")], [line(0, 4, "lucy")])
    npz = _npz(tmp_path / "in.npz", {"A": [(0.1, 3.9)]})
    with pytest.raises(ValueError, match="A~lucy"):
        refine_clusters(npz, pieces, [turn(0, 4, "A")], tmp_path / "out.npz")


def test_refined_npz_is_byte_identical_on_rewrite(tmp_path, monkeypatch):
    import time

    pieces, npz = _cut_turn_npz(tmp_path, [(0.1, 4.7), (5.3, 9.9)])
    refine_clusters(npz, pieces, refined_turns(pieces), tmp_path / "a.npz")
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 7200)
    refine_clusters(npz, pieces, refined_turns(pieces), tmp_path / "b.npz")
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()


# -- cut edge cases pinned from the Task 4 review's probe (/tmp/rev-t4/cutprobe.py) ----------


def _spans(turns: list[Turn], lines: list[NamedLine]) -> list[tuple[float, float, str | None]]:
    """The pieces as (start, end, key), after checking the invariants every cut keeps: each
    turn is covered end to end with no gap or overlap, no piece is empty, and a keyed piece
    overlaps a line of its own key."""
    pieces, _ = attribute(turns, lines)
    for i, t in enumerate(turns):
        ps = [p for p in pieces if p.turn == i]
        assert ps[0].start == pytest.approx(t.start) and ps[-1].end == pytest.approx(t.end)
        assert all(a.end == pytest.approx(b.start) for a, b in zip(ps, ps[1:], strict=False))
        assert all(p.end > p.start for p in ps)
        for p in ps:
            if p.key is not None:
                assert any(
                    ln.key == p.key
                    and ln.cls is not NameClass.MULTI
                    and min(ln.end, p.end) > max(ln.start, p.start)
                    for ln in lines
                )
    return [(round(p.start, 3), round(p.end, 3), p.key) for p in pieces]


def _multi(start: float, end: float) -> NamedLine:
    return line(start, end, None, cls=NameClass.MULTI)


def test_cut_with_a_gap_at_the_start_gives_it_to_the_first_side():
    lines = [line(3, 6, "a"), line(6.5, 10, "b")]
    assert _spans([turn(0, 10, "S")], lines) == [(0, 6.25, "a"), (6.25, 10, "b")]


def test_gap_at_the_end_under_turn_share_names_nothing():
    lines = [line(0, 3, "a"), line(3.5, 6, "b")]  # 5.5 s of 10: under 0.6
    assert _spans([turn(0, 10, "S")], lines) == [(0, 10, None)]


def test_three_way_overlap_keeps_only_the_uncontested_ends():
    lines = [line(0, 6, "a"), line(2, 8, "b"), line(4, 10, "c")]
    assert _spans([turn(0, 10, "S")], lines) == [(0, 2, "a"), (2, 8, None), (8, 10, "c")]


def test_three_way_stagger_unnames_each_overlap():
    lines = [line(0, 4, "a"), line(3, 7, "b"), line(6, 10, "c")]
    assert _spans([turn(0, 10, "S")], lines) == [
        (0, 3, "a"),
        (3, 4, None),
        (4, 6, "b"),
        (6, 7, None),
        (7, 10, "c"),
    ]


def test_whole_turn_span_with_nested_lines_unnames_the_nested_stretches():
    lines = [
        line(0, 20, "a"),
        line(2, 3, "b"),
        line(5, 7, "c"),
        line(9, 10, "b"),
        line(12, 15, "d"),
        line(17, 18, "c"),
    ]
    assert _spans([turn(0, 20, "S")], lines) == [
        (0, 2, "a"),
        (2, 3, None),
        (3, 5, "a"),
        (5, 7, None),
        (7, 9, "a"),
        (9, 10, None),
        (10, 12, "a"),
        (12, 15, None),
        (15, 17, "a"),
        (17, 18, None),
        (18, 20, "a"),
    ]


def test_all_multi_lines_name_nothing():
    assert _spans([turn(0, 10, "S")], [_multi(0, 5), _multi(5, 10)]) == [(0, 10, None)]


def test_multi_line_over_two_keys_names_nothing():
    lines = [_multi(0, 10), line(0, 5, "a"), line(5, 10, "b")]
    assert _spans([turn(0, 10, "S")], lines) == [(0, 10, None)]
