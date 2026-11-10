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


def test_hidden_lines_act_unnamed_but_stay_in_line_ids():
    pieces, _ = attribute([turn(0, 3, "A")], [line(0, 3, "wendy", id=5)], hidden={5})
    assert pieces[0].key is None and pieces[0].line_ids == [5]


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
