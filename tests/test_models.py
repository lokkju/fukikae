from pathlib import Path

from fukikae.artifacts import read_json, write_json
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Diarization,
    Segment,
    Separation,
    Transcript,
    Turn,
    Word,
)


def test_transcript_roundtrip(tmp_path):
    t = Transcript(
        language="ko",
        backend="x",
        word_timing=True,
        segments=[
            Segment(start=0.0, end=1.5, text="안녕", words=[Word(start=0.0, end=1.5, text="안녕")])
        ],
    )
    write_json(tmp_path / "t.json", t)
    assert read_json(tmp_path / "t.json", Transcript) == t


def test_diarization_and_assignments_roundtrip(tmp_path):
    d = Diarization(backend="p", turns=[Turn(start=0.0, end=2.0, speaker="S0")])
    write_json(tmp_path / "d.json", d)
    assert read_json(tmp_path / "d.json", Diarization) == d
    a = Assignments(
        episode="e1",
        threshold=0.5,
        clusters=[
            ClusterAssignment(
                cluster="S0",
                actor_id=None,
                character_id=None,
                score=None,
                clean_seconds=21.0,
                status="new",
            )
        ],
    )
    write_json(tmp_path / "a.json", a)
    assert read_json(tmp_path / "a.json", Assignments) == a


def test_separation_paths_serialize_as_strings():
    s = Separation(vocals=Path("a.wav"), background=Path("b.wav"), vocals_16k=Path("c.wav"))
    assert s.model_dump(mode="json")["vocals"] == "a.wav"


def test_tags_round_trip():
    from fukikae.models import Tags, TagSpan

    t = Tags(
        backend="sensevoice",
        events=[TagSpan(start=0.0, end=7.5, label="bgm")],
        emotions=[TagSpan(start=1.0, end=2.0, label="happy", confidence=None)],
    )
    assert Tags.model_validate_json(t.model_dump_json()) == t
    assert t.schema_version == 1


def test_cluster_assignment_defaults_are_omitted_from_json():
    plain = ClusterAssignment(
        cluster="S0", actor_id=1, character_id=1, score=0.9, clean_seconds=3.0, status="matched"
    )
    assert "name" not in plain.model_dump_json() and "source" not in plain.model_dump_json()
    named = plain.model_copy(update={"name": "natsu", "source": "name"})
    raw = named.model_dump_json()
    assert '"name":"natsu"' in raw and '"source":"name"' in raw
    assert ClusterAssignment.model_validate_json(raw) == named
    assert ClusterAssignment.model_validate_json(plain.model_dump_json()) == plain
