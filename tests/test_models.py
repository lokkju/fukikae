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
