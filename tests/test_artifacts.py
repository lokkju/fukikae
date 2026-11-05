import pytest

from fukikae.artifacts import ArtifactVersionError, atomic_path, read_json, write_json
from fukikae.models import Transcript


def test_read_rejects_newer_schema(tmp_path):
    (tmp_path / "t.json").write_text(
        '{"schema_version": 99, "language": "ko", "backend": "x",'
        ' "word_timing": false, "segments": []}'
    )
    with pytest.raises(ArtifactVersionError):
        read_json(tmp_path / "t.json", Transcript)


def test_read_accepts_equal_version(tmp_path):
    t = Transcript(language="ko", backend="x", word_timing=False, segments=[])
    write_json(tmp_path / "t.json", t)
    assert read_json(tmp_path / "t.json", Transcript).schema_version == 1


def test_atomic_path_leaves_nothing_on_error(tmp_path):
    target = tmp_path / "a.wav"
    with pytest.raises(RuntimeError), atomic_path(target) as p:
        p.write_bytes(b"partial")
        raise RuntimeError
    assert list(tmp_path.iterdir()) == []


def test_atomic_path_renames_on_success(tmp_path):
    target = tmp_path / "a.wav"
    with atomic_path(target) as p:
        assert p.parent == tmp_path and p != target
        p.write_bytes(b"ok")
    assert target.read_bytes() == b"ok"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_path_keeps_suffix_and_directory(tmp_path):
    target = tmp_path / "a.wav"
    with atomic_path(target) as p:
        assert p.suffix == ".wav"
        assert p.parent == tmp_path
        p.write_bytes(b"x")
