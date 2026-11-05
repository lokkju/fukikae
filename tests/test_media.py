import pytest

from fukikae.media import AudioTrack, MediaError, ProbeResult, choose_audio_track, probe, run_ffmpeg


def _probe():
    return ProbeResult(
        audio=[
            AudioTrack(1, "aac", 2, "eng", None, True),
            AudioTrack(2, "aac", 6, "jpn", None, False),
        ],
        subtitles=[],
    )


def test_choose_track_prefers_source_language():
    assert choose_audio_track(_probe(), "ja", None).index == 2
    assert choose_audio_track(_probe(), "jpn", None).index == 2
    assert choose_audio_track(_probe(), None, None).index == 2  # first non-English


def test_choose_track_override_and_fallback():
    assert choose_audio_track(_probe(), "ja", 1).index == 1
    only_eng = ProbeResult(audio=[AudioTrack(0, "aac", 2, "eng", None, True)], subtitles=[])
    assert choose_audio_track(only_eng, None, None).index == 0
    with pytest.raises(MediaError):
        choose_audio_track(only_eng, None, 9)


@pytest.mark.ffmpeg
def test_probe_real(readonly_copy):
    result = probe(readonly_copy)
    assert [(t.channels, t.language) for t in result.audio] == [(2, "eng"), (6, "jpn")]
    assert result.audio[0].default


@pytest.mark.ffmpeg
def test_run_ffmpeg_error_has_stderr_tail(tmp_path):
    with pytest.raises(MediaError, match="nope"):
        run_ffmpeg(["-i", str(tmp_path / "nope.mkv"), str(tmp_path / "o.wav")])
