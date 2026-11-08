import pytest

from fukikae.media import (
    AudioTrack,
    MediaError,
    ProbeResult,
    SubTrack,
    choose_audio_track,
    extract_subtitle,
    find_target_subtitles,
    probe,
    run_ffmpeg,
)


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


def test_choose_track_by_title():
    p = ProbeResult(
        audio=[
            AudioTrack(1, "aac", 2, "jpn", None, False),
            AudioTrack(2, "aac", 2, "spa", "Spanish (Latin America)", False),
            AudioTrack(3, "aac", 2, "spa", "Spanish (Castilian)", False),
        ],
        subtitles=[],
    )
    assert choose_audio_track(p, "es", None, title="castilian").index == 3
    assert choose_audio_track(p, "es", None).index == 2
    assert choose_audio_track(p, "es", 2, title="castilian").index == 2


def test_choose_track_title_without_match_raises():
    p = ProbeResult(
        audio=[
            AudioTrack(1, "aac", 2, "spa", "Spanish (Latin America)", False),
            AudioTrack(2, "aac", 2, "spa", "Spanish (Castilian)", False),
        ],
        subtitles=[],
    )
    with pytest.raises(MediaError, match="absent"):
        choose_audio_track(p, "es", None, title="absent")
    # a title naming a track of another language does not count
    jp = ProbeResult(audio=[AudioTrack(1, "aac", 2, "jpn", "Castilian", False)], subtitles=[])
    with pytest.raises(MediaError):
        choose_audio_track(jp, "es", None, title="castilian")
    assert choose_audio_track(p, "es", 2, title="castilian").index == 2


@pytest.mark.ffmpeg
def test_probe_real(readonly_copy):
    result = probe(readonly_copy)
    assert [(t.channels, t.language) for t in result.audio] == [(2, "eng"), (6, "jpn")]
    assert result.audio[0].default


@pytest.mark.ffmpeg
def test_run_ffmpeg_error_has_stderr_tail(tmp_path):
    with pytest.raises(MediaError, match="nope"):
        run_ffmpeg(["-i", str(tmp_path / "nope.mkv"), str(tmp_path / "o.wav")])


def _subs_probe() -> ProbeResult:
    return ProbeResult(
        audio=[],
        subtitles=[
            SubTrack(3, "ass", "eng", "Forced", True),
            SubTrack(4, "hdmv_pgs_subtitle", "eng", None, False),
            SubTrack(5, "ass", "eng", None, False),
            SubTrack(6, "ass", "ger", None, False),
            SubTrack(7, "subrip", "eng", "Signs & Songs", False),
        ],
    )


def test_find_target_subtitles_order(tmp_path):
    src = tmp_path / "show S01E12.mkv"
    # first non-forced text track of the language
    assert find_target_subtitles(src, _subs_probe(), "en", {}) == (None, _subs_probe().subtitles[2])
    assert find_target_subtitles(src, _subs_probe(), "de", {}) == (None, _subs_probe().subtitles[3])
    assert find_target_subtitles(src, _subs_probe(), "fr", {}) == (None, None)
    # a sidecar beats an embedded track; either language code
    side = tmp_path / "show S01E12.eng.ass"
    side.write_text("x", encoding="utf-8")
    assert find_target_subtitles(src, _subs_probe(), "en", {}) == (side, None)
    # a configured track beats the sidecar, a configured file beats both
    got = find_target_subtitles(src, _subs_probe(), "en", {"track": 7})
    assert got == (None, _subs_probe().subtitles[4])
    f = tmp_path / "mine.srt"
    f.write_text("x", encoding="utf-8")
    assert find_target_subtitles(src, _subs_probe(), "en", {"file": str(f), "track": 7}) == (
        f,
        None,
    )


def test_find_target_subtitles_refuses_bad_configuration(tmp_path):
    src = tmp_path / "e.mkv"
    with pytest.raises(MediaError, match="not found"):
        find_target_subtitles(src, _subs_probe(), "en", {"file": str(tmp_path / "nope.srt")})
    with pytest.raises(MediaError, match="subtitle track 9"):
        find_target_subtitles(src, _subs_probe(), "en", {"track": 9})
    with pytest.raises(MediaError, match="image"):
        find_target_subtitles(src, _subs_probe(), "en", {"track": 4})


@pytest.mark.ffmpeg
def test_extract_subtitle_round_trip(tmp_path):
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello.\n", encoding="utf-8")
    mkv = tmp_path / "in.mkv"
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=d=3",
            "-i",
            str(srt),
            "-map",
            "0",
            "-map",
            "1",
            "-c:a",
            "aac",
            "-c:s",
            "ass",
            "-metadata:s:s:0",
            "language=eng",
            str(mkv),
        ]
    )
    p = probe(mkv)
    (track,) = p.subtitles
    out = extract_subtitle(mkv, track, tmp_path / "out" / "subs.en")
    assert out.suffix == ".ass" and "Hello." in out.read_text(encoding="utf-8")
