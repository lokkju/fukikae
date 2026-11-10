import pytest

from fukikae.media import (
    AudioTrack,
    MediaError,
    ProbeResult,
    SubTrack,
    choose_audio_track,
    extract_subtitle,
    extract_subtitles,
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
    assert out.name == "subs.en.ass" and "Hello." in out.read_text(encoding="utf-8")
    assert extract_subtitle(mkv, track, tmp_path / "x.srt").name == "x.ass"


@pytest.mark.ffmpeg
def test_extract_subtitles_reads_several_tracks_in_one_pass(tmp_path):
    srts = []
    for i, text in enumerate(("Hello.", "Hallo.")):
        srt = tmp_path / f"in{i}.srt"
        srt.write_text(f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n", encoding="utf-8")
        srts.append(srt)
    mkv = tmp_path / "in.mkv"
    run_ffmpeg(
        ["-f", "lavfi", "-i", "anullsrc=d=3", "-i", str(srts[0]), "-i", str(srts[1]),
         "-map", "0", "-map", "1", "-map", "2", "-c:a", "aac", "-c:s", "ass", str(mkv)]
    )  # fmt: skip
    a, b = probe(mkv).subtitles
    outs = extract_subtitles(mkv, [(a, tmp_path / "a"), (b, tmp_path / "o" / "b.srt")])
    assert [o.name for o in outs] == ["a.ass", "b.ass"]
    assert "Hello." in outs[0].read_text(encoding="utf-8")
    assert "Hallo." in outs[1].read_text(encoding="utf-8")
    assert extract_subtitles(mkv, []) == []


# -- ruling CD25: the dialogue track, never forced or signs-and-songs-only ------------------


def _cd25(*tracks: SubTrack, duration: float | None = 1440.0) -> ProbeResult:
    return ProbeResult(audio=[], subtitles=list(tracks), duration=duration)


def test_cd25_skips_forced_and_signs_tracks_for_the_dialogue_track(tmp_path):
    src = tmp_path / "show S01E12.mkv"
    full = SubTrack(4, "ass", "eng", None, False, cues=357)
    probe_ = _cd25(
        SubTrack(3, "ass", "eng", "Forced", True, cues=3),
        SubTrack(2, "ass", "eng", "Signs & Songs", False, cues=40),
        full,
    )
    assert find_target_subtitles(src, probe_, "en", {}) == (None, full)


def test_cd25_a_sparse_track_is_no_dialogue_track(tmp_path):
    src = tmp_path / "show S01E12.mkv"
    sparse = SubTrack(5, "ass", "eng", None, False, cues=12)  # 0.5 cues a minute
    assert find_target_subtitles(src, _cd25(sparse), "en", {}) == (None, None)
    full = SubTrack(6, "ass", "eng", None, False, cues=300)
    assert find_target_subtitles(src, _cd25(sparse, full), "en", {}) == (None, full)
    # far below the densest track of the language, though above the absolute floor
    thin = SubTrack(7, "ass", "eng", None, False, cues=80)
    assert find_target_subtitles(src, _cd25(thin, full), "en", {}) == (None, full)


def test_cd25_every_line_beats_a_forced_flag(tmp_path):
    """FT E11: the only English track is flagged forced but titled English and carries every
    line (344 cues in 24 minutes): it is the dialogue track."""
    src = tmp_path / "show S01E11.mkv"
    e11 = SubTrack(2, "ass", "eng", "English", True, cues=344)
    assert find_target_subtitles(src, _cd25(e11, duration=1428.0), "en", {}) == (None, e11)
    # a title that says forced is believed; so is a flag on a track of unknown size
    named = SubTrack(2, "ass", "eng", "Forced", True, cues=344)
    assert find_target_subtitles(src, _cd25(named), "en", {}) == (None, None)
    unknown = SubTrack(2, "ass", "eng", "English", True)
    assert find_target_subtitles(src, _cd25(unknown), "en", {}) == (None, None)


def test_cd25_prefers_the_plain_track_over_cc_and_a_flagged_one(tmp_path):
    src = tmp_path / "show S01E13.mkv"
    cc = SubTrack(5, "subrip", "eng", "CC", False, cues=508)
    sdh = SubTrack(6, "subrip", "eng", None, False, cues=500, hearing_impaired=True)
    flagged = SubTrack(7, "ass", "eng", "English", True, cues=260)
    plain = SubTrack(4, "ass", "eng", None, False, cues=254)
    assert find_target_subtitles(src, _cd25(cc, sdh, flagged, plain), "en", {}) == (None, plain)
    assert find_target_subtitles(src, _cd25(cc, flagged), "en", {}) == (None, flagged)
    assert find_target_subtitles(src, _cd25(cc), "en", {}) == (None, cc)


def test_cd25_unknown_counts_keep_the_old_order(tmp_path):
    """Without cue counts (no NUMBER_OF_FRAMES tag) or a duration, density can't be judged:
    the first unflagged, untitled-forced text track of the language, as before."""
    src = tmp_path / "show S01E12.mkv"
    assert find_target_subtitles(src, _subs_probe(), "en", {}) == (None, _subs_probe().subtitles[2])
    t = SubTrack(4, "ass", "eng", None, False, cues=10)
    assert find_target_subtitles(src, _cd25(t, duration=None), "en", {}) == (None, t)


def test_probe_reads_cue_counts_duration_and_hearing_impaired():
    from fukikae.media import parse_probe

    doc = {
        "format": {"duration": "1428.5"},
        "streams": [
            {"index": 2, "codec_type": "subtitle", "codec_name": "ass",
             "tags": {"language": "eng", "title": "English", "NUMBER_OF_FRAMES": "344"},
             "disposition": {"forced": 1, "hearing_impaired": 0}},
            {"index": 3, "codec_type": "subtitle", "codec_name": "subrip",
             "tags": {"language": "eng", "NUMBER_OF_FRAMES-eng": "508"},
             "disposition": {"forced": 0, "hearing_impaired": 1}},
            {"index": 4, "codec_type": "subtitle", "codec_name": "ass", "tags": {}},
        ],
    }  # fmt: skip
    p = parse_probe(doc)
    assert p.duration == 1428.5
    assert p.subtitles[0] == SubTrack(2, "ass", "eng", "English", True, cues=344)
    assert p.subtitles[1].cues == 508 and p.subtitles[1].hearing_impaired
    assert p.subtitles[2].cues is None


def test_cd25_a_dense_typeset_signs_track_cannot_knock_out_the_dialogue(tmp_path):
    """Fix round 1: a forced typeset signs track with 1200 events (karaoke and signs styled
    line by line) is no measure of dialogue density."""
    src = tmp_path / "show S01E12.mkv"
    signs = SubTrack(3, "ass", "eng", None, True, cues=1200)
    named = SubTrack(5, "ass", "eng", "Signs", False, cues=1200)
    dialogue = SubTrack(4, "ass", "eng", None, False, cues=350)
    assert find_target_subtitles(src, _cd25(signs, dialogue), "en", {}) == (None, dialogue)
    assert find_target_subtitles(src, _cd25(named, dialogue), "en", {}) == (None, dialogue)


def test_cd25_a_lone_forced_signs_track_needs_dialogue_density(tmp_path):
    """A forced-flagged track counts as dialogue only from 8 cues a minute (E11: 14.4)."""
    src = tmp_path / "show S01E12.mkv"
    for cues in (72, 100, 144):  # 3, 4.2 and 6 a minute over 24 minutes
        lone = SubTrack(3, "ass", "eng", None, True, cues=cues)
        assert find_target_subtitles(src, _cd25(lone), "en", {}) == (None, None)
    e11 = SubTrack(2, "ass", "eng", "English", True, cues=344)
    assert find_target_subtitles(src, _cd25(e11, duration=1428.0), "en", {}) == (None, e11)


def test_cd25_full_subtitles_with_songs_are_dialogue(tmp_path):
    src = tmp_path / "show S01E12.mkv"
    for title in ("Full Subtitles (incl. Songs)", "Dialogue + Songs", "Complete (Songs)"):
        full = SubTrack(4, "ass", "eng", title, False, cues=350)
        assert find_target_subtitles(src, _cd25(full), "en", {}) == (None, full), title
        thin = SubTrack(4, "ass", "eng", title, False, cues=40)
        assert find_target_subtitles(src, _cd25(thin), "en", {}) == (None, None), title
        unknown = SubTrack(4, "ass", "eng", title, False)
        assert find_target_subtitles(src, _cd25(unknown), "en", {}) == (None, None), title
    songs = SubTrack(4, "ass", "eng", "Songs", False, cues=350)
    assert find_target_subtitles(src, _cd25(songs), "en", {}) == (None, None)


def test_cd25_an_untitled_dense_typeset_track_neither_sets_the_bar_nor_wins(tmp_path):
    """Fix round 2: an untitled, unflagged typeset track with 1200 events (50 a minute) is
    capped at 30 a minute for the bar, and ranks after a plausible dialogue track."""
    src = tmp_path / "show S01E12.mkv"
    typeset = SubTrack(3, "ass", "eng", None, False, cues=1200)
    dialogue = SubTrack(4, "ass", "eng", None, False, cues=350)
    assert find_target_subtitles(src, _cd25(typeset, dialogue), "en", {}) == (None, dialogue)
    # a dialogue track at 12 a minute stays above 40% of the capped bar (12 of 30)
    slower = SubTrack(4, "ass", "eng", None, False, cues=290)
    assert find_target_subtitles(src, _cd25(typeset, slower), "en", {}) == (None, slower)


def test_cd25_full_subtitles_with_signs_are_dialogue(tmp_path):
    src = tmp_path / "show S01E12.mkv"
    for title in ("Full Subtitles (incl. Signs)", "Dialogue + Signs"):
        full = SubTrack(4, "ass", "eng", title, False, cues=350)
        assert find_target_subtitles(src, _cd25(full), "en", {}) == (None, full), title
        thin = SubTrack(4, "ass", "eng", title, False, cues=40)
        assert find_target_subtitles(src, _cd25(thin), "en", {}) == (None, None), title
        unknown = SubTrack(4, "ass", "eng", title, False)
        assert find_target_subtitles(src, _cd25(unknown), "en", {}) == (None, None), title
    signs = SubTrack(4, "ass", "eng", "Signs", False, cues=350)
    assert find_target_subtitles(src, _cd25(signs), "en", {}) == (None, None)


def test_extract_subtitles_removes_its_files_when_ffmpeg_fails(tmp_path, monkeypatch):
    from pathlib import Path

    from fukikae import media

    def fail(args: list[str]) -> None:
        for a in args:
            if a.endswith(".ass"):
                Path(a).write_text("half", encoding="utf-8")
        raise media.MediaError("ffmpeg failed")

    monkeypatch.setattr(media, "run_ffmpeg", fail)
    a = media.SubTrack(index=2, codec="ass", language="eng", title=None, forced=False, cues=1)
    b = media.SubTrack(index=3, codec="ass", language="ger", title=None, forced=False, cues=1)
    with pytest.raises(media.MediaError):
        media.extract_subtitles(tmp_path / "in.mkv", [(a, tmp_path / "a"), (b, tmp_path / "b")])
    assert list(tmp_path.iterdir()) == []
