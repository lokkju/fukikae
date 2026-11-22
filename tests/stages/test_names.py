from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fukikae import media
from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, ConfigError, load_config
from fukikae.job import cache_key, run_stages
from fukikae.models import Diarization, Turn
from fukikae.names import detect
from fukikae.names.inputs import identity_inputs
from fukikae.names.models import NamesResult
from fukikae.stages import names as names_mod
from fukikae.stages.characters import CharactersStage
from fukikae.stages.names import NamesStage
from fukikae.workspace import Episode

WHO = ["NATSU", "LUCY", "HAPPY"]
CLUSTER = {"NATSU": "SPEAKER_00", "HAPPY": "SPEAKER_00", "LUCY": "SPEAKER_01"}
_DURS = [1.6, 2.1, 1.8, 2.4, 1.7, 2.2, 1.9, 2.6, 1.5, 2.0]
_GAPS = [0.5, 0.9, 0.4, 0.7, 1.0, 0.6, 0.8, 0.5, 0.4, 0.9, 0.7]


def _timeline(n: int = 30) -> list[tuple[float, float, str]]:
    t, out = 0.5, []
    for i in range(n):
        d, g = _DURS[i % len(_DURS)], _GAPS[i % len(_GAPS)]
        out.append((round(t, 3), round(t + d, 3), WHO[i % 3]))
        t += d + g
    return out


LINES = _timeline()


def _stamp(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass(
    named: bool = True,
    rename: dict[str, str] | None = None,
    names_at: dict[int, str] | None = None,
) -> str:
    rename = rename or {}
    names_at = names_at or {}
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour",
        "Style: Default,Arial,24,&H00FFFFFF",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    rows = [
        f"Dialogue: 0,{_stamp(a)},{_stamp(b)},Default,"
        f"{names_at.get(i, rename.get(n, n)) if named else ''},"
        f"0,0,0,,Line number {i}."
        for i, (a, b, n) in enumerate(LINES)
    ]
    return "\n".join([*head, *rows, ""])


def make_ws(tmp_path: Path, names: str = "", named: bool = True) -> tuple[Episode, Path]:
    root = tmp_path / "ws"
    root.mkdir(parents=True)
    (root / "fukikae.toml").write_text(f'target_language="en"\nsource_language="ja"\n{names}')
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    source = media_dir / "S01E01.mkv"
    (media_dir / "S01E01.en.ass").write_text(ass(named), encoding="utf-8")
    ep = Episode(
        source=source,
        id="S01E01",
        work=root / ".fukikae" / "episodes" / "S01E01",
        workspace_root=root,
    )
    d = ep.work / "04-diarize"
    d.mkdir(parents=True)
    turns = [Turn(start=a - 0.05, end=b + 0.05, speaker=CLUSTER[n]) for a, b, n in LINES]
    write_json(d / "turns.json", Diarization(backend="pyannote", turns=turns))
    rng = np.random.default_rng(0)
    arrays: dict[str, Any] = {"labels": np.array(["SPEAKER_00", "SPEAKER_01"], dtype=str)}
    for lb in ("SPEAKER_00", "SPEAKER_01"):
        spans = sorted(
            ((t.start + 0.1, t.end - 0.1) for t in turns if t.speaker == lb),
            key=lambda s: s[0] - s[1],
        )
        arrays[f"{lb}/embeddings"] = rng.standard_normal((len(spans), 4)).astype(np.float32)
        arrays[f"{lb}/spans"] = np.array(spans, dtype=np.float64)
        arrays[f"{lb}/clean_seconds"] = np.array(sum(e - s for s, e in spans))
    np.savez(d / "clusters.npz", **arrays)
    return ep, root


def cfg_of(root: Path) -> Config:
    return load_config(root, user_config=None)


def set_names(root: Path, body: str) -> Config:
    (root / "fukikae.toml").write_text(
        f'target_language="en"\nsource_language="ja"\n[names]\n{body}\n'
    )
    return cfg_of(root)


def result(ep: Episode) -> NamesResult:
    return read_json(ep.work / "04c-names" / "names.json", NamesResult)


def explode(*_a: object, **_k: object) -> None:
    raise AssertionError("subtitles were read")


def test_named_sidecar_writes_refined_turns_and_clusters(tmp_path):
    ep, root = make_ws(tmp_path)
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert r.used and r.verdict == "full" and r.reason is None
    assert r.source is not None and r.source["how"] == "target"
    assert {lb for lb, info in r.labels.items() if info.key} == {
        "SPEAKER_00~natsu",
        "SPEAKER_00~happy",
        "SPEAKER_01~lucy",
    }
    turns = read_json(ep.work / "04c-names" / "turns.json", Diarization)
    assert {t.speaker for t in turns.turns} == set(r.labels)
    with np.load(ep.work / "04c-names" / "clusters.npz") as z:
        assert sorted(str(x) for x in z["labels"]) == sorted(r.labels)
        assert z["SPEAKER_01~lucy/embeddings"].shape == (10, 4)
    assert r.coverage["named_turns"] == 30 and r.coverage["cut_turns"] == 0
    assert all(p.turn == i for i, p in enumerate(r.pieces))


def test_blank_names_write_only_names_json(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert not r.used and r.verdict == "none" and r.reason and "share" in r.reason
    assert r.tried and r.tried[0]["how"] == "target"
    assert not (ep.work / "04c-names" / "turns.json").exists()
    assert not (ep.work / "04c-names" / "clusters.npz").exists()


def test_use_off_reads_no_subtitles(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, 'use="off"')
    monkeypatch.setattr(media, "extract_subtitles", explode)
    monkeypatch.setattr(detect, "assess_path", explode)
    monkeypatch.setattr(detect, "candidate_tracks", explode)
    stage = NamesStage(cfg)
    assert stage.inputs(ep) == [
        ep.work / "04-diarize" / "turns.json",
        ep.work / "04-diarize" / "clusters.npz",
    ]
    stage.run(ep, cfg)
    assert result(ep).verdict == "off" and not result(ep).used


def test_holdout_episode_is_unused_and_reads_no_names(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, 'holdout_episodes=["S01E01"]')
    monkeypatch.setattr(detect, "assess_path", explode)
    monkeypatch.setattr(detect, "candidate_tracks", explode)
    NamesStage(cfg).run(ep, cfg)
    r = result(ep)
    assert r.verdict == "holdout" and not r.used and r.lines == [] and r.source is None


def test_hidden_lines_are_stable_and_act_unnamed(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, "hide_fraction=0.2\nhide_seed=0")
    NamesStage(cfg).run(ep, cfg)
    a = result(ep)
    NamesStage(cfg).run(ep, cfg)
    b = result(ep)
    assert a.hidden == b.hidden and len(a.hidden) == 6  # 2 of each key's 10 lines
    hidden = set(a.hidden)
    assert all(not hidden & set(p.line_ids) for p in a.pieces)
    by_id = {ln.id: ln for ln in a.lines}
    assert all(by_id[i].key is not None for i in hidden)  # hidden lines keep their key
    for i in hidden:  # each hidden line had a turn of its own: now unnamed
        ln = by_id[i]
        (p,) = [p for p in a.pieces if p.start <= ln.start and ln.end <= p.end]
        assert p.key is None
    assert a.coverage["hidden"] == 6
    c_cfg = set_names(root, "hide_fraction=0.2\nhide_seed=1")
    NamesStage(c_cfg).run(ep, c_cfg)
    assert result(ep).hidden != a.hidden


def test_config_key_changes_with_holdout_and_hide_only(tmp_path):
    ep, root = make_ws(tmp_path)
    stage = NamesStage()

    def key(body: str) -> dict:
        return stage.config_key(set_names(root, body), ep)

    base = key("")
    assert (
        len({str(base), str(key('holdout_episodes=["S01E01"]')), str(key("hide_fraction=0.2"))})
        == 3
    )
    # nothing this episode's names stage reads
    assert key('holdout_episodes=["S01E09"]') == base
    assert key("min_new_seconds=8") == base
    assert key("hide_seed=4") == base  # no lines hidden, so the seed is moot
    assert key("hide_fraction=0.2\nhide_seed=4") != key("hide_fraction=0.2")
    assert key("turn_share=0.7") != base


def test_inputs_follow_the_subtitle_content_only(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    stage = NamesStage(cfg)
    before = cache_key(stage, ep, cfg)
    side = ep.source.with_name("S01E01.en.ass")
    side.write_text(ass(), encoding="utf-8")  # same content, new mtime
    assert cache_key(stage, ep, cfg) == before
    side.write_text(ass(rename={"HAPPY": "CHARLE"}), encoding="utf-8")
    assert cache_key(stage, ep, cfg) != before


def test_inputs_follow_the_diarize_output(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    stage = NamesStage(cfg)
    before = cache_key(stage, ep, cfg)
    turns = ep.work / "04-diarize" / "turns.json"
    d = read_json(turns, Diarization)
    write_json(turns, d.model_copy(update={"turns": d.turns[:-1]}))
    assert cache_key(stage, ep, cfg) != before


def test_embedded_track_is_extracted_once(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    ep.source.with_name("S01E01.en.ass").unlink()
    ep.source.write_bytes(b"mkv")
    track = media.SubTrack(index=3, codec="ass", language="eng", title=None, forced=False, cues=30)
    monkeypatch.setattr(
        media, "probe", lambda _p: media.ProbeResult(subtitles=[track], duration=90.0)
    )
    calls: list[int] = []

    def extract(_source: Path, wanted: list[tuple[media.SubTrack, Path]]) -> list[Path]:
        out = []
        for t, path in wanted:
            calls.append(t.index)
            dest = Path(path).with_name(Path(path).name + ".ass")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(ass(), encoding="utf-8")
            out.append(dest)
        return out

    monkeypatch.setattr(media, "extract_subtitles", extract)
    cfg = cfg_of(root)
    stage = NamesStage(cfg)
    stage.inputs(ep)
    NamesStage(cfg).run(ep, cfg)
    assert calls == [3]
    r = result(ep)
    assert r.used and r.source is not None and r.source["track"] == 3


def test_missing_names_file_is_a_config_error(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, 'file="names/{episode}.ass"')
    stage = NamesStage(cfg)
    assert stage.inputs(ep)[:2] == [
        ep.work / "04-diarize" / "turns.json",
        ep.work / "04-diarize" / "clusters.npz",
    ]
    with pytest.raises(ConfigError, match="not found"):
        stage.run(ep, cfg)
    # config_key never raises outside run(): the error is part of the key, and run() raises it
    key = stage.config_key(cfg, ep)
    assert "not found" in key["candidates_error"]


def test_names_file_is_read_first(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    names_dir = ep.source.parent / "names"
    names_dir.mkdir()
    (names_dir / "S01E01.ass").write_text(ass(), encoding="utf-8")
    cfg = set_names(root, 'file="names/{episode}.ass"')
    NamesStage(cfg).run(ep, cfg)
    r = result(ep)
    assert r.used and r.source is not None and r.source["how"] == "file"


def test_rerun_is_byte_identical(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    out = ep.work / "04c-names"
    NamesStage(cfg).run(ep, cfg)
    first = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    NamesStage(cfg).run(ep, cfg)
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == first
    assert set(first) == {"names.json", "turns.json", "clusters.npz"}


def test_a_rerun_without_names_removes_the_refined_files(tmp_path):
    ep, root = make_ws(tmp_path)
    NamesStage().run(ep, cfg_of(root))
    cfg = set_names(root, 'use="off"')
    NamesStage(cfg).run(ep, cfg)
    assert not (ep.work / "04c-names" / "turns.json").exists()
    assert identity_inputs(ep.work)[0].parent.name == "04-diarize"


def test_identity_inputs_follow_names_json(tmp_path):
    named, nroot = make_ws(tmp_path / "a")
    blank, broot = make_ws(tmp_path / "b", named=False)
    assert identity_inputs(named.work) == (
        named.work / "04-diarize" / "turns.json",
        named.work / "04-diarize" / "clusters.npz",
        None,
    )
    NamesStage().run(named, cfg_of(nroot))
    NamesStage().run(blank, cfg_of(broot))
    t, c, r = identity_inputs(named.work)
    assert t.parent.name == c.parent.name == "04c-names" and r is not None and r.used
    t, c, r = identity_inputs(blank.work)
    assert t.parent.name == c.parent.name == "04-diarize" and r is not None and not r.used


def test_without_names_characters_output_is_unchanged(tmp_path):
    def characters_twice(ep: Episode, cfg: Config, names_between: bool) -> bytes:
        CharactersStage().run(ep, cfg)
        if names_between:
            NamesStage(cfg).run(ep, cfg)
            names_json, *pair = CharactersStage().inputs(ep)
            assert names_json == ep.work / "04c-names" / "names.json" and names_json.exists()
            assert identity_inputs(ep.work)[:2] == tuple(pair)  # 04-diarize's, names unused
        CharactersStage().run(ep, cfg)
        return (ep.work / "05-characters" / "assignments.json").read_bytes()

    a, aroot = make_ws(tmp_path / "a", named=False)
    b, broot = make_ws(tmp_path / "b", named=False)
    assert characters_twice(a, cfg_of(aroot), False) == characters_twice(b, cfg_of(broot), True)
    assert sorted(p.name for p in (b.work / "04c-names").iterdir() if p.is_file()) == ["names.json"]


def test_second_run_is_up_to_date(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    assert run_stages(ep, cfg, [NamesStage(cfg)]) == ["names"]
    assert run_stages(ep, cfg, [NamesStage(cfg)]) == []


def test_pipeline_runs_names_between_diarize_and_characters():
    from fukikae.cli import pipeline_stages

    names = [s.name for s in pipeline_stages(Config())]
    assert names.index("names") == names.index("diarize") + 1 == names.index("characters") - 1


def test_missing_names_track_is_a_config_error(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, "track=9")
    with pytest.raises(ConfigError, match="no subtitle track 9"):
        NamesStage(cfg).run(ep, cfg)


def test_hidden_lines_count_as_unnamed_in_detection(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, "hide_fraction=0.2\nhide_seed=0")
    NamesStage(cfg).run(ep, cfg)
    r = result(ep)
    assert len(r.hidden) == 6
    assert r.coverage["dialogue"] == 30 and r.coverage["named"] == 24
    assert r.verdict == "partial"  # 80% named once the hidden lines are unnamed


def _inner_typo_ws(tmp_path: Path) -> tuple[Episode, Path]:
    """Lines 2 and 5 (HAPPY's) spelled NATSO/I: an inner-voice key one edit from natsu."""
    ep, root = make_ws(tmp_path)
    side = ep.source.with_name("S01E01.en.ass")
    side.write_text(ass(names_at={2: "NATSO/I", 5: "NATSO/I"}), encoding="utf-8")
    return ep, root


def test_hidden_keys_reach_no_label_exclusion_or_suggestion(tmp_path, monkeypatch):
    ep, root = _inner_typo_ws(tmp_path / "shown")
    NamesStage().run(ep, cfg_of(root))
    shown = result(ep)
    assert ("natso", "natsu") in shown.suggestions
    assert any(info.key == "natso" for info in shown.labels.values())
    assert len(shown.excluded) == 2

    def hide_natso(lines, *_a, **_k):
        return sorted(ln.id for ln in lines if ln.key == "natso")

    monkeypatch.setattr(names_mod, "hide_lines", hide_natso)
    ep, root = _inner_typo_ws(tmp_path / "hidden")
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert len(r.hidden) == 2
    assert all(ln.key == "natso" for ln in r.lines if ln.id in set(r.hidden))
    assert not any(info.key == "natso" for info in r.labels.values())
    assert r.excluded == [] and r.suggestions == []
    assert not any(set(p.line_ids) & set(r.hidden) for p in r.pieces)


def test_failed_rerun_leaves_no_names_json_beside_new_files(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    out = ep.work / "04c-names"
    NamesStage(cfg).run(ep, cfg)
    assert (out / "names.json").exists()

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(names_mod, "refine_clusters", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        NamesStage(cfg).run(ep, cfg)
    assert not (out / "names.json").exists()
    assert not (out / "turns.json").exists() and not (out / "clusters.npz").exists()
    assert identity_inputs(ep.work)[:2] == (
        ep.work / "04-diarize" / "turns.json",
        ep.work / "04-diarize" / "clusters.npz",
    )


def test_track_copy_follows_an_edit_with_the_same_size_and_mtime(tmp_path):
    import os

    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    before = cache_key(NamesStage(cfg), ep, cfg)
    side = ep.source.with_name("S01E01.en.ass")
    st = side.stat()
    text = side.read_text(encoding="utf-8")
    new = text.replace("HAPPY", "CHARL")  # same length
    assert len(new) == len(text) and new != text
    side.write_text(new, encoding="utf-8")
    os.utime(side, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert side.stat().st_size == st.st_size and side.stat().st_mtime_ns == st.st_mtime_ns
    assert cache_key(NamesStage(cfg), ep, cfg) != before


@pytest.mark.parametrize("body", ['holdout_episodes=["S01E01"]', 'use="off"'])
def test_unread_names_remove_the_track_copies(tmp_path, body):
    ep, root = make_ws(tmp_path)
    NamesStage().run(ep, cfg_of(root))
    tracks = ep.work / "04c-names" / "tracks"
    assert tracks.is_dir() and any(tracks.iterdir())
    cfg = set_names(root, body)
    NamesStage(cfg).run(ep, cfg)
    assert not tracks.exists()


def test_holdout_matches_any_spelling_of_the_episode(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, 'holdout_episodes=["s01e01"]')
    monkeypatch.setattr(detect, "candidate_tracks", explode)
    NamesStage(cfg).run(ep, cfg)
    assert result(ep).verdict == "holdout"


def test_config_key_under_use_off_reads_nothing(tmp_path, monkeypatch):
    ep, root = make_ws(tmp_path)
    cfg = set_names(root, 'use="off"')
    monkeypatch.setattr(media, "extract_subtitles", explode)
    monkeypatch.setattr(media, "probe", explode)
    monkeypatch.setattr(detect, "candidate_tracks", explode)
    assert NamesStage(cfg).config_key(cfg, ep) == {
        "version": names_mod.NAMES_VERSION,
        "use": "off",
    }


def _embedded_ws(tmp_path: Path, monkeypatch) -> tuple[Episode, Path]:
    ep, root = make_ws(tmp_path)
    ep.source.with_name("S01E01.en.ass").unlink()
    ep.source.write_bytes(b"mkv")
    track = media.SubTrack(index=3, codec="ass", language="eng", title=None, forced=False, cues=30)
    monkeypatch.setattr(
        media, "probe", lambda _p: media.ProbeResult(subtitles=[track], duration=90.0)
    )
    return ep, root


def test_ffmpeg_failure_leaves_the_track_index_unadvanced(tmp_path, monkeypatch):
    import os

    ep, root = _embedded_ws(tmp_path, monkeypatch)

    def extract(_source: Path, wanted: list[tuple[media.SubTrack, Path]]) -> list[Path]:
        out = []
        for _t, path in wanted:
            dest = Path(path).with_name(Path(path).name + ".ass")
            dest.write_text(ass(), encoding="utf-8")
            out.append(dest)
        return out

    monkeypatch.setattr(media, "extract_subtitles", extract)
    cfg = cfg_of(root)
    NamesStage(cfg).run(ep, cfg)
    index = ep.work / "04c-names" / "tracks" / "index.json"
    first = index.read_bytes()

    def fail(*_a: object, **_k: object) -> list[Path]:
        raise media.MediaError("ffmpeg failed")

    monkeypatch.setattr(media, "extract_subtitles", fail)
    st = ep.source.stat()
    os.utime(ep.source, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a new source version
    with pytest.raises(media.MediaError):
        NamesStage(cfg).run(ep, cfg)
    assert index.read_bytes() == first
    assert not (ep.work / "04c-names" / "names.json").exists()


# ---- operator pitch splits (Task 15) ----

RATE = 16000
HZ = {"NATSU": 120.0, "HAPPY": 250.0, "LUCY": 220.0}


def _harmonic(hz: float, n: int) -> np.ndarray:
    t = np.arange(n) / RATE
    x = sum(a * np.sin(2 * np.pi * hz * k * t) for k, a in enumerate((1, 0.5, 0.33, 0.25), 1))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def write_vocals(ep: Episode, hz_at: dict[int, float] | None = None) -> None:
    """A tone per subtitle line at its speaker's pitch (``hz_at`` overrides by line index)."""
    from scipy.io import wavfile

    hz_at = hz_at or {}
    x = np.zeros(int((LINES[-1][1] + 2) * RATE), np.float32)
    for i, (a, b, who) in enumerate(LINES):
        lo, hi = int(a * RATE), int(b * RATE)
        x[lo:hi] = _harmonic(hz_at.get(i, HZ[who]), hi - lo)
    d = ep.work / "01-separate"
    d.mkdir(parents=True, exist_ok=True)
    wavfile.write(d / "vocals.wav", RATE, (np.clip(x, -1, 1) * 32767).astype("<i2"))


def write_split(root: Path, cluster: str = "SPEAKER_00", **bands: float) -> Path:
    import json

    path = root / "studio" / "splits" / "S01E01.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {cluster: {"by": "pitch", "male_below": 165.0, "female_above": 200.0, **bands}}
    path.write_text(json.dumps(body, sort_keys=True))
    return path


def _labels_by_who(ep: Episode) -> dict[str, set[str]]:
    turns = read_json(ep.work / "04c-names" / "turns.json", Diarization).turns
    out: dict[str, set[str]] = {}
    for t in turns:
        mid = (t.start + t.end) / 2
        who = next(n for a, b, n in LINES if a <= mid <= b)
        out.setdefault(who, set()).add(t.speaker)
    return out


def test_pitch_split_relabels_turns_by_their_median(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    write_split(root)
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert not r.used and r.verdict == "none" and r.refined
    assert _labels_by_who(ep) == {
        "NATSU": {"SPEAKER_00~voice-m"},
        "HAPPY": {"SPEAKER_00~voice-f"},
        "LUCY": {"SPEAKER_01"},
    }
    assert r.labels["SPEAKER_00~voice-f"].cls == "voice"
    assert r.labels["SPEAKER_00~voice-f"].parent == "SPEAKER_00"
    s = r.voice_splits["SPEAKER_00"]
    assert s.labels == ["SPEAKER_00~voice-f", "SPEAKER_00~voice-m"] and s.seconds["u"] == 0
    with np.load(ep.work / "04c-names" / "clusters.npz") as z:
        assert z["SPEAKER_00~voice-m/embeddings"].shape == (10, 4)
        assert z["SPEAKER_00~voice-f/embeddings"].shape == (10, 4)
        assert z["SPEAKER_01/embeddings"].shape == (10, 4)
    t, c, got = identity_inputs(ep.work)
    assert t.parent.name == c.parent.name == "04c-names" and got is not None


def test_unknown_turns_join_the_larger_part(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    happy = [i for i, (_, _, n) in enumerate(LINES) if n == "HAPPY"]
    write_vocals(ep, dict.fromkeys(happy[:3], 182.0))  # between the bands
    write_split(root)
    NamesStage().run(ep, cfg_of(root))
    s = result(ep).voice_splits["SPEAKER_00"]
    natsu = sum(b - a + 0.1 for a, b, n in LINES if n == "NATSU")
    assert s.seconds["m"] == pytest.approx(natsu, abs=1e-6) and s.seconds["m"] > s.seconds["f"]
    assert s.larger == "m"
    turns = read_json(ep.work / "04c-names" / "turns.json", Diarization).turns
    for i in happy[:3]:
        a, b, _ = LINES[i]
        (t,) = [t for t in turns if t.start < (a + b) / 2 < t.end]
        assert t.speaker == "SPEAKER_00~voice-m"


def test_no_split_file_leaves_the_key_unchanged(tmp_path):
    ep, root = make_ws(tmp_path)
    cfg = cfg_of(root)
    stage = NamesStage(cfg)
    key0, inputs0 = stage.config_key(cfg, ep), stage.inputs(ep)
    assert "splits" not in key0
    assert all(p.name != "vocals.wav" for p in inputs0)
    before = cache_key(stage, ep, cfg)
    write_vocals(ep)
    assert cache_key(NamesStage(cfg), ep, cfg) == before  # the stem alone changes nothing
    path = write_split(root)
    with_split = cache_key(NamesStage(cfg), ep, cfg)
    assert with_split != before
    assert ep.work / "01-separate" / "vocals.wav" in NamesStage(cfg).inputs(ep)
    write_split(root, male_below=150.0)
    assert cache_key(NamesStage(cfg), ep, cfg) != with_split
    path.unlink()
    assert cache_key(NamesStage(cfg), ep, cfg) == before


def test_split_with_subtitle_names_splits_only_the_remainder(tmp_path):
    ep, root = make_ws(tmp_path)
    blank = {0: "", 3: "", 2: "", 5: ""}  # two NATSU and two HAPPY lines lose their names
    (tmp_path / "media" / "S01E01.en.ass").write_text(ass(names_at=blank), encoding="utf-8")
    write_vocals(ep)
    write_split(root)
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert r.used and r.refined
    turns = read_json(ep.work / "04c-names" / "turns.json", Diarization).turns
    by_line = {}
    for i, (a, b, _) in enumerate(LINES):
        (t,) = [t for t in turns if t.start < (a + b) / 2 < t.end]
        by_line[i] = t.speaker
    assert by_line[0] == by_line[3] == "SPEAKER_00~voice-m"
    assert by_line[2] == by_line[5] == "SPEAKER_00~voice-f"
    assert by_line[6] == "SPEAKER_00~natsu" and by_line[8] == "SPEAKER_00~happy"
    assert "SPEAKER_00" not in {t.speaker for t in turns}
    assert r.labels["SPEAKER_00~natsu"].cls == "named"
    assert r.voice_splits["SPEAKER_00"].seconds["m"] == pytest.approx(
        sum(LINES[i][1] - LINES[i][0] + 0.1 for i in (0, 3)), abs=1e-6
    )


def test_characters_stage_matches_the_parts_separately(tmp_path):
    from fukikae.models import Assignments

    ep, root = make_ws(tmp_path, names="[registry]\nmin_new_seconds = 5\n", named=False)
    write_vocals(ep)
    write_split(root)
    cfg = cfg_of(root)
    NamesStage(cfg).run(ep, cfg)
    assert CharactersStage().inputs(ep)[1] == ep.work / "04c-names" / "turns.json"
    CharactersStage().run(ep, cfg)
    asg = read_json(ep.work / "05-characters" / "assignments.json", Assignments)
    by = {c.cluster: c for c in asg.clusters}
    assert {"SPEAKER_00~voice-f", "SPEAKER_00~voice-m", "SPEAKER_01"} == set(by)
    f, m = by["SPEAKER_00~voice-f"], by["SPEAKER_00~voice-m"]
    assert f.actor_id is not None and m.actor_id is not None and f.actor_id != m.actor_id


def test_removing_the_split_file_restores_the_unsplit_output(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    path = write_split(root)
    NamesStage().run(ep, cfg_of(root))
    path.unlink()
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert not r.refined and not r.voice_splits
    assert not (ep.work / "04c-names" / "turns.json").exists()
    assert identity_inputs(ep.work)[0].parent.name == "04-diarize"
    assert '"voice_splits"' not in (ep.work / "04c-names" / "names.json").read_text()


def test_a_split_for_a_cluster_not_in_the_episode_changes_nothing(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    write_split(root, cluster="SPEAKER_09")
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    assert not r.refined and "no unnamed turns" in (r.voice_splits["SPEAKER_09"].reason or "")
    assert not (ep.work / "04c-names" / "turns.json").exists()


@pytest.mark.parametrize(
    "body",
    ['{"SPEAKER_00": {"by": "cluster"}}', "[1, 2]", "{not json",
     '{"SPEAKER_00": {"by": "pitch", "male_below": 210, "female_above": 200}}'],
)  # fmt: skip
def test_a_malformed_split_file_is_refused(tmp_path, body):
    from fukikae.names.inputs import SplitFileError

    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    path = write_split(root)
    path.write_text(body)
    with pytest.raises(SplitFileError, match="S01E01.json"):
        NamesStage().run(ep, cfg_of(root))


def test_a_split_without_the_vocals_stem_is_refused(tmp_path):
    from fukikae.names.inputs import SplitFileError

    ep, root = make_ws(tmp_path, named=False)
    write_split(root)
    with pytest.raises(SplitFileError, match="vocals.wav"):
        NamesStage().run(ep, cfg_of(root))


def test_names_show_reports_the_pitch_split(tmp_path):
    from fukikae.names.cli import show_episode
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    write_split(root)
    NamesStage().run(ep, cfg_of(root))
    media_dir = tmp_path / "media"
    (media_dir / "S01E01.mkv").touch()
    (root / "fukikae.toml").write_text(
        f'target_language="en"\nsource_language="ja"\nsource="{media_dir}"\n'
    )
    out = show_episode(Workspace(root), "S01E01")
    assert any(
        ln.startswith("pitch split SPEAKER_00 (male below 165 Hz, female above 200 Hz): ")
        and ln.endswith("SPEAKER_00~voice-f, SPEAKER_00~voice-m")
        for ln in out
    ), out


def test_assignments_older_than_a_split_are_skipped_for_references(tmp_path):
    from fukikae.cards.auto import checked_names
    from fukikae.models import Assignments

    ep, root = make_ws(tmp_path, names="[registry]\nmin_new_seconds = 5\n", named=False)
    write_vocals(ep)
    cfg = cfg_of(root)
    NamesStage(cfg).run(ep, cfg)
    CharactersStage().run(ep, cfg)
    asg_path = ep.work / "05-characters" / "assignments.json"
    assert checked_names(ep.work, read_json(asg_path, Assignments)) == (None, None)
    write_split(root)
    NamesStage(cfg).run(ep, cfg)
    names, why = checked_names(ep.work, read_json(asg_path, Assignments))
    assert names is None and why is not None and "older than its names" in why
    CharactersStage().run(ep, cfg)
    assert checked_names(ep.work, read_json(asg_path, Assignments)) == (None, None)


# ---- identities after a pitch split (Task 15 review: the controller's ruling) ----

MIN_NEW = "[registry]\nmin_new_seconds = 5\n"


def _asg(ep: Episode) -> dict[str, Any]:
    from fukikae.models import Assignments

    a = read_json(ep.work / "05-characters" / "assignments.json", Assignments)
    return {c.cluster: c for c in a.clusters}


def _run_both(ep: Episode, cfg: Config) -> dict[str, Any]:
    NamesStage(cfg).run(ep, cfg)
    CharactersStage().run(ep, cfg)
    return _asg(ep)


def _structured_rows(
    ep: Episode, drop_who: str | None = None, by_who: dict[str, list[float]] | None = None
) -> None:
    """Rewrite the diarize rows: SPEAKER_00's near one direction, SPEAKER_01's near another
    (or each speaker's near ``by_who``'s direction), so acoustic matching is predictable;
    ``drop_who``'s rows left out (a rowless part)."""
    rng = np.random.default_rng(1)
    turns = read_json(ep.work / "04-diarize" / "turns.json", Diarization).turns
    centre = {"SPEAKER_00": np.array([1.0, 0, 0, 0]), "SPEAKER_01": np.array([0, 1.0, 0, 0])}
    arrays: dict[str, Any] = {"labels": np.array(["SPEAKER_00", "SPEAKER_01"], dtype=str)}
    for lb in ("SPEAKER_00", "SPEAKER_01"):
        spans = []
        for t in turns:
            mid = (t.start + t.end) / 2
            who = next(n for a, b, n in LINES if a <= mid <= b)
            if t.speaker == lb and who != drop_who:
                spans.append((t.start + 0.1, t.end - 0.1, who))
        rows = np.array(
            [
                (np.array(by_who[w]) if by_who else centre[lb]) + rng.normal(0, 0.05, 4)
                for *_x, w in spans
            ]
        ).reshape(-1, 4)
        spans = [(a, b) for a, b, _w in spans]
        arrays[f"{lb}/embeddings"] = rows.astype(np.float32)
        arrays[f"{lb}/spans"] = np.array(spans, dtype=np.float64).reshape(-1, 2)
        arrays[f"{lb}/clean_seconds"] = np.array(sum(e - s for s, e in spans))
    np.savez(ep.work / "04-diarize" / "clusters.npz", **arrays)


def _parts_by_length(ep: Episode) -> tuple[str, str]:
    """(longer, shorter) voice part of SPEAKER_00 by the refined turns, ``~voice-f`` longer
    on a tie."""
    turns = read_json(ep.work / "04c-names" / "turns.json", Diarization).turns
    secs = {"f": 0.0, "m": 0.0}
    for t in turns:
        if t.speaker.startswith("SPEAKER_00~voice-"):
            secs[t.speaker[-1]] += t.end - t.start
    f_longer = round(secs["f"], 6) >= round(secs["m"], 6)
    return (
        ("SPEAKER_00~voice-f", "SPEAKER_00~voice-m")
        if f_longer
        else ("SPEAKER_00~voice-m", "SPEAKER_00~voice-f")
    )


# a NATSU line at HAPPY's pitch makes the high part longer; a HAPPY line at NATSU's the low
LONGER = {"f": {0: HZ["HAPPY"]}, "m": {2: HZ["NATSU"]}}


@pytest.mark.parametrize("longer_part", ["f", "m"])
def test_a_split_after_a_characters_run_gives_one_part_a_new_character(tmp_path, longer_part):
    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER[longer_part])
    cfg = cfg_of(root)
    before = _run_both(ep, cfg)
    parent = before["SPEAKER_00"]
    assert parent.character_id is not None
    write_split(root)
    after = _run_both(ep, cfg)
    longer, shorter = _parts_by_length(ep)
    assert longer == f"SPEAKER_00~voice-{longer_part}"
    keep, new = after[longer], after[shorter]
    # no stored gender: the part with more seconds keeps the parent's character
    assert (keep.actor_id, keep.character_id) == (parent.actor_id, parent.character_id)
    assert keep.status == "pinned" and keep.split == "kept"
    assert new.character_id not in (None, parent.character_id)
    assert new.actor_id not in (None, parent.actor_id)
    assert new.status == "new" and new.split == "new"
    assert after["SPEAKER_01"].character_id == before["SPEAKER_01"].character_id


def test_a_stored_gender_picks_the_part_that_keeps_the_character(tmp_path):
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])  # the stored gender f picks the shorter ~voice-f
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    shorter = "SPEAKER_00~voice-f"
    assert parent.character_id is not None
    with Registry(Workspace(root).registry_path) as reg:
        reg.set_character_gender(parent.character_id, "f")
    write_split(root)
    after = _run_both(ep, cfg)
    assert _parts_by_length(ep)[1] == shorter
    assert after[shorter].character_id == parent.character_id
    assert after[shorter].split == "kept"
    other = next(lb for lb in after if lb.startswith("SPEAKER_00~") and lb != shorter)
    assert after[other].character_id not in (None, parent.character_id)
    assert after[other].status == "new"


def test_reruns_after_a_split_keep_both_assignments(tmp_path):
    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])
    cfg = cfg_of(root)
    _run_both(ep, cfg)
    write_split(root)
    first = _run_both(ep, cfg)
    for _ in range(2):
        again = _run_both(ep, cfg)
        assert {lb: (a.actor_id, a.character_id) for lb, a in again.items()} == {
            lb: (a.actor_id, a.character_id) for lb, a in first.items()
        }
        assert {a.status for a in again.values()} == {"pinned"}
    longer, shorter = _parts_by_length(ep)
    assert again[longer].split == "kept" and again[shorter].split == "new"


@pytest.mark.parametrize("stored", [None, "f"])
def test_a_rowless_voice_part_is_kept_or_new_never_an_extra(tmp_path, stored):
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep)
    _structured_rows(ep, drop_who="HAPPY")  # the high part keeps no embedding row
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    assert parent.character_id is not None
    if stored is not None:
        with Registry(Workspace(root).registry_path) as reg:
            reg.set_character_gender(parent.character_id, stored)  # type: ignore[arg-type]
    write_split(root)
    after = _run_both(ep, cfg)
    with np.load(ep.work / "04c-names" / "clusters.npz") as z:
        assert z["SPEAKER_00~voice-f/embeddings"].shape[0] == 0
    f, m = after["SPEAKER_00~voice-f"], after["SPEAKER_00~voice-m"]
    keeper = f if stored == "f" else after[_parts_by_length(ep)[0]]
    other = m if keeper is f else f
    assert keeper.character_id == parent.character_id and keeper.split == "kept"
    assert other.status == "new" and other.character_id not in (None, parent.character_id)
    assert "extra" not in {f.status, m.status}
    again = _run_both(ep, cfg)  # the rowless part keeps its character on a rerun too
    assert again["SPEAKER_00~voice-f"].character_id == f.character_id
    assert again["SPEAKER_00~voice-m"].character_id == m.character_id


def test_a_split_on_a_fresh_episode_keeps_the_acoustic_match_for_one_part(tmp_path):
    """No prior assignments: the parent cluster's acoustic match is the character one part
    keeps; the other part is new, whatever it scores against that actor."""
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])
    _structured_rows(ep)
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    assert parent.character_id is not None
    with Registry(Workspace(root).registry_path) as reg:  # named: a fresh run keeps it
        reg.rename_character(parent.character_id, "Natsu")
    write_split(root)
    NamesStage(cfg).run(ep, cfg)
    CharactersStage(rematch=True).run(ep, cfg)
    after = _asg(ep)
    longer, shorter = _parts_by_length(ep)
    assert after[longer].character_id == parent.character_id
    assert after[longer].status == "matched" and after[longer].split == "kept"
    assert after[shorter].status == "new" and after[shorter].actor_id != parent.actor_id


def test_a_characters_rerun_pins_its_prior_assignments(tmp_path):
    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    cfg = cfg_of(root)
    CharactersStage().run(ep, cfg)
    first = _asg(ep)
    CharactersStage().run(ep, cfg)
    again = _asg(ep)
    assert {lb: a.status for lb, a in first.items()} == {"SPEAKER_00": "new", "SPEAKER_01": "new"}
    assert {lb: a.status for lb, a in again.items()} == {
        "SPEAKER_00": "pinned",
        "SPEAKER_01": "pinned",
    }
    assert {lb: a.character_id for lb, a in again.items()} == {
        lb: a.character_id for lb, a in first.items()
    }
    assert all(a.split is None for a in again.values())
    text = (ep.work / "05-characters" / "assignments.json").read_text()
    assert '"split"' not in text  # unsplit assignments are written as before


def test_registry_resolve_agrees_on_the_split(tmp_path):
    from fukikae.registry.resolve import resolve_episodes
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])
    _structured_rows(ep)
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    assert parent.character_id is not None
    ws = Workspace(root)
    with Registry(ws.registry_path) as reg:  # named, so resolve keeps the character
        reg.rename_character(parent.character_id, "Natsu")
    write_split(root)
    _run_both(ep, cfg)  # resolve reads assignments that answer the refined clusters
    resolve_episodes(ws, [ep], threshold=0.6)
    after = _asg(ep)
    longer, shorter = _parts_by_length(ep)
    assert after[longer].character_id == parent.character_id and after[longer].split == "kept"
    assert after[shorter].character_id not in (None, parent.character_id)
    assert after[shorter].status == "new" and after[shorter].split == "new"
    with Registry(ws.registry_path) as reg:  # a later stored gender never flips the keeper
        reg.set_character_gender(parent.character_id, shorter[-1])  # type: ignore[arg-type]
    kept = {lb: a.character_id for lb, a in after.items()}
    resolve_episodes(ws, [ep], threshold=0.6)
    assert {lb: a.character_id for lb, a in _asg(ep).items() if "~voice-" in lb} == {
        lb: c for lb, c in kept.items() if "~voice-" in lb
    }


def test_registry_resolve_never_makes_a_rowless_voice_part_an_extra(tmp_path):
    from fukikae.registry.resolve import resolve_episodes
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep)
    _structured_rows(ep, drop_who="HAPPY")
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    assert parent.character_id is not None
    ws = Workspace(root)
    with Registry(ws.registry_path) as reg:
        reg.rename_character(parent.character_id, "Natsu")
        reg.set_character_gender(parent.character_id, "m")
    write_split(root)
    _run_both(ep, cfg)  # resolve reads assignments that answer the refined clusters
    resolve_episodes(ws, [ep], threshold=0.6)
    after = _asg(ep)
    assert after["SPEAKER_00~voice-m"].character_id == parent.character_id
    f = after["SPEAKER_00~voice-f"]
    assert f.status == "new" and f.character_id not in (None, parent.character_id)


# ---- the split file's fingerprint guard ----


def _write_split_fp(root: Path, fingerprint: Any) -> None:
    import json

    path = write_split(root)
    body = json.loads(path.read_text())
    body["SPEAKER_00"]["fingerprint"] = fingerprint
    path.write_text(json.dumps(body))


def _diarize_fp(ep: Episode, cluster: str = "SPEAKER_00") -> dict[str, Any] | None:
    from fukikae.names.inputs import cluster_fingerprint

    turns = read_json(ep.work / "04-diarize" / "turns.json", Diarization).turns
    return cluster_fingerprint(turns, cluster)


def test_a_split_with_a_matching_fingerprint_is_applied_and_recorded_as_checked(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    _write_split_fp(root, _diarize_fp(ep))
    NamesStage().run(ep, cfg_of(root))
    v = result(ep).voice_splits["SPEAKER_00"]
    assert v.check == "matched" and v.labels == ["SPEAKER_00~voice-f", "SPEAKER_00~voice-m"]


def test_a_split_file_without_a_fingerprint_is_applied_as_unchecked(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    write_split(root)
    NamesStage().run(ep, cfg_of(root))
    v = result(ep).voice_splits["SPEAKER_00"]
    assert v.check == "unchecked" and len(v.labels) == 2


def test_a_split_whose_cluster_changed_since_the_decision_is_skipped(tmp_path):
    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    _write_split_fp(root, _diarize_fp(ep, "SPEAKER_01"))  # another cluster's turns
    NamesStage().run(ep, cfg_of(root))
    r = result(ep)
    v = r.voice_splits["SPEAKER_00"]
    assert v.check == "changed" and v.labels == [] and not r.refined
    assert v.reason is not None and "re-split" in v.reason and "10 turns" in v.reason
    assert not (ep.work / "04c-names" / "turns.json").exists()


@pytest.mark.parametrize(
    "fp", [{"turns": 3}, {"turns": 0, "seconds": 1.0, "sha256": "0" * 64}, "abc", []]
)
def test_a_malformed_fingerprint_is_refused(tmp_path, fp):
    from fukikae.names.inputs import SplitFileError

    ep, root = make_ws(tmp_path, named=False)
    write_vocals(ep)
    _write_split_fp(root, fp)
    with pytest.raises(SplitFileError, match="fingerprint"):
        NamesStage().run(ep, cfg_of(root))


# ---- the keeper rule decides only for parts without an identity (Task 15 re-review) ----


def _decide(root: Path, op: str, **fields: Any) -> None:
    from fukikae.registry.store import Registry
    from fukikae.studio.apply import apply_decision, stamp
    from fukikae.studio.model import parse_decision
    from fukikae.workspace import Workspace

    ws = Workspace(root)
    with Registry(ws.registry_path) as reg:
        d = stamp(reg, parse_decision({"op": op, **fields}))
    apply_decision(ws, d)


def _who_has(asg: dict[str, Any]) -> dict[str, tuple[int | None, int | None]]:
    return {lb: (a.actor_id, a.character_id) for lb, a in asg.items()}


def _split_once(tmp_path: Path, structured: bool = True):
    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])
    if structured:
        _structured_rows(ep)
    cfg = cfg_of(root)
    parent = _run_both(ep, cfg)["SPEAKER_00"]
    write_split(root)
    after = _run_both(ep, cfg)
    longer, shorter = _parts_by_length(ep)
    return ep, root, cfg, parent, after, longer, shorter


def _rematch(ep: Episode, cfg: Config) -> dict[str, Any]:
    NamesStage(cfg).run(ep, cfg)
    CharactersStage(rematch=True).run(ep, cfg)
    return _asg(ep)


def _resolve(root: Path, ep: Episode) -> dict[str, Any]:
    from fukikae.registry.resolve import resolve_episodes
    from fukikae.workspace import Workspace

    resolve_episodes(Workspace(root), [ep], threshold=0.6)
    return _asg(ep)


def _voice(asg: dict[str, Any]) -> dict[str, tuple[int | None, int | None]]:
    return {lb: v for lb, v in _who_has(asg).items() if "~voice-" in lb}


def test_rematch_keeps_the_voice_parts_identities(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path)
    for _ in range(2):
        assert _voice(_rematch(ep, cfg)) == _voice(after)


def test_an_operator_merge_of_the_new_part_into_the_keeper_stays(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path, structured=False)
    keep_c, new_c = after[longer].character_id, after[shorter].character_id
    _decide(root, "merge", keep=keep_c, drop=new_c)
    merged = _asg(ep)
    assert merged[shorter].character_id == keep_c and merged[shorter].split is None
    for run in (_run_both, _run_both, _rematch):
        a = run(ep, cfg)
        assert a[shorter].character_id == keep_c and a[longer].character_id == keep_c
        assert a[shorter].split is None


def test_a_named_locked_new_part_keeps_its_character_on_reruns_and_rematch(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path, structured=False)
    new_c = after[shorter].character_id
    _decide(root, "name", character=new_c, name="Happy")
    _decide(root, "lock", character=new_c)
    for run in (_run_both, _rematch, _rematch):
        a = run(ep, cfg)
        assert a[shorter].character_id == new_c
        assert a[longer].character_id == parent.character_id


def test_a_stored_gender_change_never_flips_the_keeper(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path)
    _decide(root, "name", character=after[longer].character_id, name="Natsu")
    _decide(root, "gender", character=after[longer].character_id, gender=shorter[-1])
    assert _voice(_run_both(ep, cfg)) == _voice(after)
    assert _voice(_resolve(root, ep)) == _voice(after)
    assert _voice(_rematch(ep, cfg)) == _voice(after)


def test_removing_the_split_restores_byte_identical_unsplit_assignments(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path, structured=False)
    (root / "studio" / "splits" / "S01E01.json").unlink()
    a = _run_both(ep, cfg)
    assert a["SPEAKER_00"].character_id == parent.character_id
    assert a["SPEAKER_00"].split is None
    assert '"split"' not in (ep.work / "05-characters" / "assignments.json").read_text()


def test_resolve_keeps_both_named_parts_and_agrees_with_the_characters_stage(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path)
    _decide(root, "name", character=after[longer].character_id, name="Natsu")
    _decide(root, "name", character=after[shorter].character_id, name="Happy")
    want = _voice(_run_both(ep, cfg))
    assert want == _voice(after)
    assert _voice(_resolve(root, ep)) == want
    assert _voice(_resolve(root, ep)) == want
    assert _voice(_run_both(ep, cfg)) == want
    assert _voice(_rematch(ep, cfg)) == want


def test_resolve_keeps_an_operator_merge_of_the_parts(tmp_path):
    ep, root, cfg, parent, after, longer, shorter = _split_once(tmp_path)
    keep_c = after[longer].character_id
    _decide(root, "name", character=keep_c, name="Natsu")
    _decide(root, "merge", keep=keep_c, drop=after[shorter].character_id)
    r = _resolve(root, ep)
    assert r[longer].character_id == keep_c and r[shorter].character_id == keep_c


def test_a_first_run_non_keeper_matched_to_another_character_keeps_it(tmp_path):
    """No earlier assignments: the parent's match (Natsu) goes to the keeper; the other part
    matches Happy, an existing character other than the keeper's, and keeps it rather than
    becoming new."""
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    ep, root = make_ws(tmp_path, names=MIN_NEW, named=False)
    write_vocals(ep, LONGER["m"])
    _structured_rows(ep, by_who={"NATSU": [1, 0, 0, 0], "HAPPY": [0.6, 0.8, 0, 0],
                                 "LUCY": [0, 0, 1, 0]})  # fmt: skip
    with Registry(Workspace(root).registry_path) as reg:
        natsu = reg.create_actor_with_character(np.array([1, 0, 0, 0], np.float32))
        happy = reg.create_actor_with_character(np.array([0, 1, 0, 0], np.float32))
        reg.rename_character(natsu[1], "Natsu")
        reg.rename_character(happy[1], "Happy")
        reg.set_character_gender(natsu[1], "m")
    write_split(root)
    cfg = cfg_of(root)
    a = _run_both(ep, cfg)
    assert _who_has(a)["SPEAKER_00~voice-m"] == natsu and a["SPEAKER_00~voice-m"].split == "kept"
    assert _who_has(a)["SPEAKER_00~voice-f"] == happy
    assert a["SPEAKER_00~voice-f"].status == "matched"
    assert _voice(_run_both(ep, cfg)) == _voice(a)
    assert _voice(_resolve(root, ep)) == _voice(a)
