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
            assert identity_inputs(ep.work)[:2] == tuple(CharactersStage().inputs(ep))
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
