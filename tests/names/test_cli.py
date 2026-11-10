"""`fukikae names scan|show` (Task 10)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae import media
from fukikae.artifacts import read_json, write_json
from fukikae.cli import app
from fukikae.config import load_config
from fukikae.models import Diarization, Turn
from fukikae.names.inputs import names_report_path
from fukikae.names.models import NamesReport
from fukikae.registry.store import Registry
from fukikae.stages.characters import CharactersStage
from fukikae.stages.names import NamesStage
from fukikae.workspace import Workspace
from tests.stages.test_names import _stamp

runner = CliRunner()

HEAD = [
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
CLUSTER = {"NEBARU": "SPEAKER_00", "NABARU": "SPEAKER_00", "LUCY": "SPEAKER_01",
           "HAPPY": "SPEAKER_01"}  # fmt: skip


def _lines(counts: dict[str, int]) -> list[tuple[float, float, str]]:
    """Lines round-robin over the names until each has its count, 2 s each with 0.6 s gaps."""
    left = dict(counts)
    out: list[tuple[float, float, str]] = []
    t = 0.5
    while any(left.values()):
        for name in counts:
            if left[name]:
                out.append((round(t, 3), round(t + 2.0, 3), name))
                left[name] -= 1
                t += 2.6
    return out


def _episode(root: Path, media_dir: Path, eid: str, counts: dict[str, int]) -> None:
    lines = _lines(counts)
    (media_dir / f"{eid}.mkv").write_bytes(b"")
    rows = [
        f"Dialogue: 0,{_stamp(a)},{_stamp(b)},Default,{n},0,0,0,,Line {i}."
        for i, (a, b, n) in enumerate(lines)
    ]
    (media_dir / f"{eid}.en.ass").write_text("\n".join([*HEAD, *rows, ""]), encoding="utf-8")
    d = Workspace(root).work_dir(eid) / "04-diarize"
    d.mkdir(parents=True)
    turns = [Turn(start=a - 0.05, end=b + 0.05, speaker=CLUSTER[n]) for a, b, n in lines]
    write_json(d / "turns.json", Diarization(backend="pyannote", turns=turns))
    rng = np.random.default_rng(len(lines))
    labels = sorted({t.speaker for t in turns})
    arrays: dict[str, Any] = {"labels": np.array(labels, dtype=str)}
    for lb in labels:
        spans = sorted(
            ((t.start + 0.1, t.end - 0.1) for t in turns if t.speaker == lb),
            key=lambda s: s[0] - s[1],
        )
        arrays[f"{lb}/embeddings"] = rng.standard_normal((len(spans), 4)).astype(np.float32)
        arrays[f"{lb}/spans"] = np.array(spans, dtype=np.float64)
        arrays[f"{lb}/clean_seconds"] = np.array(sum(e - s for s, e in spans))
    np.savez(d / "clusters.npz", **arrays)


# E01: NABARU 3 lines beside NEBARU 18 (a typo); E02: NABARU 3 beside NEBARU 2. Pooled over
# the two episodes NABARU has 6 lines, too many for a typo, so only a per-episode count
# suggests the alias.
E01 = {"NEBARU": 18, "LUCY": 10, "NABARU": 3}
E02 = {"NEBARU": 2, "LUCY": 10, "HAPPY": 10, "NABARU": 3}


@pytest.fixture
def cli_ws(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "ws"
    media_dir = tmp_path / "media"
    root.mkdir()
    media_dir.mkdir()
    (root / "fukikae.toml").write_text(
        f'target_language = "en"\nsource_language = "ja"\nsource = "{media_dir}"\n'
    )
    _episode(root, media_dir, "S01E01", E01)
    _episode(root, media_dir, "S01E02", E02)
    monkeypatch.setattr(media, "probe", lambda *_a, **_k: media.ProbeResult())
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".fukikae/cache" not in str(p)
    }


def _run_names(root: Path, *episodes: str) -> None:
    ws = Workspace(root)
    cfg = load_config(root, user_config=None)
    for e in episodes:
        ep = ws.find_episode(e)
        NamesStage(cfg).run(ep, cfg)
        CharactersStage().run(ep, cfg)


def test_scan_prints_verdict_and_changes_nothing(cli_ws: Path):
    before = _snapshot(cli_ws)
    r = runner.invoke(app, ["names", "scan", str(cli_ws)])
    assert r.exit_code == 0, r.output
    assert "S01E01: full from target file S01E01.en.ass" in r.output
    assert "S01E02: full" in r.output
    assert _snapshot(cli_ws) == before


def test_scan_suggests_aliases_per_episode(cli_ws: Path):
    """3 NABARU lines against 18 NEBARU in one episode is a typo suggestion; counted over the
    series (6 against 20) it would not be, so scan counts each episode on its own."""
    from collections import Counter

    from fukikae.names.normalize import suggest_aliases

    assert suggest_aliases(Counter({"nabaru": 6, "nebaru": 20, "lucy": 20})) == []
    r = runner.invoke(app, ["names", "scan", str(cli_ws)])
    assert r.exit_code == 0, r.output
    by_ep = r.output.split("S01E02:")
    assert "'nabaru' (3 lines) is one edit from 'nebaru' (18 lines)" in by_ep[0]
    assert f"`fukikae registry alias {cli_ws} <character> NABARU`" in by_ep[0]
    assert "suggestion" not in by_ep[1]


def test_scan_names_the_character_an_alias_would_join(cli_ws: Path):
    _run_names(cli_ws, "S01E01")
    with Registry(Workspace(cli_ws).registry_path) as reg:
        c = reg.character_for_key("nebaru")
    assert c is not None
    r = runner.invoke(app, ["names", "scan", str(cli_ws), "S01E01"])
    assert f"`fukikae registry alias {cli_ws} {c.id} NABARU`" in r.output


def test_show_lists_split_keys_and_typo_suggestion(cli_ws: Path):
    _run_names(cli_ws, "S01E01")
    r = runner.invoke(app, ["names", "show", str(cli_ws), "S01E01"])
    assert r.exit_code == 0, r.output
    assert "S01E01: names full from target file S01E01.en.ass" in r.output
    assert "coverage: 31 of 31 dialogue lines named (100%), 3 keys" in r.output
    keys = [ln for ln in r.output.splitlines() if ln.startswith("  nebaru ")]
    assert len(keys) == 1 and "SPEAKER_00~nebaru" in keys[0] and keys[0].endswith("created")
    assert "registry alias" in r.output and "NABARU" in r.output


def test_the_characters_stage_persists_how_each_key_went(cli_ws: Path):
    """names-report.json: created on the first episode, known on the next; written only when
    names were used, and never a stage output."""
    _run_names(cli_ws, "S01E01", "S01E02")
    ws = Workspace(cli_ws)
    first = read_json(names_report_path(ws.work_dir("S01E01")), NamesReport)
    second = read_json(names_report_path(ws.work_dir("S01E02")), NamesReport)
    # NABARU's 3 lines are 6 s, over min_new_seconds: the typo makes a character, which is
    # why scan and show suggest the alias
    assert {d.key: d.how for d in first.keys} == {
        "nebaru": "created", "lucy": "created", "nabaru": "created"
    }  # fmt: skip
    how = {d.key: d.how for d in second.keys}
    assert how["lucy"] == "known" and how["happy"] == "created"
    assert how["nebaru"] == "known"  # 2 lines here, but E01 made it a character
    ep = ws.find_episode("S01E01")
    assert names_report_path(ep.work) not in CharactersStage().outputs(ep)
    (cli_ws / "fukikae.toml").write_text(
        (cli_ws / "fukikae.toml").read_text() + '\n[names]\nuse = "off"\n'
    )
    _run_names(cli_ws, "S01E01")
    assert not names_report_path(ws.work_dir("S01E01")).exists()


def test_show_without_names_says_why(cli_ws: Path):
    (cli_ws / "fukikae.toml").write_text(
        (cli_ws / "fukikae.toml").read_text() + '\n[names]\nuse = "off"\n'
    )
    _run_names(cli_ws, "S01E01")
    r = runner.invoke(app, ["names", "show", str(cli_ws), "S01E01"])
    assert r.exit_code == 0, r.output
    assert r.output.splitlines() == ['S01E01: names off from no track; [names] use = "off"']


def test_show_needs_the_names_stage(cli_ws: Path):
    r = runner.invoke(app, ["names", "show", str(cli_ws), "S01E01"])
    assert r.exit_code == 1 and "run the names stage" in r.output


def test_show_says_when_assignments_are_older_than_the_names(cli_ws: Path):
    _run_names(cli_ws, "S01E01")
    ws = Workspace(cli_ws)
    cfg = load_config(cli_ws, user_config=None)
    side = cli_ws.parent / "media" / "S01E01.en.ass"
    side.write_text(side.read_text().replace(",LUCY,", ",HAPPY,", 2), encoding="utf-8")
    NamesStage(cfg).run(ws.find_episode("S01E01"), cfg)  # names refined after characters ran
    r = runner.invoke(app, ["names", "show", str(cli_ws), "S01E01"])
    assert r.exit_code == 0, r.output
    assert "older than its names" in r.output and "keys (" not in r.output


# -- show: merge hints, cards and unnamed exemplars, on hand-made episodes -----------------


def _merge_ws(tmp_path: Path):
    """S01E14 of tests.cards.test_auto, with character 1 (unnamed) keeping S0's remainder
    while S0's named speech went to Natsu (character 2, 15 s) and Wendy (character 3, 5 s)."""
    from tests.cards.test_auto import NAMED_EP, NATSU_NAMES, NATSU_TURNS, _named_episode

    root = tmp_path / "ws"
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    (media_dir / f"{NAMED_EP}.mkv").write_bytes(b"")
    (root / "fukikae.toml").write_text(
        f'target_language = "en"\nsource_language = "ja"\nsource = "{media_dir}"\n'
    )
    reg = Registry(Workspace(root).registry_path)
    ids = [reg.create_actor_with_character(np.ones(4, np.float32))[1] for _ in range(3)]
    reg.name_character(ids[1], "Natsu", "natsu", "en")
    reg.name_character(ids[2], "Wendy", "wendy", "en")
    clusters = [
        {"cluster": "S0~natsu", "character_id": ids[1], "name": "natsu", "source": "name"},
        {"cluster": "S0", "character_id": ids[0]},
        {"cluster": "S0~wendy", "character_id": ids[2], "name": "wendy", "source": "name"},
    ]
    _named_episode(root, NAMED_EP, NATSU_TURNS, clusters, names=NATSU_NAMES,
                   excluded=[(40.0, 45.0)])  # fmt: skip
    return root, reg, ids


def test_show_suggests_merging_a_nameless_leftover_into_the_named_character(tmp_path: Path):
    root, reg, (leftover, natsu, _) = _merge_ws(tmp_path)
    actors = {c.id: c.actor_id for c in reg.characters()}
    reg.close()
    r = runner.invoke(app, ["names", "show", str(root), "S01E14"])
    assert r.exit_code == 0, r.output
    (hint,) = [ln for ln in r.output.splitlines() if "registry merge" in ln]
    assert f"character {leftover} (unnamed) kept S0's unnamed remainder (5.0 s)" in hint
    assert f"went to character {natsu} (Natsu) (20.0 of 25.0 s)" in hint
    assert f"`fukikae registry merge {root} {actors[natsu]} {actors[leftover]}`" in hint


def test_show_flags_stale_clone_cards_and_unnamed_exemplars(tmp_path: Path):
    from fukikae.cards.store import CardStore
    from tests.cards.test_auto import NAMED_EP, _auto_clone

    root, reg, (leftover, natsu, _) = _merge_ws(tmp_path)
    store = CardStore(root)
    _auto_clone(reg, store, natsu, [(NAMED_EP, 30.0, 35.0)], by="assign")
    natsu_actor = next(c.actor_id for c in reg.characters() if c.id == natsu)
    reg.add_exemplars(natsu_actor, "S01E10", "SPEAKER_03", np.ones((2, 4), np.float32), 4.0)
    reg.close()
    r = runner.invoke(app, ["names", "show", str(root), NAMED_EP])
    assert r.exit_code == 0, r.output
    out = r.output
    assert "clone cards:" in out
    assert f"character {natsu} (Natsu)'s card" in out and "now named WENDY" in out
    assert "`fukikae voices clone` replaces it" in out
    assert f"character {natsu} (Natsu) holds exemplars from S01E10, run without names" in out


def test_scan_cache_refreshes_when_the_source_is_replaced_in_place(cli_ws: Path, monkeypatch):
    """F8: the scan cache stamps the source like the stage's track copies, inode and ctime
    included, so a source replaced with the same size and mtime is extracted again."""
    import os
    import shutil

    from fukikae.names import detect
    from fukikae.names.cli import _scan_tracks

    calls: list[Path] = []

    def extract(_src, wanted):
        for _, dest in wanted:
            dest.write_text("x")
            calls.append(dest)
        return [d for _, d in wanted]

    monkeypatch.setattr(media, "extract_subtitles", extract)
    ws = Workspace(cli_ws)
    ep = ws.find_episode("S01E01")
    ep.source.write_bytes(b"0123456789")
    track = media.SubTrack(index=3, codec="ass", language="en", title=None, forced=False)
    choice = detect.TrackChoice(file=None, track=track, language="en", how="target")
    _scan_tracks(ep, ws, [choice])
    _scan_tracks(ep, ws, [choice])
    assert len(calls) == 1
    st = ep.source.stat()
    copy = ep.source.with_suffix(".new")
    shutil.copyfile(ep.source, copy)
    os.replace(copy, ep.source)  # a new inode, same size
    os.utime(ep.source, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime
    _scan_tracks(ep, ws, [choice])
    assert len(calls) == 2


def _merge_hints(root: Path) -> list[str]:
    r = runner.invoke(app, ["names", "show", str(root), "S01E14"])
    assert r.exit_code == 0, r.output
    return [ln for ln in r.output.splitlines() if "registry merge" in ln]


def test_the_merge_hint_folds_the_leftover_into_the_named_character(tmp_path: Path):
    """Running the hint's command keeps the named character (Natsu) and folds the leftover
    into it: KEEP is Natsu's actor, DROP the leftover's."""
    root, reg, (leftover, natsu, _) = _merge_ws(tmp_path)
    reg.close()
    (hint,) = _merge_hints(root)
    keep, drop = hint.split("`fukikae registry merge ")[1].split("`")[0].split()[1:]
    r = runner.invoke(app, ["registry", "merge", str(root), keep, drop])
    assert r.exit_code == 0, r.output
    with Registry(Workspace(root).registry_path) as reg:
        chars = {c.id: c for c in reg.characters()}
    assert leftover not in chars and chars[natsu].name == "Natsu"


def test_no_merge_hint_for_a_remainder_under_min_new_seconds(tmp_path: Path):
    """A remainder shorter than [names] min_new_seconds (5 s) is too little voice to say two
    characters are one."""
    root, reg, _ = _merge_ws(tmp_path)
    reg.close()
    (root / "fukikae.toml").write_text(
        (root / "fukikae.toml").read_text() + "[names]\nmin_new_seconds = 5.5\n"
    )
    assert _merge_hints(root) == []


def test_no_merge_hint_when_both_characters_are_one_actor(tmp_path: Path):
    import sqlite3

    root, reg, (leftover, natsu, _) = _merge_ws(tmp_path)
    actors = {c.id: c.actor_id for c in reg.characters()}
    reg.close()
    con = sqlite3.connect(Workspace(root).registry_path)
    with con:
        con.execute("UPDATE characters SET actor_id = ? WHERE id = ?", (actors[natsu], leftover))
    con.close()
    assert _merge_hints(root) == []


def test_show_marks_report_entries_the_registry_or_assignments_no_longer_hold(cli_ws: Path):
    """F7: names-report.json is written by the characters stage; a later `registry merge`
    folds NABARU's character into NEBARU's and remaps the assignments, so the report's
    character for nabaru no longer exists and show marks the entry stale."""
    _run_names(cli_ws, "S01E01")
    with Registry(Workspace(cli_ws).registry_path) as reg:
        neb = reg.character_for_key("nebaru")
        nab = reg.character_for_key("nabaru")
    assert neb is not None and nab is not None
    r = runner.invoke(app, ["registry", "merge", str(cli_ws), str(neb.actor_id), str(nab.actor_id)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["names", "show", str(cli_ws), "S01E01"])
    assert r.exit_code == 0, r.output
    (line,) = [ln for ln in r.output.splitlines() if ln.startswith("  nabaru ")]
    assert f"stale: character {nab.id} is gone from the registry" in line
    assert f"the assignments give {', '.join(['SPEAKER_00~nabaru'])} to character {neb.id}" in line
    assert "fukikae run" in line
    (ok,) = [ln for ln in r.output.splitlines() if ln.startswith("  lucy ")]
    assert "stale" not in ok


def test_show_lists_clone_cards_under_the_floor(tmp_path: Path, monkeypatch):
    """D32: `names show` prints `floor_lines` for the episode's characters with the clone
    cards it flags."""
    from tests.cards.test_auto import NAMED_EP

    root, reg, _ = _merge_ws(tmp_path)
    reg.close()
    seen: dict = {}

    def lines(_ws, _reg, _store, characters, _language, **_):
        seen["characters"] = sorted(characters)
        return ["character 9's card x (auto) is below the clone floor (1 clip and 6.3 s)"]

    monkeypatch.setattr("fukikae.cards.auto.floor_lines", lines)
    r = runner.invoke(app, ["names", "show", str(root), NAMED_EP])
    assert r.exit_code == 0, r.output
    assert "clone cards:" in r.output
    assert "character 9's card x (auto) is below the clone floor" in r.output
    assert seen["characters"]


def test_show_appends_the_floor_to_a_kept_stale_cards_line(tmp_path: Path):
    from fukikae.cards.store import CardStore
    from tests.cards.test_auto import NAMED_EP, _auto_clone

    root, reg, (_, natsu, _) = _merge_ws(tmp_path)
    card = _auto_clone(reg, CardStore(root), natsu, [(NAMED_EP, 30.0, 35.0)], locked=True)
    reg.close()
    r = runner.invoke(app, ["names", "show", str(root), NAMED_EP])
    assert r.exit_code == 0, r.output
    (line,) = [ln for ln in r.output.splitlines() if "clone floor" in ln or "now named" in ln]
    assert line.strip() == (
        f"character {natsu} (Natsu)'s card {card} (auto) was cloned from turns now named "
        "WENDY; `fukikae voices clone` replaces it; below the clone floor (1 clip and 5.0 s, "
        "under the floor of 2 clips and 10 s)"
    )
