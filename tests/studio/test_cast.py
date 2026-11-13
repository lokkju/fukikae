"""Cast priors from a hand-written ``<ws>/cast.toml`` (Task 22, decision 10): read strictly,
carried by the season index as suggestions with their source, and applied only through an
operator decision (``gender_by = "operator"``). Never a voice (D41), never names evidence."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fukikae.registry.store import Registry
from fukikae.studio.apply import apply_decision
from fukikae.studio.cast import (
    CastEntry,
    CastError,
    cast_gender_decision,
    queue_header,
    read_cast,
)
from tests.studio.test_season import _build, _entry, ws  # noqa: F401 (fixture)

SRC = Path(__file__).resolve().parents[2] / "src" / "fukikae"


def _write(root: Path, text: str) -> Path:
    path = root / "cast.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_cast_file_is_empty(tmp_path):
    assert read_cast(tmp_path) == []


def test_read_cast_validates_genders(tmp_path):
    _write(tmp_path, '[[cast]]\nname = "Lucy"\ngender = "f"\n\n[[cast]]\nname = "Happy"\n')
    got = read_cast(tmp_path)
    assert [(e.name, e.gender) for e in got] == [("Lucy", "f"), ("Happy", None)]
    for bad in ('"female"', '"F"', '"x"', "1", '""'):
        path = _write(tmp_path, f'[[cast]]\nname = "Lucy"\n\n[[cast]]\nname = "Natsu"\n'
                                f"gender = {bad}\n")  # fmt: skip
        with pytest.raises(CastError) as e:
            read_cast(tmp_path)
        msg = str(e.value)
        assert str(path) in msg and "line 4" in msg and "gender" in msg and '"f" or "m"' in msg


def test_entries_carry_aliases_external_ids_and_their_line(tmp_path):
    _write(tmp_path, '# Fairy Tail\n\n[[cast]]\nname = "Natsu Dragneel"\ngender = "m"\n'
                     'external_id = 12345\naliases = ["Natsu", "Salamander"]\n')  # fmt: skip
    (e,) = read_cast(tmp_path)
    assert e == CastEntry(name="Natsu Dragneel", gender="m", external_id="12345",
                          aliases=["Natsu", "Salamander"], line=3)  # fmt: skip


def test_unknown_keys_fail_naming_the_file_and_line(tmp_path):
    path = _write(tmp_path, '[[cast]]\nname = "Lucy"\n\n[[cast]]\nname = "Natsu"\nrole = "x"\n')
    with pytest.raises(CastError, match=r"line 4.*unknown key\(s\): role") as e:
        read_cast(tmp_path)
    assert str(path) in str(e.value)
    _write(tmp_path, 'series = "Fairy Tail"\n[[cast]]\nname = "Lucy"\n')
    with pytest.raises(CastError, match=r"unknown key\(s\): series"):
        read_cast(tmp_path)
    _write(tmp_path, 'cast = "Lucy"\n')
    with pytest.raises(CastError, match=r"\[\[cast\]\]"):
        read_cast(tmp_path)


@pytest.mark.parametrize("key", ["voice", "card", "stock", "voice_id", "clone"])
def test_a_cast_entry_never_names_a_voice(tmp_path, key):
    """D41: every character is cloned from its own speech; cast.toml names no voice."""
    _write(tmp_path, f'[[cast]]\nname = "Lucy"\n{key} = "af_heart"\n')
    with pytest.raises(CastError, match=r"D41") as e:
        read_cast(tmp_path)
    assert "line 1" in str(e.value)


def test_names_must_key_someone_and_be_unique(tmp_path):
    _write(tmp_path, '[[cast]]\ngender = "f"\n')
    with pytest.raises(CastError, match=r"line 1.*no name"):
        read_cast(tmp_path)
    _write(tmp_path, '[[cast]]\nname = "   "\n')
    with pytest.raises(CastError, match=r"names nobody"):
        read_cast(tmp_path)
    _write(tmp_path, '[[cast]]\nname = "Lucy"\n\n[[cast]]\nname = "LUCY"\n')
    with pytest.raises(CastError, match=r"line 4.*'LUCY'.*line 1"):
        read_cast(tmp_path)
    _write(tmp_path, '[[cast]]\nname = "Natsu"\n\n[[cast]]\nname = "Gray"\naliases = ["natsu"]\n')
    with pytest.raises(CastError, match=r"line 4.*'natsu'.*line 1"):
        read_cast(tmp_path)
    _write(tmp_path, '[[cast]]\nname = "A"\nexternal_id = "7"\n\n'
                     '[[cast]]\nname = "B"\nexternal_id = 7\n')  # fmt: skip
    with pytest.raises(CastError, match=r"line 5.*external_id '7'.*line 1"):
        read_cast(tmp_path)
    _write(tmp_path, '[[cast]]\nname = "A"\naliases = "B"\n')
    with pytest.raises(CastError, match=r"aliases must be a list"):
        read_cast(tmp_path)


def test_a_syntax_error_names_the_file(tmp_path):
    path = _write(tmp_path, '[[cast]]\nname = "Lucy\n')
    with pytest.raises(CastError) as e:
        read_cast(tmp_path)
    assert str(path) in str(e.value) and "line 2" in str(e.value)


# -- the season index --------------------------------------------------------------------------

CAST = """\
[[cast]]
name = "Lucy Heartfilia"
gender = "f"
aliases = ["Lucy"]

[[cast]]
name = "Natsu"
gender = "m"

[[cast]]
name = "Erza"
gender = "f"
"""


def test_season_doc_carries_the_cast(ws):  # noqa: F811
    _write(ws.root, CAST)
    doc = _build(ws)
    assert [e.name for e in doc.cast] == ["Lucy Heartfilia", "Natsu", "Erza"]
    assert doc.cast_error is None
    lucy, natsu = _entry(doc, 1).cast, _entry(doc, 2).cast
    assert lucy is not None and natsu is not None
    assert (lucy.entry, lucy.by, lucy.gender, lucy.source) == (0, "alias", "f", "cast.toml line 1")
    assert (natsu.entry, natsu.by, natsu.gender) == (1, "name", "m")
    assert _entry(doc, 3).cast is None  # unnamed: the page offers the cast names instead
    assert queue_header(doc) == "cast lists 3 speaking roles; this season has 5 characters"


def test_the_queue_header_counts_no_ignored_character(ws):  # noqa: F811
    _write(ws.root, CAST)
    with Registry(ws.registry_path) as reg:
        reg.set_character_kind(5, "ignored")
    doc = _build(ws)
    assert queue_header(doc) == "cast lists 3 speaking roles; this season has 4 characters"
    (ws.root / "cast.toml").unlink()
    assert queue_header(_build(ws)) is None


def test_an_external_id_match_wins(ws):  # noqa: F811
    with Registry(ws.registry_path) as reg:
        reg.set_character_external_id(3, "tmdb:77")
    _write(ws.root, CAST + '\n[[cast]]\nname = "Happy"\nexternal_id = "tmdb:77"\n')
    m = _entry(_build(ws), 3).cast
    assert m is not None and (m.entry, m.by, m.gender) == (3, "external_id", None)


def test_cast_toml_joins_the_index_inputs(ws):  # noqa: F811
    from fukikae.config import load_config
    from fukikae.studio.season import inputs_digest

    cfg = load_config(ws.root, user_config=None)
    before = inputs_digest(ws, cfg)
    _write(ws.root, CAST)
    with_cast = inputs_digest(ws, cfg)
    assert with_cast != before
    _write(ws.root, CAST.replace('"Erza"', '"Erza Scarlet"'))
    assert inputs_digest(ws, cfg) != with_cast


def test_a_broken_cast_file_is_reported_by_the_index_not_fatal(ws):  # noqa: F811
    _write(ws.root, '[[cast]]\nname = "Lucy"\nvoice = "af_heart"\n')
    doc = _build(ws)
    assert doc.cast == [] and all(c.cast is None for c in doc.characters)
    assert doc.cast_error is not None and "D41" in doc.cast_error
    assert len(doc.characters) == 5


def test_cast_gender_is_never_applied_without_a_decision(ws):  # noqa: F811
    _write(ws.root, CAST)
    doc = _build(ws)
    with Registry(ws.registry_path) as reg:
        assert all(c.gender is None and c.gender_by is None for c in reg.characters())
        aliases = reg.aliases()
    lucy = _entry(doc, 1)
    d = cast_gender_decision(lucy)  # what the page's one click posts
    assert (d.op, d.character, d.gender, d.digest) == ("gender", 1, "f", lucy.digest)
    apply_decision(ws, d)
    with Registry(ws.registry_path) as reg:
        c = next(c for c in reg.characters() if c.id == 1)
        assert (c.gender, c.gender_by) == ("f", "operator")
        assert next(c for c in reg.characters() if c.id == 2).gender is None
        assert reg.aliases() == aliases  # cast aliases are never registry aliases
    after = _entry(_build(ws), 1).cast
    assert after is not None and after.gender is None  # stored now: nothing left to offer


def test_an_operator_gender_is_never_offered_over(ws):  # noqa: F811
    _write(ws.root, CAST)
    with Registry(ws.registry_path) as reg:
        reg.set_character_gender(1, "m", by="operator")
        reg.set_character_gender(2, "f", by="cast")
    doc = _build(ws)
    lucy, natsu = _entry(doc, 1), _entry(doc, 2)
    assert lucy.cast is not None and lucy.cast.gender is None  # the operator's m stands
    with pytest.raises(ValueError, match="nothing to offer"):
        cast_gender_decision(lucy)
    assert natsu.cast is not None and natsu.cast.gender == "m"  # below the operator only


def test_the_cast_never_changes_names_evidence_or_registry_rows(ws):  # noqa: F811
    import sqlite3

    def dump() -> list[str]:
        con = sqlite3.connect(ws.registry_path)
        try:
            return list(con.iterdump())
        finally:
            con.close()

    plain = _build(ws)
    before = dump()
    _write(ws.root, CAST)
    with_cast = _build(ws)
    assert dump() == before
    for a, b in zip(plain.characters, with_cast.characters, strict=True):
        assert (a.names, a.suggestions, a.flags, a.name) == (b.names, b.suggestions, b.flags,
                                                            b.name)  # fmt: skip
    assert plain.queue == with_cast.queue


def test_only_the_studio_reads_the_workspace_cast_file():
    """Cast names never reach the names stage, identity or eval (held-out episodes stay
    blind): only the studio's season index reads `<ws>/cast.toml`."""
    allowed = {SRC / "studio" / "cast.py", SRC / "studio" / "season.py",
               SRC / "studio" / "model.py", SRC / "cards" / "packs.py"}  # fmt: skip
    readers = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"studio\.cast\b|read_cast|cast\.toml", text):
            readers.add(path)
    assert readers <= allowed, sorted(str(p.relative_to(SRC)) for p in readers - allowed)
    packs = (SRC / "cards" / "packs.py").read_text(encoding="utf-8")
    assert "studio.cast" not in packs  # its cast.toml is the pack's own file
