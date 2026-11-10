import sqlite3
from pathlib import Path

import numpy as np
import pytest

from fukikae.registry.store import (
    AliasConflictError,
    CardAssignment,
    ProtectedAssignmentError,
    Registry,
)


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def norm_mean(rows: np.ndarray) -> np.ndarray:
    m = rows.astype(np.float32).mean(axis=0)
    return m / np.linalg.norm(m)


@pytest.fixture
def reg(tmp_path: Path):
    with Registry(tmp_path / "nested" / "dir" / "registry.sqlite") as r:
        yield r


def test_create_actor_with_character(reg: Registry):
    c = unit(1, 0, 0)
    aid, cid = reg.create_actor_with_character(c)
    (a,) = reg.actors()
    (ch,) = reg.characters()
    assert a.id == aid and ch.id == cid and ch.actor_id == aid
    assert a.centroid is not None and np.allclose(a.centroid, c)
    assert a.centroid.dtype == np.float32
    assert not a.locked and ch.external_id is None


def test_centroid_none_without_exemplars(reg: Registry):
    aid, _ = reg.create_actor_with_character(unit(1, 0))
    reg.set_actor(aid, name="x")
    assert reg.actors()[0].name == "x"


def test_merge_moves_and_recomputes(reg: Registry):
    a1, c1 = reg.create_actor_with_character(unit(1, 0, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1, 0))
    e1 = np.stack([unit(1, 0.1, 0), unit(1, 0.2, 0)])
    e2 = np.stack([unit(0.1, 1, 0)])
    reg.add_exemplars(a1, "e1", "c0", e1, 10.0)
    reg.add_exemplars(a2, "e2", "c0", e2, 5.0)
    assert reg.merge_actors(a1, a2) == {c2: c1}
    actors = reg.actors()
    assert [a.id for a in actors] == [a1]
    # v1: one character per actor; DROP's characters fold into KEEP's lowest-id one
    assert [(c.id, c.actor_id) for c in reg.characters()] == [(c1, a1)]
    assert reg.exemplar_count(a1) == 3
    expected = norm_mean(np.concatenate([e1, e2]))
    assert actors[0].centroid is not None
    assert np.allclose(actors[0].centroid, expected, atol=1e-6)


def test_merge_errors(reg: Registry):
    a1, _ = reg.create_actor_with_character(unit(1, 0))
    with pytest.raises(ValueError):
        reg.merge_actors(a1, a1)
    with pytest.raises(KeyError):
        reg.merge_actors(a1, 999)
    with pytest.raises(KeyError):
        reg.merge_actors(999, a1)


def test_locked_centroid_unchanged(reg: Registry):
    c = unit(1, 0, 0)
    aid, _ = reg.create_actor_with_character(c)
    reg.lock_actor(aid, True)
    reg.add_exemplars(aid, "e1", "c0", np.stack([unit(0, 1, 0)]), 3.0)
    reg.recompute_centroid(aid)
    a = reg.actors()[0]
    assert a.locked and a.centroid is not None
    assert np.allclose(a.centroid, c)


def test_delete_episode_recomputes(reg: Registry):
    aid, _ = reg.create_actor_with_character(unit(1, 0, 0))
    ea = np.stack([unit(1, 0, 0)])
    eb = np.stack([unit(0, 1, 0)])
    reg.add_exemplars(aid, "e1", "c0", ea, 3.0)
    reg.add_exemplars(aid, "e2", "c0", eb, 3.0)
    reg.recompute_centroid(aid)
    reg.delete_episode("e2")
    a = reg.actors()[0]
    assert reg.exemplar_count(aid) == 1
    assert a.centroid is not None and np.allclose(a.centroid, ea[0])


def test_delete_last_episode_keeps_centroid(reg: Registry):
    aid, _ = reg.create_actor_with_character(unit(1, 0, 0))
    reg.set_actor(aid, name="Ann")  # protected from pruning; keeps its last centroid
    reg.add_exemplars(aid, "e1", "c0", np.stack([unit(0, 1, 0)]), 3.0)
    reg.recompute_centroid(aid)
    reg.delete_episode("e1")
    a = reg.actors()[0]
    assert a.centroid is not None and np.allclose(a.centroid, unit(0, 1, 0))


def test_delete_episode_skips_locked(reg: Registry):
    c = unit(1, 0, 0)
    aid, _ = reg.create_actor_with_character(c)
    reg.add_exemplars(aid, "e1", "c0", np.stack([unit(0, 1, 0)]), 3.0)
    reg.add_exemplars(aid, "e2", "c0", np.stack([unit(0, 0, 1)]), 3.0)
    reg.lock_actor(aid, True)
    reg.delete_episode("e1")
    got = reg.actors()[0].centroid
    assert got is not None and np.allclose(got, c)


def test_rename_and_persist(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    with Registry(p) as r:
        _, cid = r.create_actor_with_character(unit(1, 0))
        r.rename_character(cid, "Mira")
        with pytest.raises(KeyError):
            r.rename_character(999, "x")
    with Registry(p) as r:
        assert r.characters()[0].name == "Mira"


def test_newer_schema_rejected(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    Registry(p).close()
    con = sqlite3.connect(p)
    con.execute("UPDATE schema_version SET version = 99")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        Registry(p)
    # single row stays single on reopen of a good db
    p2 = tmp_path / "r2.sqlite"
    Registry(p2).close()
    Registry(p2).close()
    con = sqlite3.connect(p2)
    assert con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    con.close()


def test_foreign_keys_on(reg: Registry):
    with pytest.raises(sqlite3.IntegrityError):
        reg.add_exemplars(999, "e1", "c0", np.stack([unit(1, 0)]), 1.0)


def _set_voice(path: Path, character_id: int) -> None:
    with Registry(path) as reg:
        reg.assign_card(character_id, "en", "0" * 16, by="assign", locked=True)


def test_delete_episode_returns_orphans_and_prune_spares_protected(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    with Registry(p) as reg:
        ids = {}
        for name in ("only", "shared", "locked", "named", "voiced", "charnamed"):
            ids[name], _ = reg.create_actor_with_character(unit(1, 0, 0))
            reg.add_exemplars(ids[name], "e1", name, np.stack([unit(1, 0, 0)]), 3.0)
        reg.add_exemplars(ids["shared"], "e2", "x", np.stack([unit(0, 1, 0)]), 3.0)
        reg.lock_actor(ids["locked"], True)
        reg.set_actor(ids["named"], name="Ann")
        voiced_cid = next(c.id for c in reg.characters() if c.actor_id == ids["voiced"])
        named_cid = next(c.id for c in reg.characters() if c.actor_id == ids["charnamed"])
        reg.rename_character(named_cid, "Hero")
    _set_voice(p, voiced_cid)
    with Registry(p) as reg:
        orphans = reg.delete_episode("e1")
        assert sorted(orphans) == sorted(ids[n] for n in ids if n != "shared")
        deleted = reg.prune_actors(orphans)
        assert deleted == [ids["only"]]
        left = {a.id for a in reg.actors()}
        assert left == {ids[n] for n in ids if n != "only"}
        assert ids["only"] not in {c.actor_id for c in reg.characters()}


def test_prune_actors_skips_actor_with_exemplars(reg: Registry):
    aid, _ = reg.create_actor_with_character(unit(1, 0))
    reg.add_exemplars(aid, "e1", "c0", np.stack([unit(1, 0)]), 3.0)
    assert reg.prune_actors([aid]) == []
    assert [a.id for a in reg.actors()] == [aid]


def test_transaction_rolls_back_everything(reg: Registry):
    with pytest.raises(RuntimeError), reg.transaction():
        aid, _ = reg.create_actor_with_character(unit(1, 0))
        reg.add_exemplars(aid, "e1", "c0", np.stack([unit(1, 0)]), 3.0)
        raise RuntimeError("boom")
    assert reg.actors() == [] and reg.characters() == []


def test_deleted_ids_are_never_reused(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    with Registry(p) as reg:
        a1, c1 = reg.create_actor_with_character(unit(1, 0))
        a2, c2 = reg.create_actor_with_character(unit(0, 1))
        assert reg.prune_actors([a2]) == [a2]
    with Registry(p) as reg:  # survives reopening
        a3, c3 = reg.create_actor_with_character(unit(0, 1))
        assert (a3, c3) == (a2 + 1, c2 + 1)
        reg.add_exemplars(a1, "e1", "S0", np.stack([unit(1, 0)]), 3.0)
        a4, c4 = reg.split_cluster(a1, "e1", "S0")
        assert (a4, c4) == (a3 + 1, c3 + 1)


def test_exemplar_rows_of_mixed_dimensions_point_to_reembed(tmp_path):
    import pytest

    reg = Registry(tmp_path / "r.db")
    aid, _ = reg.create_actor_with_character(np.ones(3, dtype=np.float32))
    reg.add_exemplars(aid, "S01E01", "S0", np.ones((2, 3), dtype=np.float32), 9.0)
    reg.add_exemplars(aid, "S01E02", "S0", np.ones((2, 4), dtype=np.float32), 9.0)
    with pytest.raises(ValueError, match=r"actor 1 .*3, 4.*registry reembed"):
        reg.exemplar_rows()
    reg.close()


# --- schema 2: per-language card assignments and history ----------------------------------

A16, B16, C16 = "a" * 16, "b" * 16, "c" * 16

_V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE actors (id INTEGER PRIMARY KEY, name TEXT, external_id TEXT, centroid BLOB,
    locked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE characters (id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL REFERENCES actors(id),
    name TEXT, voice TEXT, locked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE exemplars (id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL REFERENCES actors(id),
    episode TEXT NOT NULL, cluster TEXT, embedding BLOB NOT NULL, seconds REAL);
CREATE TABLE id_high_water (tbl TEXT PRIMARY KEY, last INTEGER NOT NULL);
CREATE TABLE projections (series TEXT NOT NULL, kind TEXT NOT NULL, matrix BLOB NOT NULL,
    mean BLOB NOT NULL, created TEXT NOT NULL, PRIMARY KEY (series, kind));
INSERT INTO schema_version(version) VALUES (1);
INSERT INTO actors(id, name) VALUES (1, NULL), (2, NULL);
INSERT INTO characters(id, actor_id, name, voice) VALUES
    (1, 1, NULL, '{"backend":"kokoro","voice":"af_heart"}'), (2, 2, 'Lucy', NULL);
INSERT INTO id_high_water(tbl, last) VALUES ('characters', 2), ('actors', 2);
"""


def _v1_file(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.executescript(_V1_SCHEMA)
    con.commit()
    con.close()
    return path


def test_v1_registry_migrates_voice_into_history(tmp_path: Path):
    p = _v1_file(tmp_path / "r.sqlite")
    with Registry(p) as reg:
        (h,) = reg.card_history(1, "en")
        assert h.n == 1 and h.by == "migrated" and h.card_id is None and not h.locked
        assert h.note is not None and "af_heart" in h.note
        assert reg.card_for(1, "en") is None
        assert reg.card_history(2, "en") == []
        assert [(c.id, c.name, c.external_id) for c in reg.characters()] == [
            (1, None, None),
            (2, "Lucy", None),
        ]
        # a migrated voice keeps the character curated, as a v1 voice did
        assert reg.protected_actors() == {1, 2}
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
    assert con.execute("SELECT voice FROM characters").fetchall() == [(None,), (None,)]
    con.close()
    with Registry(p) as reg:  # reopening migrates nothing twice
        assert len(reg.card_history(1, "en")) == 1


def test_failed_migration_leaves_the_v1_file_untouched(tmp_path: Path, monkeypatch):
    import fukikae.registry.store as store

    p = _v1_file(tmp_path / "r.sqlite")

    def boom(_self, con):  # noqa: ANN001, ANN202
        con.execute("ALTER TABLE characters ADD COLUMN external_id TEXT")
        raise RuntimeError("boom")

    monkeypatch.setattr(store.Registry, "_migrate_1_to_2", boom)
    with pytest.raises(RuntimeError, match="boom"):
        Registry(p)
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
    cols = [r[1] for r in con.execute("PRAGMA table_info(characters)")]
    assert "external_id" not in cols
    assert con.execute("SELECT voice FROM characters WHERE id = 1").fetchone()[0] is not None
    con.close()


def test_fresh_registry_is_schema_4(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    Registry(p).close()
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
    con.close()


def _cur(reg: Registry, cid: int, lang: str) -> CardAssignment:
    a = reg.card_for(cid, lang)
    assert a is not None
    return a


def _char(reg: Registry) -> int:
    return reg.create_actor_with_character(unit(1, 0))[1]


def test_assign_history_and_revert(reg: Registry):
    cid = _char(reg)
    first = reg.assign_card(cid, "en", A16, by="auto", locked=False)
    assert (first.n, first.card_id, first.by, first.locked) == (1, A16, "auto", False)
    reg.assign_card(cid, "en", B16, by="assign", locked=True)
    cur = reg.card_for(cid, "en")
    assert cur is not None and cur.card_id == B16 and cur.locked and cur.n == 2
    back = reg.revert_card(cid, "en", None)
    assert back.card_id == A16 and back.by == "revert" and back.n == 3
    assert [h.n for h in reg.card_history(cid, "en")] == [1, 2, 3]
    assert reg.card_for(cid, "en") == back
    again = reg.revert_card(cid, "en", None)  # toggles back, like `cd -`
    assert again.card_id == B16
    to1 = reg.revert_card(cid, "en", 1)
    assert to1.card_id == A16 and to1.n == 5


def test_revert_errors(reg: Registry, tmp_path: Path):
    cid = _char(reg)
    with pytest.raises(ValueError, match="no card"):
        reg.revert_card(cid, "en", None)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    with pytest.raises(ValueError, match="no earlier"):
        reg.revert_card(cid, "en", None)
    with pytest.raises(ValueError, match="no history row 7"):
        reg.revert_card(cid, "en", 7)
    p = _v1_file(tmp_path / "v1.sqlite")
    with Registry(p) as r1:
        r1.assign_card(1, "en", A16, by="assign", locked=True)
        with pytest.raises(ValueError, match="row 1 has no card"):
            r1.revert_card(1, "en", 1)


def test_assign_validates_inputs(reg: Registry):
    cid = _char(reg)
    with pytest.raises(ValueError, match="card id"):
        reg.assign_card(cid, "en", "nothex", by="assign", locked=True)
    with pytest.raises(ValueError, match="by"):
        reg.assign_card(cid, "en", A16, by="whim", locked=True)
    with pytest.raises(KeyError):
        reg.assign_card(999, "en", A16, by="assign", locked=True)
    reg.assign_card(cid, "en", A16, by="import:Fairy Tail EN", locked=False)
    assert _cur(reg, cid, "en").by == "import:Fairy Tail EN"


def test_languages_never_replace_each_other(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="assign", locked=True)
    reg.assign_card(cid, "es", C16, by="derive", locked=False)
    assert _cur(reg, cid, "en").card_id == A16
    assert _cur(reg, cid, "es").card_id == C16
    assert [h.n for h in reg.card_history(cid, "es")] == [1]
    reg.revert_card(cid, "en", 1)
    assert _cur(reg, cid, "es").card_id == C16


def test_lock_and_unlock_are_history_rows(reg: Registry):
    cid = _char(reg)
    with pytest.raises(ValueError, match="no card"):
        reg.lock_card(cid, "en", True)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.lock_card(cid, "en", True)
    cur = _cur(reg, cid, "en")
    assert cur.locked and cur.card_id == A16 and cur.by == "lock"
    reg.lock_card(cid, "en", False)
    assert not _cur(reg, cid, "en").locked
    assert [h.by for h in reg.card_history(cid, "en")] == ["auto", "lock", "unlock"]


def test_auto_only_character_is_prunable_and_its_rows_go(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    assert aid not in reg.protected_actors()
    assert reg.prune_actors([aid]) == [aid]
    assert reg.card_history(cid, "en") == [] and reg.card_for(cid, "en") is None


@pytest.mark.parametrize(
    ("how", "by", "locked"),
    [("assigned", "assign", True), ("locked auto", "auto", True), ("unlocked edit", "edit", False)],
)
def test_hand_assigned_character_is_protected_from_prune(reg: Registry, how, by, locked):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.assign_card(cid, "en", A16, by=by, locked=locked)
    assert aid in reg.protected_actors(), how
    assert reg.prune_actors([aid]) == [], how


def test_external_id_protects_and_round_trips(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.set_character_external_id(cid, "anidb:c123")
    assert reg.characters()[0].external_id == "anidb:c123"
    assert aid in reg.protected_actors() and reg.prune_actors([aid]) == []
    reg.set_character_external_id(cid, None)
    assert reg.characters()[0].external_id is None
    with pytest.raises(KeyError):
        reg.set_character_external_id(999, "x")


def test_merge_carries_card_rows_to_the_kept_character(reg: Registry):
    a1, c1 = reg.create_actor_with_character(unit(1, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1))
    reg.assign_card(c1, "en", A16, by="auto", locked=False)
    reg.assign_card(c2, "en", B16, by="assign", locked=True)  # curated beats auto
    reg.assign_card(c2, "es", C16, by="derive", locked=False)  # keep has no es card
    assert reg.merge_actors(a1, a2) == {c2: c1}
    assert _cur(reg, c1, "en").card_id == B16
    assert _cur(reg, c1, "es").card_id == C16
    assert {h.card_id for h in reg.card_history(c1, "en")} == {A16, B16}
    assert reg.card_history(c2, "en") == []


def test_merge_keeps_the_kept_characters_curated_card(reg: Registry):
    a1, c1 = reg.create_actor_with_character(unit(1, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1))
    reg.assign_card(c1, "en", A16, by="assign", locked=True)
    reg.assign_card(c2, "en", B16, by="assign", locked=True)
    reg.merge_actors(a1, a2)
    assert _cur(reg, c1, "en").card_id == A16


def test_character_by_ref_width_insensitive(reg: Registry):
    cid = _char(reg)
    reg.rename_character(cid, "Natsu")
    other = _char(reg)
    assert reg.character_by_ref("ＮＡＴＳＵ").id == cid
    assert reg.character_by_ref("natsu").id == cid
    assert reg.character_by_ref(str(other)).id == other
    assert reg.character_by_ref("２").id == other == 2
    with pytest.raises(ValueError, match="no character"):
        reg.character_by_ref("Gray")
    dup = _char(reg)
    reg.rename_character(dup, "NATSU")
    with pytest.raises(ValueError, match=rf"{cid}.*{dup}"):
        reg.character_by_ref("Natsu")


# --- ruling CD2: automatic assignments never replace curated cards ------------------------


def test_auto_refuses_to_replace_a_locked_card(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.lock_card(cid, "en", True)
    with pytest.raises(ProtectedAssignmentError, match="locked"):
        reg.assign_card(cid, "en", B16, by="auto", locked=False)
    assert _cur(reg, cid, "en").card_id == A16


def test_auto_and_import_refuse_to_replace_an_operator_card(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="assign", locked=False)
    with pytest.raises(ProtectedAssignmentError, match="operator-chosen"):
        reg.assign_card(cid, "en", B16, by="auto", locked=False)
    with pytest.raises(ProtectedAssignmentError):
        reg.assign_card(cid, "en", B16, by="import:Pack", locked=False)
    assert len(reg.card_history(cid, "en")) == 1
    done = reg.assign_card(cid, "en", B16, by="import:Pack", locked=False, override=True)
    assert done.card_id == B16 and done.chosen_by == "import:Pack"


def test_auto_replaces_an_auto_card_and_operators_replace_anything(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    assert reg.assign_card(cid, "en", B16, by="auto", locked=False).card_id == B16
    reg.lock_card(cid, "en", True)
    assert reg.assign_card(cid, "en", C16, by="assign", locked=True).card_id == C16


def test_auto_refuses_after_a_lock_then_unlock_of_an_operator_card(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="edit", locked=True)
    reg.lock_card(cid, "en", False)
    assert _cur(reg, cid, "en").chosen_by == "edit"
    with pytest.raises(ProtectedAssignmentError):
        reg.assign_card(cid, "en", B16, by="auto", locked=False)


def test_unlock_of_a_never_locked_auto_card_is_a_no_op(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.lock_card(cid, "en", False)
    assert [h.by for h in reg.card_history(cid, "en")] == ["auto"]
    assert aid not in reg.protected_actors()
    assert reg.prune_actors([aid]) == [aid]


def test_lock_twice_writes_one_row(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.lock_card(cid, "en", True)
    reg.lock_card(cid, "en", True)
    assert [h.by for h in reg.card_history(cid, "en")] == ["auto", "lock"]


def test_lock_then_unlock_of_an_auto_card_leaves_it_uncurated(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.lock_card(cid, "en", True)
    assert aid in reg.protected_actors()
    reg.lock_card(cid, "en", False)
    assert cid not in reg.curated_characters() and aid not in reg.protected_actors()
    assert reg.prune_actors([aid]) == [aid]


def test_unlocked_operator_card_stays_curated(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.assign_card(cid, "en", A16, by="assign", locked=False)
    reg.lock_card(cid, "en", True)
    reg.lock_card(cid, "en", False)
    assert cid in reg.curated_characters()
    assert reg.prune_actors([aid]) == []


def test_merged_history_is_in_time_order(reg: Registry, monkeypatch):
    import fukikae.registry.store as store

    times = iter(["2026-11-08T10:00:03.000000+00:00", "2026-11-08T10:00:01.000000+00:00"])
    monkeypatch.setattr(store, "_now", lambda: next(times))
    a1, c1 = reg.create_actor_with_character(unit(1, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1))
    reg.assign_card(c1, "en", A16, by="assign", locked=True)  # later, lower row id
    reg.assign_card(c2, "en", B16, by="assign", locked=True)  # earlier, higher row id
    notes: list[str] = []
    reg.merge_actors(a1, a2, report=notes)
    # rows keep their id order (n); by_time shows them by when they were assigned
    assert [h.card_id for h in reg.card_history(c1, "en")] == [A16, B16]
    assert [h.card_id for h in reg.card_history(c1, "en", by_time=True)] == [B16, A16]
    assert _cur(reg, c1, "en").card_id == A16 and _cur(reg, c1, "en").n == 1
    assert any(A16 in n and B16 in n and "stays in history" in n for n in notes)


def test_merge_external_ids(reg: Registry):
    a1, c1 = reg.create_actor_with_character(unit(1, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1))
    reg.set_character_external_id(c2, "anidb:2")
    reg.merge_actors(a1, a2)
    assert reg.characters()[0].external_id == "anidb:2"  # keep had none: takes drop's
    a3, c3 = reg.create_actor_with_character(unit(1, 1))
    reg.set_character_external_id(c3, "anidb:3")
    notes: list[str] = []
    reg.merge_actors(a1, a3, report=notes)
    assert reg.characters()[0].external_id == "anidb:2"  # both set: keep's wins
    assert any("anidb:2" in n and "anidb:3" in n for n in notes)


def test_split_gives_a_new_character_no_cards(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.add_exemplars(aid, "e1", "S0", np.stack([unit(1, 0)]), 3.0)
    reg.add_exemplars(aid, "e2", "S0", np.stack([unit(1, 0.1)]), 3.0)
    reg.assign_card(cid, "en", A16, by="assign", locked=True)
    _, new_cid = reg.split_cluster(aid, "e2", "S0")
    assert reg.card_languages(new_cid) == [] and reg.card_history(new_cid, "en") == []
    assert _cur(reg, cid, "en").card_id == A16


# --- ruling CD3: imported cards are operator acts ----------------------------------------


def test_auto_refuses_to_replace_an_imported_card(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="import:Pack", locked=False)
    with pytest.raises(ProtectedAssignmentError, match="import:Pack"):
        reg.assign_card(cid, "en", B16, by="auto", locked=False)
    assert _cur(reg, cid, "en").card_id == A16
    assert reg.assign_card(cid, "en", B16, by="auto", locked=False, override=True).card_id == B16


def test_import_replaces_an_unlocked_import(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="import:One", locked=False)
    done = reg.assign_card(cid, "en", B16, by="import:Two", locked=False)
    assert done.card_id == B16 and done.chosen_by == "import:Two"


def test_import_refuses_a_locked_import_without_override(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="import:One", locked=True)
    with pytest.raises(ProtectedAssignmentError, match="locked"):
        reg.assign_card(cid, "en", B16, by="import:Two", locked=False)
    assert reg.assign_card(cid, "en", B16, by="import:Two", locked=False, override=True)


def test_import_refuses_an_operator_card_without_override(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="assign", locked=False)
    with pytest.raises(ProtectedAssignmentError, match="operator-chosen"):
        reg.assign_card(cid, "en", B16, by="import:Two", locked=False)


def test_restock_is_not_curation_and_is_guarded_like_auto(reg: Registry):
    """Ruling CD9: a restocked card is not curated (restock is no operator act), restock never
    replaces a locked, operator-chosen or imported card without override, and auto may replace
    a restocked card exactly where it may replace an auto card: when unlocked."""
    a, b, c, d = "a" * 16, "b" * 16, "c" * 16, "d" * 16
    x = _char(reg)
    reg.assign_card(x, "en", a, by="auto", locked=False)
    cur = reg.assign_card(x, "en", b, by="restock", locked=False)
    assert cur.chosen_by == "restock" and x not in reg.curated_characters()
    assert reg.assign_card(x, "en", c, by="auto", locked=False).card_id == c
    reg.assign_card(x, "en", b, by="restock", locked=False)
    reg.lock_card(x, "en", True)
    for by in ("auto", "restock"):
        with pytest.raises(ProtectedAssignmentError, match="locked"):
            reg.assign_card(x, "en", d, by=by, locked=False)

    y = _char(reg)
    reg.assign_card(y, "en", a, by="assign", locked=False)
    with pytest.raises(ProtectedAssignmentError, match="operator-chosen"):
        reg.assign_card(y, "en", d, by="restock", locked=False)
    z = _char(reg)
    reg.assign_card(z, "en", a, by="import:pack", locked=False)
    with pytest.raises(ProtectedAssignmentError, match="imported"):
        reg.assign_card(z, "en", d, by="restock", locked=False)
    assert reg.assign_card(z, "en", d, by="restock", locked=False, override=True).card_id == d


def test_merge_with_lock_rows_on_both_sides_keeps_the_operator_card_curated(reg: Registry):
    """A2: after a merge interleaves rows, a lock/unlock row still says how its own card was
    chosen, so auto never replaces the operator's card."""
    a1, keep = reg.create_actor_with_character(unit(1, 0))
    a2, drop = reg.create_actor_with_character(unit(0, 1))
    reg.assign_card(drop, "en", B16, by="assign", locked=False)  # operator, earliest
    reg.assign_card(keep, "en", A16, by="auto", locked=False)
    reg.lock_card(keep, "en", True)
    reg.lock_card(drop, "en", True)
    reg.lock_card(keep, "en", False)
    reg.lock_card(drop, "en", False)
    reg.merge_actors(a1, a2)
    cur = _cur(reg, keep, "en")
    assert cur.card_id == B16 and cur.chosen_by == "assign" and not cur.locked
    with pytest.raises(ProtectedAssignmentError):
        reg.assign_card(keep, "en", C16, by="auto", locked=False)


def test_merge_does_not_over_protect_an_auto_card_after_lock_rows(reg: Registry):
    a1, keep = reg.create_actor_with_character(unit(1, 0))
    a2, drop = reg.create_actor_with_character(unit(0, 1))
    reg.assign_card(keep, "en", A16, by="assign", locked=False)
    reg.assign_card(drop, "en", B16, by="auto", locked=False)
    reg.lock_card(drop, "en", True)
    reg.lock_card(drop, "en", False)
    reg.merge_actors(a1, a2)
    # keep's curated card wins; drop's auto card (via its unlock row) reads as auto in history
    rows = reg.card_history(keep, "en")
    assert [h.chosen_by for h in rows if h.card_id == B16] == ["auto", "auto", "auto"]


def test_history_rows_are_numbered_in_id_order(reg: Registry, monkeypatch):
    """A9: a clock step backwards never reorders one character's rows."""
    import fukikae.registry.store as store

    times = iter(["2026-11-08T10:00:03.000000+00:00", "2026-11-08T10:00:01.000000+00:00"])
    monkeypatch.setattr(store, "_now", lambda: next(times, "2026-11-08T10:00:02.000000+00:00"))
    cid = _char(reg)
    reg.assign_card(cid, "en", A16, by="auto", locked=False)
    reg.assign_card(cid, "en", B16, by="assign", locked=True)
    assert [(h.n, h.card_id) for h in reg.card_history(cid, "en")] == [(1, A16), (2, B16)]
    assert _cur(reg, cid, "en").card_id == B16 and _cur(reg, cid, "en").n == 2
    assert reg.revert_card(cid, "en", None).card_id == A16


def test_v2_registry_migrates_to_3_with_chosen_by(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    with Registry(p) as reg:
        cid = _char(reg)
        reg.assign_card(cid, "en", A16, by="assign", locked=False)
        reg.lock_card(cid, "en", True)
    con = sqlite3.connect(p)
    con.execute("ALTER TABLE card_history DROP COLUMN chosen_by")
    con.execute("UPDATE schema_version SET version = 2")
    con.commit()
    con.close()
    with Registry(p) as reg:
        assert [h.chosen_by for h in reg.card_history(cid, "en")] == ["assign", "assign"]
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
    con.close()


def test_v2_migration_normalizes_language_keys_and_keeps_the_newest_on_collision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Schema 2 stored target languages in whatever case it was given; reads now normalize
    (``PT-br`` is ``pt-BR``), so the migration rewrites the stored keys the same way. Two
    current cards that become one key keep the newer assignment (the older stays in
    history), and the collision is logged."""
    import logging

    p = tmp_path / "r.sqlite"
    with Registry(p) as reg:
        cid = _char(reg)
        other = _char(reg)
        reg.assign_card(cid, "pt-BR", A16, by="assign", locked=False)
    con = sqlite3.connect(p)
    for char, lang, card in [(cid, "PT-br", B16), (other, "EN", C16)]:
        hid = con.execute(
            "INSERT INTO card_history(character_id, target_language, card_id, assigned, by,"
            " locked, note) VALUES (?, ?, ?, '2026-11-09T00:00:00+00:00', 'assign', 0, NULL)",
            (char, lang, card),
        ).lastrowid
        con.execute(
            "INSERT INTO card_assignments(character_id, target_language, card_id, history_id)"
            " VALUES (?, ?, ?, ?)",
            (char, lang, card, hid),
        )
    con.execute("ALTER TABLE card_history DROP COLUMN chosen_by")
    con.execute("UPDATE schema_version SET version = 2")
    con.commit()
    con.close()
    with caplog.at_level(logging.WARNING, logger="fukikae.registry.store"), Registry(p) as reg:
        assert _cur(reg, cid, "pt-BR").card_id == B16
        assert [h.card_id for h in reg.card_history(cid, "pt-BR")] == [A16, B16]
        assert reg.card_languages(cid) == ["pt-BR"]
        assert reg.card_languages(other) == ["en"]
        assert _cur(reg, other, "en").card_id == C16
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert f"character {cid}" in warning and "pt-BR" in warning
    assert B16 in warning and A16 in warning
    con = sqlite3.connect(p)
    langs = {r[0] for r in con.execute("SELECT target_language FROM card_history")}
    langs |= {r[0] for r in con.execute("SELECT target_language FROM card_assignments")}
    con.close()
    assert langs == {"pt-BR", "en"}


@pytest.mark.parametrize("bad", ["9" * 101, "a\x1bb", "   "])
def test_character_external_id_is_validated(reg: Registry, bad: str):
    cid = _char(reg)
    with pytest.raises(ValueError, match="external id"):
        reg.set_character_external_id(cid, bad)


def test_character_external_id_is_stripped(reg: Registry):
    cid = _char(reg)
    reg.set_character_external_id(cid, " 1234 ")
    assert reg.characters()[0].external_id == "1234"


def test_assignment_language_keys_are_case_normalized(reg: Registry):
    cid = _char(reg)
    reg.assign_card(cid, "EN", A16, by="auto", locked=False)
    assert _cur(reg, cid, "en").card_id == A16
    assert reg.card_languages(cid) == ["en"]
    assert reg.card_for(cid, "En") is not None


# --- schema 4: name provenance and character aliases -------------------------------------


def _downgrade_to_3(path: Path) -> None:
    """A schema-4 file made back into the schema-3 shape: no name_by, no alias table."""
    con = sqlite3.connect(path)
    con.execute("DROP TABLE character_aliases")
    con.execute("ALTER TABLE characters DROP COLUMN name_by")
    con.execute("UPDATE schema_version SET version = 3")
    con.commit()
    con.close()


def _named_v3(path: Path) -> dict[str, int]:
    """A schema-3 registry with operator names (the only way schema 3 set one), two of them
    folding to one key, a possession name, a two-speaker name and an unnamed character."""
    ids: dict[str, int] = {}
    with Registry(path) as reg:
        for name in ["Natsu", "ＮＡＴＳＵ", "MAKAROV (WRAITH)", "Natsu/Happy", None]:
            cid = _char(reg)
            ids[str(name)] = cid
            if name is not None:
                with reg._tx() as con:  # schema 3 had no aliases: set the bare column
                    con.execute("UPDATE characters SET name = ? WHERE id = ?", (name, cid))
    _downgrade_to_3(path)
    return ids


def _alias_rows(path: Path) -> list[tuple]:
    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT alias, character_id, by, language FROM character_aliases ORDER BY alias"
    ).fetchall()
    con.close()
    return rows


def test_v3_registry_migrates_names_to_operator_aliases(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    import logging

    p = tmp_path / "r.sqlite"
    ids = _named_v3(p)
    with caplog.at_level(logging.WARNING, logger="fukikae.registry.store"), Registry(p) as reg:
        by = {c.id: c.name_by for c in reg.characters()}
        assert by == {
            ids["Natsu"]: "operator",
            ids["ＮＡＴＳＵ"]: "operator",
            ids["MAKAROV (WRAITH)"]: "operator",
            ids["Natsu/Happy"]: "operator",
            ids["None"]: None,
        }
        natsu = reg.character_for_key("natsu")
        assert natsu is not None and natsu.id == ids["Natsu"]  # the first character wins
        wraith = reg.character_for_key("makarov (wraith)")  # D29 #12: a name of its own
        assert wraith is not None and wraith.id == ids["MAKAROV (WRAITH)"]
        assert reg.character_for_key("makarov") is None
    assert _alias_rows(p) == [
        ("makarov (wraith)", ids["MAKAROV (WRAITH)"], "operator", None),
        ("natsu", ids["Natsu"], "operator", None),
    ]  # a two-speaker name keys nobody
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert f"character {ids['ＮＡＴＳＵ']}" in warning and f"character {ids['Natsu']}" in warning
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
    con.close()
    with Registry(p) as reg:  # reopening migrates nothing twice
        assert len(reg.aliases()) == 2


def test_v3_to_4_migration_is_idempotent(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    _named_v3(p)
    with Registry(p) as reg:
        before = _alias_rows(p)
        with reg._tx() as con:
            reg._migrate_3_to_4(con)
        assert _alias_rows(p) == before
        assert {c.name_by for c in reg.characters()} == {"operator", None}


def test_v1_registry_migrates_to_4(tmp_path: Path):
    p = _v1_file(tmp_path / "r.sqlite")
    with Registry(p) as reg:
        assert [(c.id, c.name, c.name_by) for c in reg.characters()] == [
            (1, None, None),
            (2, "Lucy", "operator"),
        ]
        lucy = reg.character_for_key("lucy")
        assert lucy is not None and lucy.id == 2
        assert len(reg.card_history(1, "en")) == 1
    with Registry(p) as reg:
        assert [a.alias for a in reg.aliases()] == ["lucy"]


def test_v2_registry_migrates_to_4(tmp_path: Path):
    p = tmp_path / "r.sqlite"
    with Registry(p) as reg:
        cid = _char(reg)
        reg.assign_card(cid, "en", A16, by="assign", locked=False)
        with reg._tx() as con:
            con.execute("UPDATE characters SET name = 'Erza' WHERE id = ?", (cid,))
    _downgrade_to_3(p)
    con = sqlite3.connect(p)
    con.execute("ALTER TABLE card_history DROP COLUMN chosen_by")
    con.execute("UPDATE schema_version SET version = 2")
    con.commit()
    con.close()
    with Registry(p) as reg:
        assert [h.chosen_by for h in reg.card_history(cid, "en")] == ["assign"]
        assert reg.characters()[0].name_by == "operator"
        erza = reg.character_for_key("erza")
        assert erza is not None and erza.id == cid
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
    con.close()


def test_failed_3_to_4_migration_leaves_the_file_untouched(tmp_path: Path, monkeypatch):
    import fukikae.registry.store as store

    p = tmp_path / "r.sqlite"
    _named_v3(p)
    real = store.Registry._migrate_3_to_4

    def boom(self, con):  # noqa: ANN001, ANN202
        real(self, con)
        raise RuntimeError("boom")

    monkeypatch.setattr(store.Registry, "_migrate_3_to_4", boom)
    with pytest.raises(RuntimeError, match="boom"):
        Registry(p)
    con = sqlite3.connect(p)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
    assert "name_by" not in [r[1] for r in con.execute("PRAGMA table_info(characters)")]
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "character_aliases" not in tables
    con.close()


def test_subs_alias_never_replaces_an_operator_alias(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "natsu", by="operator")
    with pytest.raises(AliasConflictError):
        reg.add_alias(b, "natsu", by="subs")
    with pytest.raises(AliasConflictError):
        reg.add_alias(b, "natsu", by="subs", replace=True)
    assert reg.aliases(a)[0].by == "operator"


def test_subs_alias_never_replaces_a_subs_alias(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "nabaru", by="subs", language="en")
    with pytest.raises(AliasConflictError):
        reg.add_alias(b, "nabaru", by="subs")


def test_operator_alias_replaces_subs_alias(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "nabaru", by="subs", language="en")
    got = reg.add_alias(b, "nabaru", by="operator")
    assert got.character_id == b and got.by == "operator"
    hit = reg.character_for_key("nabaru")
    assert hit is not None and hit.id == b
    assert reg.aliases(a) == []


def test_operator_alias_needs_replace_over_another_operators(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "happy", by="operator")
    with pytest.raises(AliasConflictError, match="--replace"):
        reg.add_alias(b, "happy", by="operator")
    assert reg.add_alias(b, "happy", by="operator", replace=True).character_id == b


def test_alias_on_its_own_character_is_kept_or_raised_to_operator(reg: Registry):
    a = _char(reg)
    first = reg.add_alias(a, "lucy", by="subs", language="en")
    assert reg.add_alias(a, "lucy", by="subs") == first
    assert reg.add_alias(a, "lucy", by="operator").by == "operator"
    assert reg.add_alias(a, "lucy", by="subs").by == "operator"  # never lowered


def test_alias_may_not_be_another_characters_name(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.rename_character(a, "Gray")
    with reg._tx() as con:  # a name whose alias went elsewhere: the name still keys it
        con.execute("DELETE FROM character_aliases")
    for by in ("subs", "operator"):
        with pytest.raises(AliasConflictError, match="name"):
            reg.add_alias(b, "GRAY", by=by, replace=True)  # type: ignore[arg-type]


def test_alias_keys_are_folded_and_checked(reg: Registry):
    a = _char(reg)
    assert reg.add_alias(a, "ＮＡＴＳＵ", by="operator").alias == "natsu"
    with pytest.raises(ValueError):
        reg.add_alias(a, "x", by="robot")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        reg.add_alias(999, "x", by="operator")


def test_remove_alias(reg: Registry):
    a = _char(reg)
    reg.add_alias(a, "natsu", by="operator")
    reg.remove_alias("NATSU")
    assert reg.aliases() == [] and reg.character_for_key("natsu") is None
    with pytest.raises(KeyError):
        reg.remove_alias("natsu")


def test_character_for_key_falls_back_to_names_and_refuses_two(reg: Registry):
    a, b = _char(reg), _char(reg)
    with reg._tx() as con:
        con.execute("UPDATE characters SET name = 'Wendy' WHERE id IN (?, ?)", (a, b))
    with pytest.raises(ValueError, match=f"{a}.*{b}"):
        reg.character_for_key("wendy")
    with reg._tx() as con:
        con.execute("UPDATE characters SET name = NULL WHERE id = ?", (b,))
    hit = reg.character_for_key("wendy")
    assert hit is not None and hit.id == a


def test_name_character_sets_a_subs_name_and_alias(reg: Registry):
    a = _char(reg)
    reg.name_character(a, "Wendy", "wendy", "en")
    (c,) = reg.characters()
    assert (c.name, c.name_by) == ("Wendy", "subs")
    assert [(x.alias, x.by, x.language) for x in reg.aliases(a)] == [("wendy", "subs", "en")]


def test_name_character_refuses_an_operator_named_character(reg: Registry):
    a = _char(reg)
    reg.rename_character(a, "Natsu Dragneel")
    with pytest.raises(ValueError):
        reg.name_character(a, "Natsu", "natsu", "en")


def test_name_character_never_renames_a_subs_named_character(reg: Registry):
    a = _char(reg)
    reg.name_character(a, "Wendy", "wendy", "en")
    with pytest.raises(ValueError):
        reg.name_character(a, "Charle", "charle", "en")
    assert reg.characters()[0].name == "Wendy"


@pytest.mark.parametrize("curate", ["card", "external_id", "actor_lock"])
def test_name_character_refuses_a_curated_character(reg: Registry, curate: str):
    aid, a = reg.create_actor_with_character(unit(1, 0))
    if curate == "card":
        reg.assign_card(a, "en", A16, by="auto", locked=True)
    elif curate == "external_id":
        reg.set_character_external_id(a, "1234")
    else:
        reg.lock_actor(aid, True)
    with pytest.raises(ValueError):
        reg.name_character(a, "Wendy", "wendy", "en")
    assert reg.characters()[0].name is None and reg.aliases() == []


def test_name_character_refuses_a_key_another_character_holds(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "wendy", by="subs")
    with pytest.raises(AliasConflictError):
        reg.name_character(b, "Wendy", "wendy", "en")
    assert reg.characters()[1].name is None  # nothing half-done


def test_operator_rename_keeps_the_subs_key(reg: Registry):
    a = _char(reg)
    reg.name_character(a, "Natsu", "natsu", "en")
    assert reg.rename_character(a, "Natsu Dragneel") is None
    (c,) = reg.characters()
    assert (c.name, c.name_by) == ("Natsu Dragneel", "operator")
    for key in ("natsu", "natsu dragneel"):
        hit = reg.character_for_key(key)
        assert hit is not None and hit.id == a
    assert {(x.alias, x.by) for x in reg.aliases(a)} == {
        ("natsu", "subs"),
        ("natsu dragneel", "operator"),
    }


def test_operator_rename_skips_a_key_another_character_holds_by_operator(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.add_alias(a, "erza", by="operator")
    warning = reg.rename_character(b, "Erza")
    assert warning is not None and f"character {a}" in warning
    assert reg.characters()[1].name == "Erza"
    hit = reg.character_for_key("erza")
    assert hit is not None and hit.id == a


def test_operator_rename_moves_a_subs_alias_and_outranks_a_subs_name(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.name_character(a, "Erza", "erza", "en")  # subtitles named a
    assert reg.rename_character(b, "Erza") is None
    assert {(x.alias, x.character_id, x.by) for x in reg.aliases()} == {("erza", b, "operator")}
    names = [(c.name, c.name_by) for c in reg.characters()]
    assert names == [("Erza", "subs"), ("Erza", "operator")]
    hit = reg.character_for_key("erza")
    assert hit is not None and hit.id == b


def test_operator_rename_beside_an_operator_name_warns(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.rename_character(a, "Erza")
    warning = reg.rename_character(b, "ERZA")
    assert warning is not None and f"character {a}" in warning


def test_character_for_key_ranks_operator_alias_name_then_subs_alias_name(reg: Registry):
    c1, c2, c3, c4 = (_char(reg) for _ in range(4))
    with reg._tx() as con:
        con.execute("UPDATE characters SET name = 'Lucy', name_by = 'subs' WHERE id = ?", (c1,))
    assert reg.character_for_key("lucy").id == c1  # type: ignore[union-attr]
    with reg._tx() as con:  # (add_alias refuses a key that is another character's name)
        con.execute(
            "INSERT INTO character_aliases VALUES ('lucy', ?, 'subs', NULL, '2026-11-10')", (c2,)
        )
    assert reg.character_for_key("lucy").id == c2  # type: ignore[union-attr]
    with reg._tx() as con:  # a name from before schema 4 (NULL name_by) is the operator's
        con.execute("UPDATE characters SET name = 'LUCY', name_by = NULL WHERE id = ?", (c3,))
    assert reg.character_for_key("lucy").id == c3  # type: ignore[union-attr]
    with reg._tx() as con:
        con.execute("UPDATE character_aliases SET character_id = ?, by = 'operator'", (c4,))
    assert reg.character_for_key("lucy").id == c4  # type: ignore[union-attr]


def test_a_name_without_name_by_counts_as_curated(reg: Registry):
    a, b = _char(reg), _char(reg)
    with reg._tx() as con:
        con.execute("UPDATE characters SET name = 'Old', name_by = NULL WHERE id = ?", (a,))
        con.execute("UPDATE characters SET name = 'Subs', name_by = 'subs' WHERE id = ?", (b,))
    assert reg.curated_characters() == {a}


def test_rename_of_a_two_speaker_name_adds_no_alias(reg: Registry):
    a = _char(reg)
    assert reg.rename_character(a, "MAKAROV/WRAITH") is None
    assert reg.aliases() == [] and reg.characters()[0].name_by == "operator"


def test_merge_moves_aliases_and_prune_deletes_them(reg: Registry):
    a1, c1 = reg.create_actor_with_character(unit(1, 0))
    a2, c2 = reg.create_actor_with_character(unit(0, 1))
    reg.add_exemplars(a1, "e1", "c0", np.stack([unit(1, 0.1)]), 5.0)
    reg.add_exemplars(a2, "e1", "c1", np.stack([unit(0.1, 1)]), 5.0)
    reg.add_alias(c1, "nebaru", by="operator")
    reg.name_character(c2, "Nabaru", "nabaru", "en")
    assert reg.merge_actors(a1, a2) == {c2: c1}
    assert {(x.alias, x.character_id, x.by) for x in reg.aliases()} == {
        ("nebaru", c1, "operator"),
        ("nabaru", c1, "subs"),
    }
    hit = reg.character_for_key("nabaru")
    assert hit is not None and hit.id == c1
    a3, c3 = reg.create_actor_with_character(unit(1, 1))
    reg.name_character(c3, "Loke", "loke", "en")
    assert reg.prune_actors([a3]) == [a3]
    assert reg.character_for_key("loke") is None
    assert "loke" not in {x.alias for x in reg.aliases()}


def test_a_subs_named_orphan_is_not_curated_and_prunes_with_its_alias(reg: Registry):
    # controller ruling on D29 #2: only an operator name or an operator alias curates
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.name_character(cid, "Wendy", "wendy", "en")
    assert cid not in reg.curated_characters() and aid not in reg.protected_actors()
    assert reg.prune_actors([aid]) == [aid]
    assert reg.characters() == [] and reg.aliases() == []


def test_an_orphan_holding_only_an_operator_alias_is_kept(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.add_alias(cid, "charle", by="operator")
    assert cid in reg.curated_characters() and aid in reg.protected_actors()
    assert reg.prune_actors([aid]) == []
    assert [(x.alias, x.by) for x in reg.aliases(cid)] == [("charle", "operator")]
    with pytest.raises(ValueError, match="curated"):
        reg.name_character(cid, "Charle", "charle", "en")


def test_an_operator_named_orphan_is_kept(reg: Registry):
    aid, cid = reg.create_actor_with_character(unit(1, 0))
    reg.rename_character(cid, "Erza")
    assert cid in reg.curated_characters() and reg.prune_actors([aid]) == []


def test_rename_character_is_always_the_operator(reg: Registry):
    import inspect

    assert "by" not in inspect.signature(Registry.rename_character).parameters
    a = _char(reg)
    reg.name_character(a, "Natsu", "natsu", "en")
    reg.rename_character(a, "Natsu Dragneel")
    assert reg.characters()[0].name_by == "operator"


def test_no_subs_path_overwrites_an_operator_name(reg: Registry):
    a = _char(reg)
    reg.rename_character(a, "Natsu Dragneel")
    with pytest.raises(ValueError):
        reg.name_character(a, "Zeref", "zeref", "en")
    with pytest.raises(AliasConflictError):  # nor takes its key away
        reg.add_alias(_char(reg), "natsu dragneel", by="subs")
    (c, _) = reg.characters()
    assert (c.name, c.name_by) == ("Natsu Dragneel", "operator")
    assert [(x.alias, x.by) for x in reg.aliases(a)] == [("natsu dragneel", "operator")]


@pytest.mark.parametrize(
    ("written", "key"),
    [("MAKAROV(WRAITH)", "makarov (wraith)"), ("{\\i1}NATSU", "natsu"), ("WENDY/I", "wendy")],
)
def test_alias_methods_key_names_as_the_names_stage_does(reg: Registry, written: str, key: str):
    a = _char(reg)
    assert reg.add_alias(a, written, by="operator").alias == key
    for spelled in (written, key):
        hit = reg.character_for_key(spelled)
        assert hit is not None and hit.id == a
    reg.remove_alias(written)
    assert reg.aliases() == []


@pytest.mark.parametrize("nobody", ["MAKAROV/WRAITH", "NATSU & LUCY", "SIGN", "  "])
def test_alias_methods_refuse_a_name_that_keys_nobody(reg: Registry, nobody: str):
    a = _char(reg)
    with pytest.raises(ValueError, match="names nobody"):
        reg.add_alias(a, nobody, by="operator")
    with pytest.raises(ValueError, match="names nobody"):
        reg.character_for_key(nobody)
    with pytest.raises(ValueError, match="names nobody"):
        reg.remove_alias(nobody)


def test_ambiguous_key_hint_names_rename_and_merge(reg: Registry):
    a, b = _char(reg), _char(reg)
    with reg._tx() as con:
        con.execute("UPDATE characters SET name = 'Lucy' WHERE id IN (?, ?)", (a, b))
    with pytest.raises(ValueError, match="registry rename.*registry merge"):
        reg.character_for_key("lucy")


def test_lock_character_sets_the_flag_and_refuses_unknown(reg: Registry):
    a, b = _char(reg), _char(reg)
    reg.lock_character(b, True)
    assert [c.locked for c in reg.characters()] == [False, True]
    reg.lock_character(b, False)
    assert not reg.characters()[1].locked
    with pytest.raises(KeyError):
        reg.lock_character(99, True)
    assert a
