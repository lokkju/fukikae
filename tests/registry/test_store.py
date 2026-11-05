import sqlite3
from pathlib import Path

import numpy as np
import pytest

from fukikae.registry.store import Registry


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
    assert not a.locked and ch.voice is None


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
    con = sqlite3.connect(path)
    con.execute("UPDATE characters SET voice = ? WHERE id = ?", ('{"v": 1}', character_id))
    con.commit()
    con.close()


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
