"""Stored per-series projections and the LDA fit the characters stage and the harness share."""

import numpy as np
import pytest

from fukikae.config import Config
from fukikae.registry.matchers import LdaMatcher
from fukikae.registry.projection import Projection, fit_lda, series_of
from fukikae.registry.store import Registry


def test_save_and_load_projection_round_trips_and_replaces(tmp_path):
    path = tmp_path / "r.sqlite"
    m1 = np.arange(12, dtype=np.float64).reshape(4, 3)
    with Registry(path) as reg:
        assert reg.load_projection("show", "lda") is None
        reg.save_projection("show", "lda", m1, np.zeros(4))
    with Registry(path) as reg:
        got = reg.load_projection("show", "lda")
        assert got is not None
        assert got.matrix.shape == (4, 3) and np.array_equal(got.matrix, m1)
        assert np.array_equal(got.mean, np.zeros(4)) and got.created
        reg.save_projection("show", "lda", np.ones((4, 2)), np.ones(4))
        reg.save_projection("other", "lda", np.ones((4, 1)), np.ones(4))
        again = reg.load_projection("show", "lda")
        assert again is not None and again.matrix.shape == (4, 2)
        n = reg._con.execute("SELECT COUNT(*) FROM projections WHERE series='show'").fetchone()
        assert n[0] == 1  # replaced, not added


def test_fit_lda_needs_two_labels_of_two_rows():
    x = np.eye(4)[:3]
    assert fit_lda(x, ["A", "B", "C"]) is None
    assert fit_lda(np.vstack([x, x]), ["A", "A", "A", "A", "A", "A"]) is None


def test_fit_lda_matches_the_harness_matcher_space():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 1, (10, 6)) + i for i in range(3)])
    y = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    p = fit_lda(x, y)
    assert isinstance(p, Projection) and p.matrix.shape == (6, 2)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    ref = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=2).fit(x, y)
    probe = rng.normal(0, 1, (5, 6))
    want = ref.transform(probe)
    want = want / np.linalg.norm(want, axis=1, keepdims=True)
    assert np.allclose(p.apply(probe), want, atol=1e-5)
    m = LdaMatcher()
    m._train_x, m._train_y = list(x), y
    m._lda = m._fit()
    assert np.allclose(m._project(probe), want, atol=1e-5)
    one = fit_lda(x, y, dims=1)
    assert one is not None and one.matrix.shape == (6, 1)


def test_projection_apply_is_unit_rows_and_checks_dim():
    p = Projection(np.eye(3)[:, :2], np.zeros(3))
    out = p.apply(np.array([[3.0, 4.0, 9.0]]))
    assert np.allclose(out, [[0.6, 0.8]])
    with pytest.raises(ValueError, match="dimension"):
        p.apply(np.ones((1, 5)))


def test_series_of_reads_the_workspace_setting():
    assert series_of(Config()) == "default"
    assert series_of(Config(raw={"series": "Fairy Tail"})) == "Fairy Tail"
