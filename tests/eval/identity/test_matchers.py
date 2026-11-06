"""Matcher variants on synthetic embeddings: knn, multi-centroid, lda and joint against the
centroid baseline, and the harness's simulate driving a matcher that sees every episode."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from fukikae.eval.identity.segments import Segment
from fukikae.eval.identity.simulate import simulate
from fukikae.registry.matchers import (
    MATCHERS,
    CentroidMatcher,
    JointMatcher,
    KnnMatcher,
    LdaMatcher,
    MultiCentroidMatcher,
    RegistryState,
    make_matcher,
)

DIM = 8
E = np.eye(DIM, dtype=np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def noisy(base: np.ndarray, n: int, rng: np.random.Generator, sd: float = 0.08) -> np.ndarray:
    return np.vstack([unit(base + rng.normal(0, sd, DIM)) for _ in range(n)])


def segs_for(
    episode: str, labels: Sequence[str | None], cluster: Sequence[str | None] | None = None
) -> list[Segment]:
    """Segments of 2 s laid end to end, 0.5 s apart."""
    out, t = [], 0.0
    for i, lab in enumerate(labels):
        c = None if cluster is None else cluster[i]
        out.append(Segment(episode, t, t + 2.0, lab, c))
        t += 2.5
    return out


# ---- registry: matchers are pluggable by name with options ----


def test_matchers_are_registered_by_name_with_options():
    assert {"centroid", "knn", "multi-centroid", "lda", "joint"} <= set(MATCHERS)
    m = make_matcher("knn", threshold=0.6, min_new_seconds=20.0, k=3)
    assert isinstance(m, KnnMatcher) and m.k == 3 and m.threshold == 0.6
    assert isinstance(make_matcher("multi-centroid", max_centroids=2), MultiCentroidMatcher)
    assert make_matcher("lda", dims=16).dims == 16
    assert make_matcher("lda", dims="all").dims is None
    assert isinstance(make_matcher("joint", threshold=0.5), JointMatcher)
    with pytest.raises(TypeError):
        make_matcher("knn", bogus=1)
    with pytest.raises(ValueError, match="k"):
        KnnMatcher(k=0)
    with pytest.raises(ValueError, match="max_centroids"):
        MultiCentroidMatcher(max_centroids=0)
    with pytest.raises(ValueError, match="dims"):
        LdaMatcher(dims="some")


# ---- a bimodal speaker: knn and multi-centroid keep one id, centroid splits ----


def bimodal_series() -> tuple[dict[str, list[Segment]], dict[str, np.ndarray]]:
    """A speaks in three registers (e0, e1, e2) in E01, all one diarization cluster; in E02
    only the e2 register. B (e5) is a different speaker in both. A's mean in E01 has cosine
    ~0.58 with the e2 register, under the 0.6 threshold."""
    rng = np.random.default_rng(7)
    a1 = np.vstack([noisy(E[0], 12, rng), noisy(E[1], 12, rng), noisy(E[2], 12, rng)])
    b1 = noisy(E[5], 15, rng)
    a2 = noisy(E[2], 15, rng)
    b2 = noisy(E[5], 15, rng)
    segs = {
        "E01": segs_for("E01", ["A"] * 36 + ["B"] * 15, ["S0"] * 36 + ["S1"] * 15),
        "E02": segs_for("E02", ["A"] * 15 + ["B"] * 15, ["S0"] * 15 + ["S1"] * 15),
    }
    return segs, {"E01": np.vstack([a1, b1]), "E02": np.vstack([a2, b2])}


def ids_by_label(
    assign: dict[tuple[str, int], int | None], segs: dict[str, list[Segment]]
) -> dict[tuple[str, str | None], set[int | None]]:
    out: dict[tuple[str, str | None], set[int | None]] = {}
    for (ep, i), cid in assign.items():
        out.setdefault((ep, segs[ep][i].label), set()).add(cid)
    return out


def test_centroid_splits_the_bimodal_speaker():
    segs, embs = bimodal_series()
    got = ids_by_label(simulate(["E01", "E02"], CentroidMatcher(), segs, embs), segs)
    assert got[("E01", "A")] != got[("E02", "A")]  # the e2 register became a new character


@pytest.mark.parametrize(
    "matcher",
    [KnnMatcher(k=5), KnnMatcher(k=3), MultiCentroidMatcher(max_centroids=3)],
    ids=["knn5", "knn3", "multi3"],
)
def test_knn_and_multi_centroid_keep_the_bimodal_speaker(matcher):
    segs, embs = bimodal_series()
    got = ids_by_label(simulate(["E01", "E02"], matcher, segs, embs), segs)
    assert got[("E01", "A")] == got[("E02", "A")] and None not in got[("E02", "A")]
    assert got[("E01", "B")] == got[("E02", "B")]
    assert got[("E02", "A")] != got[("E02", "B")]


def test_knn_votes_across_actors():
    """With k larger than the nearest actor's rows the vote still goes to that actor."""
    st = RegistryState()
    rng = np.random.default_rng(1)
    st.create(noisy(E[0], 3, rng))  # actor 1, 3 rows near e0
    st.create(noisy(unit(E[0] + E[3]), 20, rng))  # actor 2, 20 rows between e0 and e3
    m = KnnMatcher(k=5, min_new_seconds=0.0)
    seg = segs_for("E02", [None] * 4, ["S0"] * 4)
    ids = m.fit_episode(st, "E02", noisy(E[0], 4, rng), seg)
    assert set(ids) == {1}


def test_multi_centroid_with_one_centroid_is_the_centroid_matcher():
    segs, embs = bimodal_series()
    a = simulate(["E01", "E02"], MultiCentroidMatcher(max_centroids=1), segs, embs)
    b = simulate(["E01", "E02"], CentroidMatcher(), segs, embs)
    assert a == b


# ---- lda: fit on earlier episodes' labels, separates close voices ----


def close_voices() -> tuple[dict[str, list[Segment]], dict[str, np.ndarray]]:
    """A and B differ only along e1 (+-0.15), with large nuisance variation on e2..e4: their
    means have cosine ~0.95. E01 has them as two labeled diarization clusters; E02 as
    unclustered (subtitle-like) segments, which centroid groups at distance 0.4."""
    rng = np.random.default_rng(3)

    def voice(sign: float, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            v = E[0] + sign * 0.15 * E[1]
            v = v + rng.normal(0, 0.01, DIM) * E[1]
            v = v + rng.normal(0, 0.25, DIM) * (E[2] + E[3] + E[4])
            rows.append(unit(v))
        return np.vstack(rows)

    segs = {
        "E01": segs_for("E01", ["A"] * 15 + ["B"] * 15, ["S0"] * 15 + ["S1"] * 15),
        "E02": segs_for("E02", ["A"] * 12 + ["B"] * 12),
    }
    return segs, {
        "E01": np.vstack([voice(1, 15), voice(-1, 15)]),
        "E02": np.vstack([voice(1, 12), voice(-1, 12)]),
    }


def test_centroid_merges_the_close_voices():
    segs, embs = close_voices()
    got = ids_by_label(simulate(["E01", "E02"], CentroidMatcher(), segs, embs), segs)
    assert got[("E02", "A")] == got[("E02", "B")]


@pytest.mark.parametrize("dims", [None, 1])
def test_lda_separates_the_close_voices(dims):
    segs, embs = close_voices()
    got = ids_by_label(simulate(["E01", "E02"], LdaMatcher(dims=dims), segs, embs), segs)
    assert got[("E02", "A")] == got[("E01", "A")]
    assert got[("E02", "B")] == got[("E01", "B")]
    assert got[("E02", "A")] != got[("E02", "B")]


def test_lda_never_reads_the_episode_it_resolves():
    """E01 unlabeled: nothing earlier to fit on, so E02 resolves as centroid would, although
    E02's own segments carry labels."""
    segs, embs = close_voices()
    segs["E01"] = [Segment(s.episode, s.start, s.end, None, s.cluster) for s in segs["E01"]]
    lda = simulate(["E01", "E02"], LdaMatcher(), segs, embs)
    cen = simulate(["E01", "E02"], CentroidMatcher(), segs, embs)
    assert lda == cen
    m = LdaMatcher()
    st = RegistryState()
    m.fit_episode(st, "E01", embs["E01"], segs["E01"])
    with pytest.raises(ValueError, match="already"):
        m.fit_episode(st, "E01", embs["E01"], segs["E01"])


def test_lda_first_episode_is_centroid():
    segs, embs = close_voices()
    a = LdaMatcher().fit_episode(RegistryState(), "E01", embs["E01"], segs["E01"])
    b = CentroidMatcher().fit_episode(RegistryState(), "E01", embs["E01"], segs["E01"])
    assert a == b


# ---- joint: all episodes at once, cannot-link for overlapping segments ----


@dataclass(frozen=True)
class Seg:
    episode: str
    start: float
    end: float
    label: str | None = None
    cluster: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def test_joint_never_links_overlapping_segments():
    rng = np.random.default_rng(5)
    a = noisy(E[0], 10, rng, sd=0.02)
    # E01: A at 0-30 s in 3 s pieces, and one segment at 1-4 s (overlapping A's first) whose
    # embedding is exactly A's first: same voice by the numbers, different speaker by time.
    s1 = [Seg("E01", 3.0 * i, 3.0 * i + 3.0) for i in range(10)] + [Seg("E01", 1.0, 4.0)]
    e1 = np.vstack([a, a[:1]])
    s2 = [Seg("E02", 3.0 * i, 3.0 * i + 3.0) for i in range(10)]
    e2 = noisy(E[0], 10, rng, sd=0.02)
    m = JointMatcher(threshold=0.6, min_new_seconds=0.0)
    out = m.fit_all(["E01", "E02"], {"E01": e1, "E02": e2}, {"E01": s1, "E02": s2})
    assert out["E01"][10] is not None
    assert out["E01"][10] not in {out["E01"][0], out["E01"][1]}  # it overlaps both


def test_joint_links_a_voice_across_episodes():
    rng = np.random.default_rng(6)
    s1 = [Seg("E01", 3.0 * i, 3.0 * i + 3.0) for i in range(10)]
    s2 = [Seg("E02", 3.0 * i, 3.0 * i + 3.0) for i in range(10)]
    e1 = np.vstack([noisy(E[0], 5, rng, 0.02), noisy(E[4], 5, rng, 0.02)])
    e2 = np.vstack([noisy(E[4], 5, rng, 0.02), noisy(E[0], 5, rng, 0.02)])
    out = JointMatcher(min_new_seconds=0.0).fit_all(
        ["E01", "E02"], {"E01": e1, "E02": e2}, {"E01": s1, "E02": s2}
    )
    assert out["E01"] == [1] * 5 + [2] * 5
    assert out["E02"] == [2] * 5 + [1] * 5


def test_joint_cannot_link_survives_merges():
    """x overlaps a1 only and is a copy of a4. x and a4 merge first; from then on their group
    may not join a1's, although a4 alone would have: the constraint travels with x."""
    rng = np.random.default_rng(9)
    a = noisy(E[0], 5, rng, sd=0.02)
    segs = [Seg("E01", 10.0 * i, 10.0 * i + 3.0) for i in range(5)] + [Seg("E01", 1.0, 2.0)]
    embs = np.vstack([a, a[3:4]])
    out = JointMatcher(min_new_seconds=0.0).fit_all(["E01"], {"E01": embs}, {"E01": segs})["E01"]
    assert out[5] == out[3]
    assert out[0] != out[3]
    without_x = JointMatcher(min_new_seconds=0.0).fit_all(["E01"], {"E01": a}, {"E01": segs[:5]})[
        "E01"
    ]
    assert len(set(without_x)) == 1


def test_joint_short_clusters_are_extras_and_ids_follow_first_appearance():
    rng = np.random.default_rng(2)
    segs = [Seg("E01", 0.0, 12.0), Seg("E01", 20.0, 32.0), Seg("E01", 40.0, 42.0)]
    embs = np.vstack([noisy(E[1], 2, rng, sd=0.01), noisy(E[6], 1, rng)])
    out = JointMatcher(min_new_seconds=20.0).fit_all(["E01"], {"E01": embs}, {"E01": segs})
    assert out["E01"] == [1, 1, None]


def test_joint_seeds_from_diarization_clusters():
    """seed="clusters": an episode's diarization cluster starts as one group, so two rows of
    one cluster stay together although their embeddings disagree."""
    rng = np.random.default_rng(4)
    segs = [Seg("E01", 0.0, 15.0, cluster="S0"), Seg("E01", 20.0, 35.0, cluster="S0")]
    embs = np.vstack([noisy(E[1], 1, rng, 0.01), noisy(E[6], 1, rng, 0.01)])
    seeded = JointMatcher(min_new_seconds=0.0, seed="clusters")
    single = JointMatcher(min_new_seconds=0.0, seed="segments")
    assert len(set(seeded.fit_all(["E01"], {"E01": embs}, {"E01": segs})["E01"])) == 1
    assert len(set(single.fit_all(["E01"], {"E01": embs}, {"E01": segs})["E01"])) == 2
    with pytest.raises(ValueError, match="seed"):
        JointMatcher(seed="nope")


def test_simulate_drives_a_joint_matcher():
    segs, embs = bimodal_series()
    m = JointMatcher(threshold=0.5, min_new_seconds=0.0, seed="segments")
    got = simulate(["E01", "E02"], m, segs, embs)
    assert set(got) == {(ep, i) for ep in segs for i in range(len(segs[ep]))}
    by = ids_by_label(got, segs)
    assert by[("E01", "B")] == by[("E02", "B")]
    with pytest.raises(ValueError, match="state"):
        simulate(["E01", "E02"], m, segs, embs, state=RegistryState())


# ---- score scale: noisy rows, comparable to the centroid matcher's mean-vs-mean cosine ----


def noisy_voice(base: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rows whose pairwise cosine is ~0.5 while their mean points along `base`."""
    return np.vstack([unit(base + rng.normal(0, 0.45, DIM)) for _ in range(n)])


def test_knn_scores_on_the_centroid_scale():
    """knn's score is the cosine of the cluster mean with the local centroid (normalized
    mean) of the winner's voted exemplar rows, the centroid matcher's mean-vs-mean scale,
    not the mean row-to-row similarity, which noise pulls far lower."""
    rng = np.random.default_rng(11)
    st = RegistryState()
    st.create(noisy_voice(E[0], 40, rng))
    st.create(noisy_voice(E[5], 40, rng))
    rows = noisy_voice(E[0], 20, rng)
    m = KnnMatcher(k=5)
    m._prepare(st)  # noqa: SLF001
    mean = unit(rows.mean(axis=0))
    got = m._best(rows, mean)  # noqa: SLF001
    assert got is not None
    aid, score = got
    ex = np.vstack([*st.exemplars[1], *st.exemplars[2]])  # actor 1's rows first (40)
    sims = rows @ ex.T
    voted = sorted({int(j) for r in range(len(rows)) for j in np.argsort(-sims[r])[:5] if j < 40})
    assert aid == 1
    assert score == pytest.approx(float(mean @ unit(ex[voted].mean(axis=0))), abs=1e-5)
    assert score > float(np.mean(np.sort(sims, axis=1)[:, -5:])) + 0.1


def test_joint_centroid_linkage_links_noisy_rows_of_one_voice():
    rng = np.random.default_rng(12)
    segs = {
        "E01": [Seg("E01", 3.0 * i, 3.0 * i + 3.0, cluster="S0") for i in range(20)],
        "E02": [Seg("E02", 3.0 * i, 3.0 * i + 3.0, cluster="S0") for i in range(20)],
    }
    embs = {"E01": noisy_voice(E[0], 20, rng), "E02": noisy_voice(E[0], 20, rng)}
    cen = JointMatcher(threshold=0.6, min_new_seconds=0.0).fit_all(["E01", "E02"], embs, segs)
    assert set(cen["E01"]) == set(cen["E02"]) == {1}
    avg = JointMatcher(threshold=0.6, min_new_seconds=0.0, linkage="average").fit_all(
        ["E01", "E02"], embs, segs
    )
    assert set(avg["E01"]) != set(avg["E02"])
    with pytest.raises(ValueError, match="linkage"):
        JointMatcher(linkage="single")


# ---- fix round 1 ----


def test_centroid_shares_the_scoring_flow():
    """One fit_episode flow for every per-episode matcher; centroid is its centroid scorer."""
    from fukikae.registry import matchers as mod

    for cls in (CentroidMatcher, KnnMatcher, MultiCentroidMatcher, LdaMatcher):
        assert issubclass(cls, mod._ScoringMatcher)  # noqa: SLF001
        assert "fit_episode" not in cls.__dict__


def test_joint_episode_cannot_link_keeps_clusters_of_one_episode_apart():
    """Two non-overlapping segments of different diarization clusters in one episode, same
    voice: never one group with episode_cannot_link (the default); one group without it."""
    rng = np.random.default_rng(13)
    v = noisy(E[2], 2, rng, sd=0.01)
    segs = [Seg("E01", 0.0, 15.0, cluster="S0"), Seg("E01", 30.0, 45.0, cluster="S1")]
    for seed in ("clusters", "segments"):
        on = JointMatcher(min_new_seconds=0.0, seed=seed).fit_all(
            ["E01"], {"E01": v}, {"E01": segs}
        )
        assert on["E01"][0] != on["E01"][1]
        off = JointMatcher(min_new_seconds=0.0, seed=seed, episode_cannot_link=False).fit_all(
            ["E01"], {"E01": v}, {"E01": segs}
        )
        assert off["E01"][0] == off["E01"][1]


def test_joint_episode_cannot_link_still_links_across_episodes():
    rng = np.random.default_rng(14)
    segs = {
        "E01": [Seg("E01", 0.0, 15.0, cluster="S0")],
        "E02": [Seg("E02", 0.0, 15.0, cluster="S3")],
    }
    embs = {"E01": noisy(E[2], 1, rng, 0.01), "E02": noisy(E[2], 1, rng, 0.01)}
    out = JointMatcher(min_new_seconds=0.0).fit_all(["E01", "E02"], embs, segs)
    assert out["E01"] == out["E02"] == [1]


# ---- Ruling R11: unclustered segments group by centroid linkage ----


def noisy_subtitle_series() -> tuple[dict[str, list[Segment]], dict[str, np.ndarray]]:
    """Two voices as unclustered 2 s segments (subtitle-like) in two episodes. Single rows are
    noisy: the mean pairwise cosine within a voice is ~0.52, under the 0.6 threshold, while
    a voice's mean row sits far above it."""
    rng = np.random.default_rng(5)
    dim = 16
    eye = np.eye(dim)

    def voice(i: int, n: int) -> np.ndarray:
        return np.vstack([unit(eye[i] + rng.normal(0, 0.25, dim)) for _ in range(n)])

    segs, embs = {}, {}
    for ep in ("E01", "E02"):
        segs[ep] = segs_for(ep, ["A"] * 12 + ["B"] * 12)
        embs[ep] = np.vstack([voice(0, 12), voice(1, 12)]).astype(np.float32)
    return segs, embs


def test_subtitle_segmentation_under_centroid_no_longer_zeroes_out():
    """Average linkage left every noisy segment alone (2 s, under min_new_seconds): all
    extras. Centroid linkage groups each voice (an outlier row may stay alone) and links it
    across episodes."""
    from collections import Counter

    segs, embs = noisy_subtitle_series()
    assign = simulate(["E01", "E02"], CentroidMatcher(), segs, embs)
    modal: dict[tuple[str, str | None], int | None] = {}
    for key in [(ep, lab) for ep in ("E01", "E02") for lab in ("A", "B")]:
        ids = Counter(cid for (ep, i), cid in assign.items() if (ep, segs[ep][i].label) == key)
        modal[key], n = ids.most_common(1)[0]
        assert n >= 7  # most of the 12
    assert None not in modal.values()
    assert modal[("E01", "A")] != modal[("E01", "B")]
    assert modal[("E02", "A")] == modal[("E01", "A")]
    assert modal[("E02", "B")] == modal[("E01", "B")]


def test_group_segments_uses_centroid_linkage():
    from fukikae.registry.matchers import group_segments

    segs, embs = noisy_subtitle_series()
    groups = sorted(
        (sorted(idx) for _, idx in group_segments(embs["E01"], segs["E01"], 0.4)),
        key=len,
        reverse=True,
    )
    assert set(groups[0]) <= set(range(12)) or set(groups[0]) <= set(range(12, 24))
    assert len(groups[0]) >= 10 and len(groups[1]) >= 10
    assert set(groups[0]).isdisjoint(groups[1])
    assert {groups[0][0] < 12, groups[1][0] < 12} == {True, False}
