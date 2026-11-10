"""`fukikae eval tts-ab`: cue selection, blinding, key separation, loudness, the verdict."""

import re
import stat
import tomllib
import wave
from pathlib import Path

import numpy as np
import pytest

from fukikae.eval import ab

CANDS = ["kokoro", "indextts2", "cosyvoice3", "qwen3_tts", "qwen3_tts:xvector"]


def tone(path: Path, seconds: float, rate: int = 24000, amp: float = 0.25, freq: float = 440):
    n = max(1, round(seconds * rate))
    x = amp * np.sin(2 * np.pi * freq * np.arange(n) / rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


# ---- candidates and spec ------------------------------------------------------------------


def test_candidate_parses_backend_and_mode():
    c = ab.Candidate.parse("qwen3_tts:xvector")
    assert (c.backend, c.mode, c.name) == ("qwen3_tts", "xvector", "qwen3_tts:xvector")
    assert ab.Candidate.parse("kokoro").mode is None
    with pytest.raises(ValueError):
        ab.Candidate.parse("qwen3_tts:")


def test_spec_is_strict(tmp_path):
    spec = tmp_path / "spec.toml"
    spec.write_text(
        'seed = 7\n[[set]]\nname = "ft"\nworkspace = "/x"\nepisode = "S01E12"\n'
        'candidates = ["kokoro", "indextts2"]\n'
    )
    p = ab.load_spec(spec)
    assert p.seed == 7 and p.sets[0].name == "ft" and p.sets[0].cues == 6
    assert [c.name for c in p.sets[0].candidates] == ["kokoro", "indextts2"]
    spec.write_text(spec.read_text() + "typo = 1\n")
    with pytest.raises(ValueError, match="typo"):
        ab.load_spec(spec)


# ---- cue selection ------------------------------------------------------------------------


def _info(i: int, char: int, emotion: str | None, suspect: dict[str, str] | None = None):
    return ab.CueInfo(
        episode="S01E12", cue_id=f"c{i:04d}", character_id=char, emotion=emotion,
        suspect=suspect or {},
    )  # fmt: skip


def _pool() -> list[ab.CueInfo]:
    out = []
    i = 0
    for char in (1, 2, 3, 4):
        for k in range(8):
            out.append(_info(i, char, "angry" if k % 3 == 0 else "neutral"))
            i += 1
    out[5] = _info(5, 1, "neutral", {"indextts2": "too long"})
    out[20] = _info(20, 3, "sad", {"qwen3_tts": "WER > 0.5"})
    return out


def test_selection_spans_characters_mixes_emotion_and_keeps_suspects():
    picked = ab.select_cues(_pool(), count=6, characters=3, seed=1)
    assert len(picked) == 6
    assert len({p.character_id for p in picked}) >= 3
    emotional = [p for p in picked if p.emotional]
    assert emotional and len(emotional) < len(picked)
    flagged = {b for p in picked for b in p.suspect}
    assert flagged == {"indextts2", "qwen3_tts"}


def test_a_render_guard_flag_is_kept_beside_a_wer_flag_of_the_same_candidate():
    pool = _pool()
    pool[5] = _info(5, 1, "neutral", {"indextts2": "render guard: long"})
    pool[9] = _info(
        9, 2, "angry", {"indextts2": "eval tts WER > 0.5", "kokoro": "eval tts WER > 0.5"}
    )
    picked = ab.select_cues(pool, count=6, characters=3, seed=1)
    ids = {p.cue_id for p in picked}
    assert {"c0005", "c0009"} <= ids


def test_selection_is_deterministic_by_seed():
    a = ab.select_cues(_pool(), count=6, characters=3, seed=1)
    b = ab.select_cues(_pool(), count=6, characters=3, seed=1)
    assert [p.cue_id for p in a] == [p.cue_id for p in b]


# ---- blinding -----------------------------------------------------------------------------


def test_blind_order_is_seeded_and_varies_per_cue():
    one = ab.blind_order(CANDS, seed=3, slot="cue01")
    assert one == ab.blind_order(CANDS, seed=3, slot="cue01")
    assert sorted(one) == ["A", "B", "C", "D", "E"]
    assert sorted(one.values()) == sorted(CANDS)
    orders = {tuple(ab.blind_order(CANDS, seed=3, slot=f"cue{n:02d}").values()) for n in range(12)}
    assert len(orders) > 1


def _materials(tmp_path: Path) -> tuple[list[ab.CueMaterial], list[ab.SceneMaterial]]:
    episode_audio = tone(tmp_path / "src" / "audio.wav", 20.0, rate=48000, amp=0.5)
    english = tone(tmp_path / "src" / "en.wav", 20.0, rate=48000, amp=0.05, freq=300)
    cues = []
    for n, (start, end) in enumerate([(2.0, 4.0), (8.0, 9.5)]):
        renders = {}
        for k, c in enumerate(CANDS):
            # different lengths and levels: the pack must even both out
            renders[c] = ab.Clip(
                tone(tmp_path / "r" / f"{n}-{k}.wav", 1.0 + 0.2 * k, amp=0.02 + 0.1 * k),
                start + (0.1 if k == 2 else 0.0),
            )
        cues.append(
            ab.CueMaterial(
                set="bbt-de", episode="S01E01", cue_id=f"c{n:04d}", character_id=2,
                character="Leonard", source_text="Hallo, wie geht's?", text="Hi, how are you?",
                emotion="neutral", slot=(start, end),
                references={"original": ab.Clip(episode_audio, 0.0),
                            "english": ab.Clip(english, 0.0)},
                renders=renders, suspect={"indextts2": "too long"} if n == 0 else {},
                info={c: {"key": f"k{k}"} for k, c in enumerate(CANDS)},
            )
        )  # fmt: skip
    scene_src = {
        c: tone(tmp_path / "scene" / f"{k}.wav", 6.0, rate=48000, amp=0.05 + 0.1 * k)
        for k, c in enumerate(CANDS[:4])
    }
    scenes = [
        ab.SceneMaterial(
            set="ft-e12",
            episode="S01E12",
            window=(10.0, 16.0),
            original=ab.Clip(episode_audio, 0.0),
            renders={c: ab.Clip(p, 10.0) for c, p in scene_src.items()},
            info={c: {"cues": 3} for c in scene_src},
        )
    ]
    return cues, scenes


@pytest.fixture
def pack(tmp_path):
    cues, scenes = _materials(tmp_path)
    out = tmp_path / "pack"
    ab.write_pack(out, seed=11, cues=cues, scenes=scenes)
    return out


@pytest.mark.ffmpeg
def test_files_are_named_blind_and_page_names_no_backend(pack):
    names = sorted(p.name for p in (pack / "audio").iterdir())
    assert all(re.fullmatch(r"(cue\d\d|scene\d\d)_([A-Z]|ref|ref_en)\.wav", n) for n in names)
    for page in ("index.html", "README.txt"):
        text = (pack / page).read_text().lower()
        for b in ("kokoro", "indextts", "cosyvoice", "qwen", "xvector", "zero_shot", "suspect"):
            assert b not in text, (page, b)


@pytest.mark.ffmpeg
def test_key_maps_labels_back_and_is_never_linked(pack):
    key = tomllib.loads((pack / "key.toml").read_text())
    assert key["seed"] == 11
    labels = key["cue"]["cue01"]["labels"]
    assert sorted(labels.values()) == sorted(CANDS)
    assert key["cue"]["cue01"]["suspect"] == {"indextts2": "too long"}
    assert sorted(key["scene"]["scene01"]["labels"].values()) == sorted(CANDS[:4])
    html = (pack / "index.html").read_text()
    assert "key.toml" not in html
    # the same seed gives the same key
    again = ab.blind_order(CANDS, seed=11, slot="cue01")
    assert labels == again


@pytest.mark.ffmpeg
def test_page_is_self_contained(pack):
    html = (pack / "index.html").read_text()
    assert "http" not in html and "//" not in html.replace("<!--", "").replace("-->", "")
    srcs = re.findall(r'src="([^"]+)"', html)
    assert srcs
    for s in srcs:
        assert not s.startswith("/")
        assert (pack / s).is_file()
    assert "Hallo, wie geht" in html and "Hi, how are you?" in html and "Leonard" in html


@pytest.mark.ffmpeg
def test_every_cue_clip_has_one_length_and_one_loudness(pack):
    from fukikae.mix import loudness_stats

    for n in ("01", "02"):
        clips = sorted((pack / "audio").glob(f"cue{n}_*.wav"))
        assert len(clips) == len(CANDS) + 2
        assert len({frames(c) for c in clips}) == 1
        levels = [loudness_stats(c)["i"] for c in clips]
        assert max(levels) - min(levels) < 1.0, levels
        assert abs(levels[0] - ab.TARGET_LUFS) < 1.0


@pytest.mark.ffmpeg
def test_pack_is_owner_only(pack):
    assert stat.S_IMODE(pack.stat().st_mode) == 0o700
    for p in pack.rglob("*"):
        want = 0o700 if p.is_dir() else 0o600
        assert stat.S_IMODE(p.stat().st_mode) == want, p


@pytest.mark.ffmpeg
def test_refuses_a_non_empty_folder(tmp_path):
    cues, scenes = _materials(tmp_path)
    out = tmp_path / "pack"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(FileExistsError):
        ab.write_pack(out, seed=1, cues=cues, scenes=scenes)


# ---- verdict ------------------------------------------------------------------------------


@pytest.mark.ffmpeg
def test_verdict_template_lists_every_cue_and_scene_and_reads_empty(pack):
    text = (pack / "verdict.toml").read_text()
    data = tomllib.loads(text)
    assert set(data["cue"]) == {"cue01", "cue02"} and set(data["scene"]) == {"scene01"}
    assert data["overall"]["pick"] == ""
    v = ab.read_verdict(pack)
    assert v.pick is None and v.passed == [] and v.cues["cue01"].ranking == []


@pytest.mark.ffmpeg
def test_verdict_resolves_labels_through_the_key(pack):
    key = tomllib.loads((pack / "key.toml").read_text())
    labels = key["cue"]["cue01"]["labels"]
    best = next(lab for lab, c in labels.items() if c == "indextts2")
    other = next(lab for lab, c in labels.items() if c == "kokoro")
    text = (pack / "verdict.toml").read_text()
    text = text.replace('pick = ""', 'pick = "indextts2"', 1)
    text = text.replace("passed = []", 'passed = ["indextts2", "qwen3_tts:xvector"]', 1)
    text = text.replace(
        "[cue.cue01]\nranking = []", f'[cue.cue01]\nranking = ["{best}", "{other}"]', 1
    )
    text += (
        '\n[[override]]\nset = "bbt-de"\ncharacter = 2\nbackend = "qwen3_tts"\n'
        'mode = "xvector"\nnotes = "icl collapses"\n'
    )
    (pack / "verdict.toml").write_text(text)
    v = ab.read_verdict(pack)
    assert v.pick == ab.Candidate("indextts2", None)
    assert [c.name for c in v.passed] == ["indextts2", "qwen3_tts:xvector"]
    assert [c.name for c in v.cues["cue01"].ranking] == ["indextts2", "kokoro"]
    assert v.overrides[0].candidate == ab.Candidate("qwen3_tts", "xvector")
    assert v.overrides[0].character == 2 and v.overrides[0].set == "bbt-de"


@pytest.mark.ffmpeg
def test_verdict_stock_and_bad_values(pack):
    path = pack / "verdict.toml"
    base = path.read_text()
    path.write_text(base.replace('pick = ""', 'pick = "stock"', 1))
    v = ab.read_verdict(pack)
    assert v.pick is None and v.stock
    path.write_text(base.replace('pick = ""', 'pick = "vall-e"', 1))
    with pytest.raises(ValueError, match="vall-e"):
        ab.read_verdict(pack)
    path.write_text(base.replace("[cue.cue01]\nranking = []", '[cue.cue01]\nranking = ["Q"]', 1))
    with pytest.raises(ValueError, match="Q"):
        ab.read_verdict(pack)


def test_cli_reports_a_bad_spec_and_a_used_folder(tmp_path):
    from typer.testing import CliRunner

    from fukikae.cli import app

    spec = tmp_path / "spec.toml"
    spec.write_text("seed = 1\nbogus = 2\n")
    r = CliRunner().invoke(app, ["eval", "tts-ab", str(spec), "-o", str(tmp_path / "p")])
    assert r.exit_code == 1 and "bogus" in r.output
    used = tmp_path / "used"
    used.mkdir()
    (used / "f").write_text("")
    spec.write_text(
        'seed = 1\n[[set]]\nname = "a"\nworkspace = "/nowhere"\nepisode = "S01E01"\n'
        'candidates = ["kokoro", "indextts2"]\n'
    )
    r = CliRunner().invoke(app, ["eval", "tts-ab", str(spec), "-o", str(used)])
    assert r.exit_code == 1 and "not empty" in r.output


def test_tts_ab_reuses_the_stage_fitted_render(tmp_path):
    """B4: a cue the tts stage already rendered for its slot (its plan's key derives from
    this job's render key) is reused from the cache, not rendered again at another rate."""
    from types import SimpleNamespace

    from fukikae.artifacts import write_json
    from fukikae.cards.card import parse_card
    from fukikae.eval.ab import Candidate, _Renderer
    from fukikae.models import TtsPlan, TtsPlanCue
    from fukikae.stages.tts import fitted_key
    from fukikae.workspace import Workspace
    from tests.cards.test_card import STOCK_CARD

    ws = Workspace(tmp_path / "ws")
    work = ws.work_dir("E1")
    card = parse_card(STOCK_CARD)
    job = SimpleNamespace(
        cue=SimpleNamespace(id="c0001"), card_id="a" * 16, card=card, key="k" * 16,
        backend_cfg=None, cls=None,
    )  # fmt: skip
    final = fitted_key(job.key, 1.15)
    cache = ws.root / ".fukikae" / "tts-cache"
    cache.mkdir(parents=True)
    (cache / f"{final}.wav").write_bytes(b"RIFF")
    plan = TtsPlanCue(
        cue_id="c0001", character_id=1, extra=None, card_id=job.card_id, key=final,
        seconds=1.9, slot=2.0, rate_factor=1.15,
    )  # fmt: skip
    (work / "07-tts").mkdir(parents=True)
    write_json(work / "07-tts" / "plan.json", TtsPlan(episode="E1", cues=[plan]))
    r = _Renderer().render(Candidate("kokoro"), job, None, ws, None, work=work)  # type: ignore[arg-type]
    assert r.renders_made == 0 and r.key == final and r.rate_factor == 1.15
    assert r.path == cache / f"{final}.wav" and r.seconds == 1.9
