import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from fukikae.cards import auto, stock
from fukikae.cards.auto import character_spans, ensure_assignments, extras_voices
from fukikae.cards.card import CardError, card_id, parse_card
from fukikae.cards.select import estimate_gender
from fukikae.cards.stock import STOCK_POOLS, pick_stock, stock_card
from fukikae.cards.store import CardStore
from fukikae.config import Config
from fukikae.models import Assignments, ClusterAssignment, Delivery, DubCue, Turn
from fukikae.registry.store import ProtectedAssignmentError, Registry
from tests.cards.test_card import STOCK_CARD

POOL = STOCK_POOLS["kokoro"]["en"]
CFG = Config(raw={"tts": {"backend": "kokoro"}})
RATE = 16000


def harmonic(f0: float, seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for k in range(1, 6):
        x += np.sin(2 * np.pi * f0 * k * t) / k
    return (x * 0.2).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def vocals(tmp_path: Path) -> Path:
    """0-6 s a 120 Hz voice, 6-10 s a 220 Hz voice, 10-12 s silence."""
    x = np.concatenate([harmonic(120, 6), harmonic(220, 4), np.zeros(2 * RATE, np.float32)])
    return write_wav(tmp_path / "vocals.wav", x)


def test_pick_stock_by_gender_then_reuse():
    assert pick_stock("kokoro", "en", "f", set()) == POOL["f"][0]
    assert pick_stock("kokoro", "en", "m", {POOL["m"][0]}) == POOL["m"][1]
    # exhausted: reuse from the start, then the least reused
    assert pick_stock("kokoro", "en", "m", set(POOL["m"])) == POOL["m"][0]
    assert pick_stock("kokoro", "en", "m", [*POOL["m"], POOL["m"][0]]) == POOL["m"][1]
    assert pick_stock("kokoro", "en", "f", set(), extras=True) == POOL["extras_f"][0]


def test_pick_stock_unknown_gender_alternates():
    used: list[str] = []
    picks = []
    for _ in range(4):
        v = pick_stock("kokoro", "en", "u", used)
        used.append(v)
        picks.append(v)
    assert picks == [POOL["f"][0], POOL["m"][0], POOL["f"][1], POOL["m"][1]]


def test_pick_stock_without_a_pool_names_backend_and_language():
    with pytest.raises(CardError, match="kokoro.*'de'"):
        pick_stock("kokoro", "de", "f", set())


def test_stock_card_id_is_stable_and_meta_free():
    a = stock_card("kokoro", "af_heart", "en", "rev1")
    b = stock_card("kokoro", "af_heart", "en", "rev1")
    assert card_id(a) == card_id(b)
    assert a.voice.source == "stock" and a.voice.voice == "af_heart" and a.voice.backend == "kokoro"
    assert card_id(stock_card("kokoro", "af_heart", "en", "rev2")) != card_id(a)
    assert card_id(stock_card("kokoro", "af_heart", "es", "rev1")) != card_id(a)


def test_parse_stock_ref():
    assert stock.parse_stock_ref("stock:kokoro:af_heart") == ("kokoro", "af_heart")
    for bad in ("stock:kokoro", "stock::af_heart", "kokoro:af_heart", "stock:kokoro:"):
        with pytest.raises(CardError):
            stock.parse_stock_ref(bad)


def test_estimate_gender_on_synthetic_tones(vocals: Path):
    assert estimate_gender(vocals, [(0.0, 6.0)]) == "m"
    assert estimate_gender(vocals, [(6.0, 10.0)]) == "f"
    assert estimate_gender(vocals, [(10.0, 12.0)]) == "u"  # nothing voiced
    assert estimate_gender(vocals, []) == "u"


def test_estimate_gender_reads_stereo_float_at_any_rate(tmp_path: Path):
    from scipy.io import wavfile

    x = harmonic(115, 3, 44100)
    path = tmp_path / "v.wav"
    wavfile.write(path, 44100, np.stack([x, x], axis=1))
    assert estimate_gender(path, [(0.0, 3.0)]) == "m"


def _registry(tmp_path: Path, n: int) -> tuple[Registry, list[int]]:
    reg = Registry(tmp_path / "ws" / ".fukikae" / "registry" / "registry.sqlite")
    ids = [reg.create_actor_with_character(np.ones(4, np.float32))[1] for _ in range(n)]
    return reg, ids


def test_ensure_assignments_creates_unlocked_auto_stock_cards(tmp_path: Path, vocals: Path):
    reg, (a, b) = _registry(tmp_path, 2)
    store = CardStore(tmp_path / "ws")
    lines = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)], b: [(6.0, 10.0)]}, vocals)
    for cid, gender in ((a, "m"), (b, "f")):
        cur = reg.card_for(cid, "en")
        assert cur is not None and cur.by == "auto" and not cur.locked
        assert cur.card_id is not None
        card = store.get(cur.card_id)
        assert card.voice.backend == "kokoro" and card.voice.source == "stock"
        assert card.voice.voice == POOL[gender][0]
    assert len(lines) == 2
    assert any(f"character {a}" in ln and "m" in ln and POOL["m"][0] in ln for ln in lines)
    reg.close()


def test_ensure_assignments_gives_the_most_talkative_the_first_voice(tmp_path: Path, vocals: Path):
    reg, (a, b) = _registry(tmp_path, 2)
    store = CardStore(tmp_path / "ws")
    # both male; b speaks longer, so b gets the pool's first voice
    ensure_assignments(reg, store, CFG, {a: [(0.0, 2.0)], b: [(2.0, 6.0)]}, vocals)

    def voice(cid: int) -> str | None:
        cur = reg.card_for(cid, "en")
        assert cur is not None and cur.card_id is not None
        return store.get(cur.card_id).voice.voice

    assert (voice(b), voice(a)) == (POOL["m"][0], POOL["m"][1])
    reg.close()


def test_ensure_assignments_skips_voices_already_assigned(tmp_path: Path, vocals: Path):
    reg, (a, b) = _registry(tmp_path, 2)
    store = CardStore(tmp_path / "ws")
    first = store.put(stock_card("kokoro", POOL["m"][0], "en", stock.revision("kokoro")))
    reg.assign_card(a, "en", first, by="assign", locked=True)
    ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)], b: [(0.0, 6.0)]}, vocals)
    cur = reg.card_for(b, "en")
    assert cur is not None and cur.card_id is not None
    assert store.get(cur.card_id).voice.voice == POOL["m"][1]
    reg.close()


def test_existing_assignment_is_never_replaced(tmp_path: Path, vocals: Path):
    reg, (a, b, c) = _registry(tmp_path, 3)
    store = CardStore(tmp_path / "ws")
    mine = store.put(parse_card(STOCK_CARD))
    other = store.put(parse_card(STOCK_CARD.replace("af_heart", "am_adam")))
    third = store.put(parse_card(STOCK_CARD.replace("af_heart", "bf_emma")))
    reg.assign_card(a, "en", mine, by="assign", locked=True)
    reg.assign_card(b, "en", other, by="import:pack", locked=False)
    reg.assign_card(c, "en", third, by="auto", locked=False)
    before = {x: reg.card_history(x, "en") for x in (a, b, c)}
    spans = {a: [(0.0, 6.0)], b: [(6.0, 10.0)], c: [(0.0, 6.0)]}
    assert ensure_assignments(reg, store, CFG, spans, vocals) == []
    assert {x: reg.card_history(x, "en") for x in (a, b, c)} == before
    reg.close()


def test_protected_assignment_race_keeps_the_existing_card(
    tmp_path: Path, vocals: Path, monkeypatch
):
    """A card that appears between the check and the write (another process) is kept: auto
    never overrides (CD2/CD3)."""
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    mine = store.put(parse_card(STOCK_CARD))
    reg.assign_card(a, "en", mine, by="assign", locked=True)
    real = reg.card_for
    monkeypatch.setattr(reg, "card_for", lambda cid, lang: None if cid == a else real(cid, lang))
    seen: list[tuple] = []
    real_assign = reg.assign_card

    def spy(*args, **kw):
        seen.append((args, kw))
        return real_assign(*args, **kw)

    monkeypatch.setattr(reg, "assign_card", spy)
    lines = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)]}, vocals)
    assert seen and seen[0][1].get("override", False) is False and seen[0][1]["by"] == "auto"
    assert real(a, "en").card_id == mine  # type: ignore[union-attr]
    assert len(lines) == 1 and "kept" in lines[0]
    reg.close()


def test_protected_error_is_the_registry_error():
    assert issubclass(ProtectedAssignmentError, ValueError)
    assert auto.ProtectedAssignmentError is ProtectedAssignmentError


def _cue(i: int, character: int | None, extra: str | None, text: str | None = "hi") -> DubCue:
    return DubCue(
        id=f"c{i:04d}",
        start=float(i),
        end=i + 0.9,
        character_id=character,
        extra=extra,
        speaker=f"extra:{extra}" if extra else f"char{character}",
        source_text="x",
        text=text,
        budget=20,
        delivery=Delivery(reference=(float(i), i + 0.9)),
    )


def test_extras_voices_stable_by_first_appearance():
    cues = [
        _cue(1, None, "S3"),
        _cue(2, 1, None),
        _cue(3, None, "unknown", text=None),  # an errored cue still fixes the order
        _cue(4, None, "S3"),
        _cue(5, None, "S7"),
    ]
    got = extras_voices(cues, CFG)
    assert got == {
        "S3": POOL["extras_f"][0],
        "unknown": POOL["extras_m"][0],
        "S7": POOL["extras_f"][1],
    }
    assert extras_voices(cues, CFG) == got


def test_character_spans_leave_out_extras_and_review():
    asg = Assignments(
        episode="e1",
        threshold=0.6,
        clusters=[
            ClusterAssignment(
                cluster="S0", actor_id=1, character_id=1, score=0.9, clean_seconds=9,
                status="matched",
            ),
            ClusterAssignment(
                cluster="S1", actor_id=2, character_id=2, score=0.4, clean_seconds=9,
                status="review",
            ),
            ClusterAssignment(
                cluster="S2", actor_id=None, character_id=None, score=None, clean_seconds=1,
                status="extra",
            ),
        ],
        turns=[
            Turn(start=0, end=1, speaker="S0"),
            Turn(start=1, end=2, speaker="S1"),
            Turn(start=2, end=3, speaker="S0"),
            Turn(start=3, end=4, speaker="S2"),
        ],
    )  # fmt: skip
    assert character_spans(asg) == {1: [(0.0, 1.0), (2.0, 3.0)]}


def test_gender_reads_the_median_not_the_low_end(tmp_path: Path):
    """7 s at 300 Hz and 3 s at 125 Hz: the median reads the larger part (A0); the old 20th
    percentile read the low end and called it male."""
    x = np.concatenate([harmonic(300, 7), harmonic(125, 3)])
    path = write_wav(tmp_path / "mixed.wav", x)
    assert estimate_gender(path, [(0.0, 10.0)]) == "f"


def test_gender_thresholds_come_from_config(vocals: Path):
    assert estimate_gender(vocals, [(6.0, 10.0)], female_above=300.0) == "u"
    band = Config(raw={"tts": {"backend": "kokoro", "gender": {"female_above": 300.0}}})
    assert auto.gender_thresholds(band) == (165.0, 300.0)


def test_default_bands_come_from_pitch():
    from fukikae.cards import select
    from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW

    assert auto.gender_thresholds(CFG) == (MALE_BELOW, FEMALE_ABOVE) == (165.0, 200.0)
    assert (select.MALE_BELOW, select.FEMALE_ABOVE) == (MALE_BELOW, FEMALE_ABOVE)


def test_anime_profile_keeps_its_band_until_the_operator_decides():
    """Task 2's calibration passed no band (docs/results/2026-11-11-gender-bands.md), so the
    anime override stays until the operator rules on it."""
    from fukikae.config import load_profile

    g = load_profile("anime")["tts"]["gender"]
    assert g["male_below"] == 165.0 and g["female_above"] == 300.0


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(auto, "_MEDIAN_WARNED", set())


@pytest.mark.usefixtures("fresh_warning")
def test_workspace_gender_table_warns_about_the_median(caplog):
    band = Config(raw={"tts": {"backend": "kokoro", "gender": {"female_above": 300.0}}})
    with caplog.at_level("WARNING", logger="fukikae.cards.auto"):
        auto.gender_thresholds(band)
        auto.gender_thresholds(band)  # once per run
    warned = [r for r in caplog.records if "median F0" in r.getMessage()]
    assert len(warned) == 1
    assert "[tts.gender] now reads the median F0, not the 20th percentile" in warned[0].getMessage()
    assert "male_below and female_above" in warned[0].getMessage()


@pytest.mark.usefixtures("fresh_warning")
def test_profile_gender_table_does_not_warn(caplog):
    band = Config(
        raw={"tts": {"backend": "kokoro"}},
        profile_raw={"tts": {"gender": {"male_below": 160.0, "female_above": 210.0}}},
    )
    with caplog.at_level("WARNING", logger="fukikae.cards.auto"):
        assert auto.gender_thresholds(band) == (160.0, 210.0)
    assert not [r for r in caplog.records if "median F0" in r.getMessage()]


def test_ensure_assignments_uses_the_configured_band(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    band = Config(raw={"tts": {"backend": "kokoro", "gender": {"female_above": 300.0}}})
    (line,) = ensure_assignments(reg, store, band, {a: [(6.0, 10.0)]}, vocals)
    assert "gender estimate u" in line and "F0 2" in line
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.note == line.split(": ", 1)[1]
    reg.close()


# -- the clone branch (Task 12, D12 with the D24 verdict) -----------------------------------


def _cloning(refs: dict, vocals: Path):
    from fukikae.cards.auto import CharacterRefs, Cloning
    from fukikae.cards.select import RefPick
    from tests.cards.test_select import CloneSynth

    def find(cid: int) -> CharacterRefs:
        spans = refs.get(cid, [])
        picks = [RefPick(a, b, "セリフ", 30.0, emo) for a, b, emo in spans]
        seconds = sum(b - a for a, b, _ in spans)
        if seconds < 6.0:
            return CharacterRefs(seconds=seconds)
        return CharacterRefs(seconds=seconds, picks=picks, episode="S01E12", vocals=vocals)

    return Cloning(backend=CloneSynth, source_language="ja", references=find)


def test_clone_backend_set_builds_clone_card(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    cloning = _cloning({a: [(0.0, 4.0, None), (6.0, 10.0, "angry")]}, vocals)
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 10.0)]}, vocals, cloning=cloning)
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.by == "auto" and not cur.locked and cur.card_id
    card = store.get(cur.card_id)
    assert card.voice.backend == "clonefake" and card.voice.source == "clone"
    assert [r.origin for r in card.voice.references] == [
        {"episode": "S01E12", "start": 0.0, "end": 4.0},
        {"episode": "S01E12", "start": 6.0, "end": 10.0},
    ]
    assert card.voice.references[0].language == "ja"
    assert "clone clonefake" in line and "8.0 s" in line and "S01E12" in line
    assert "references may carry emotion: angry" in line  # ruling CD5
    assert cur.note == line.split(": ", 1)[1]
    reg.close()


def test_short_speech_falls_back_to_stock_with_report_line(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    cloning = _cloning({a: [(0.0, 4.2, None)]}, vocals)
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)]}, vocals, cloning=cloning)
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.card_id is not None
    card = store.get(cur.card_id)
    assert card.voice.source == "stock" and card.voice.voice == POOL["m"][0]
    assert f"character {a}: 4.2 s of clean speech, under 6 s; stock voice {POOL['m'][0]}" in line
    reg.close()


def test_no_clone_backend_keeps_stock(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)]}, vocals, cloning=None)
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.card_id is not None
    assert store.get(cur.card_id).voice.source == "stock" and "under 6 s" not in line
    reg.close()


def test_auto_never_replaces_existing_assignment(tmp_path: Path, vocals: Path):
    """CD2/CD3/CD9: with cloning on, an operator's, an import's, a restock's or an earlier
    auto card is kept, and nothing is cut or stored for it."""
    reg, ids = _registry(tmp_path, 4)
    store = CardStore(tmp_path / "ws")
    for cid, by, locked in zip(
        ids, ("assign", "import:pack", "auto", "restock"), (True, False, False, False), strict=True
    ):
        reg.assign_card(cid, "en", store.put(parse_card(STOCK_CARD)), by=by, locked=locked)
    before = {x: reg.card_history(x, "en") for x in ids}
    cloning = _cloning({x: [(0.0, 4.0, None), (6.0, 10.0, None)] for x in ids}, vocals)
    spans = {x: [(0.0, 10.0)] for x in ids}
    assert ensure_assignments(reg, store, CFG, spans, vocals, cloning=cloning) == []
    assert {x: reg.card_history(x, "en") for x in ids} == before
    assert not list((tmp_path / "ws").rglob("*.wav"))
    reg.close()


def test_clone_race_keeps_the_existing_card(tmp_path: Path, vocals: Path, monkeypatch):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    mine = store.put(parse_card(STOCK_CARD))
    reg.assign_card(a, "en", mine, by="assign", locked=True)
    real = reg.card_for
    monkeypatch.setattr(reg, "card_for", lambda cid, lang: None if cid == a else real(cid, lang))
    cloning = _cloning({a: [(0.0, 4.0, None), (6.0, 10.0, None)]}, vocals)
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 10.0)]}, vocals, cloning=cloning)
    assert "kept" in line and real(a, "en").card_id == mine  # type: ignore[union-attr]
    reg.close()


def test_clone_backend_comes_from_tts_config():
    from fukikae.cards.auto import clone_backend

    assert clone_backend(CFG) is None
    assert clone_backend(Config(raw={"tts": {"backend": "kokoro", "clone_backend": "x"}})) == "x"
    assert clone_backend(Config(raw={"tts": {"clone_backend": ""}})) is None


@pytest.mark.parametrize("profile", ["default", "anime", "live-action"])
def test_profiles_clone_with_indextts2_by_default(profile: str):
    """D24: IndexTTS2 is the default clone backend; Kokoro stays the stock and extras one."""
    from fukikae.config import load_profile

    tts = load_profile(profile)["tts"]
    assert tts["clone_backend"] == "indextts2" and tts["backend"] == "kokoro"


def test_clone_backend_is_no_backend_option():
    from fukikae.stages.tts import backend_config, stage_settings

    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    assert "clone_backend" not in stage_settings(cfg)[0].options
    assert "clone_backend" not in backend_config(cfg, "kokoro").options


def test_workspace_cloning_reads_the_episode_artifacts(tmp_path: Path, monkeypatch):
    """The production clone source: the character's eligible turns in the workspace's
    episodes (CD5 ranking), the current episode's transcript language as the source
    language, and the clean-speech seconds for the stock fallback's report."""
    from fukikae.artifacts import write_json
    from fukikae.cards.auto import workspace_cloning
    from fukikae.models import Segment, Transcript
    from fukikae.workspace import Workspace
    from tests.cards.test_select import _vocals

    root = tmp_path / "ws"
    ws = Workspace(root)
    work = ws.work_dir("S01E12")
    (work / "01-separate").mkdir(parents=True)
    (work / "05-characters").mkdir()
    (work / "03-asr").mkdir()
    a_turns = [Turn(start=s, end=s + 5, speaker="S0") for s in (0, 10, 20)]
    b_turns = [Turn(start=30, end=34, speaker="S1")]
    _vocals(
        work / "01-separate" / "vocals.wav", [(t.start, t.end, 0.01) for t in a_turns + b_turns]
    )
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode="S01E12", threshold=0.6,
            clusters=[
                ClusterAssignment(cluster="S0", actor_id=1, character_id=1, score=None,
                                  clean_seconds=15, status="pinned"),
                ClusterAssignment(cluster="S1", actor_id=2, character_id=2, score=None,
                                  clean_seconds=4, status="pinned"),
            ],
            turns=a_turns + b_turns,
        ),
    )  # fmt: skip
    segs = [Segment(start=t.start, end=t.end, text="これはテストのセリフです", words=[])
            for t in a_turns + b_turns]  # fmt: skip
    write_json(
        work / "03-asr" / "transcript.json",
        Transcript(language="ja", backend="x", word_timing=False, segments=segs),
    )
    monkeypatch.setattr("fukikae.cards.auto.clone_problem", lambda *_: None)
    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    cloning = workspace_cloning(root, cfg, "S01E12")
    assert cloning is not None
    assert cloning.backend.name == "indextts2" and cloning.source_language == "ja"
    one = cloning.references(1)
    assert one.picks is not None and one.episode == "S01E12" and one.seconds == 15.0
    assert one.vocals == work / "01-separate" / "vocals.wav" and one.language == "ja"
    two = cloning.references(2)
    assert two.picks is None and two.seconds == 4.0
    assert workspace_cloning(root, CFG, "S01E12") is None


# -- fix round 1: no clone card before the clone backend can run -----------------------------


def test_unusable_clone_backend_falls_back_to_stock_naming_setup(tmp_path: Path, vocals: Path):
    from fukikae.cards.auto import Cloning
    from tests.cards.test_select import CloneSynth

    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    asked: list[int] = []

    def find(cid: int):
        asked.append(cid)
        raise AssertionError("no references are read for an unusable backend")

    cloning = Cloning(
        backend=CloneSynth, source_language="ja", references=find,
        unavailable="no worker environment at /x/.venvs/clonefake",
    )  # fmt: skip
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)]}, vocals, cloning=cloning)
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.card_id is not None and not asked
    assert store.get(cur.card_id).voice.source == "stock"
    assert "clonefake is not usable" in line and "fukikae backends setup clonefake" in line
    assert "no worker environment" in line
    reg.close()


def test_clone_problem_needs_the_worker_env_and_its_lock(tmp_path: Path, monkeypatch):
    from fukikae.backends import loader
    from fukikae.backends.tts import remote
    from fukikae.cards.auto import clone_problem

    cls = cast(Any, loader.backend_class("tts", "indextts2"))
    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    monkeypatch.setenv("FUKIKAE_WORKER_CHECKOUT", str(tmp_path))
    why = clone_problem(cls, cfg)
    assert why is not None and "worker environment" in why
    venv = tmp_path / ".venvs" / cls.extra
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    project = tmp_path / "workers" / cls.extra
    project.mkdir(parents=True)
    (project / "uv.lock").write_text("lock v1")
    assert "lock hash" in (clone_problem(cls, cfg) or "")
    remote.record_lock(project, venv)
    monkeypatch.setattr("fukikae.stages.tts.synth_device", lambda *_: "cuda:0")
    assert clone_problem(cls, cfg) is None
    (project / "uv.lock").write_text("lock v2")
    assert "another version" in (clone_problem(cls, cfg) or "")


def test_clone_problem_needs_the_default_device(monkeypatch):
    from fukikae.backends import loader
    from fukikae.cards.auto import clone_problem

    cls = cast(Any, loader.backend_class("tts", "indextts2"))
    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    monkeypatch.setattr(type(cls({}, "cpu")), "lock_problem", lambda _self: None)
    monkeypatch.setattr(type(cls({}, "cpu")), "interpreter", lambda _self: Path("/bin/sh"))

    from fukikae.stages.tts import NoUsableGpuError

    def no_gpu(*_):
        raise NoUsableGpuError("indextts2 found no usable GPU")

    monkeypatch.setattr("fukikae.stages.tts.synth_device", no_gpu)
    assert "no usable GPU" in (clone_problem(cls, cfg) or "")


def test_workspace_cloning_reports_an_unusable_backend(tmp_path: Path, monkeypatch):
    from fukikae.cards.auto import workspace_cloning

    monkeypatch.setenv("FUKIKAE_WORKER_CHECKOUT", str(tmp_path))
    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    cloning = workspace_cloning(tmp_path / "ws", cfg, "S01E01")
    assert (
        cloning is not None and cloning.unavailable and "worker environment" in cloning.unavailable
    )


# -- fix round 2: device typos fail; unusable-backend fallbacks are upgraded later -----------


def test_clone_problem_raises_a_device_typo(monkeypatch):
    from fukikae.backends import loader
    from fukikae.cards.auto import clone_problem
    from fukikae.config import ConfigError

    cls = cast(Any, loader.backend_class("tts", "indextts2"))
    monkeypatch.setattr(type(cls({}, "cpu")), "lock_problem", lambda _self: None)
    monkeypatch.setattr(type(cls({}, "cpu")), "interpreter", lambda _self: Path("/bin/sh"))
    for table in ({"device": "gpu"}, {"device": "cpu"}):
        cfg = Config(raw={"tts": {"backend": "kokoro", "indextts2": table}})
        with pytest.raises(ConfigError):
            clone_problem(cls, cfg)


def _unusable():
    from fukikae.cards.auto import Cloning
    from tests.cards.test_select import CloneSynth

    return Cloning(CloneSynth, "ja", lambda _cid: None, unavailable="no worker environment")  # type: ignore[arg-type]


def test_unusable_backend_fallback_is_marked_and_upgraded_once_usable(tmp_path: Path, vocals: Path):
    reg, (a, b, c) = _registry(tmp_path, 3)
    store = CardStore(tmp_path / "ws")
    spans = {x: [(0.0, 10.0)] for x in (a, b, c)}
    ensure_assignments(reg, store, CFG, spans, vocals, cloning=_unusable())
    for x in (a, b, c):
        cur = reg.card_for(x, "en")
        assert cur is not None and cur.by == "auto-fallback" and not cur.locked
    reg.lock_card(b, "en", True)  # locked: never upgraded
    mine = store.put(parse_card(STOCK_CARD))
    reg.assign_card(c, "en", mine, by="assign", locked=False)  # operator's: never upgraded
    cloning = _cloning({x: [(0.0, 4.0, None), (6.0, 10.0, None)] for x in (a, b, c)}, vocals)
    lines = ensure_assignments(reg, store, CFG, spans, vocals, cloning=cloning)
    cur = reg.card_for(a, "en")
    assert cur is not None and cur.by == "auto" and cur.card_id is not None
    assert store.get(cur.card_id).voice.source == "clone"
    assert len(lines) == 1 and "replaces the stock fallback" in lines[0]
    assert store.get(reg.card_for(b, "en").card_id).voice.source == "stock"  # type: ignore[union-attr,arg-type]
    assert reg.card_for(c, "en").card_id == mine  # type: ignore[union-attr]
    reg.close()


def test_under_6_s_stock_is_plain_auto_and_stays(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    short = _cloning({a: [(0.0, 4.2, None)]}, vocals)
    ensure_assignments(reg, store, CFG, {a: [(0.0, 6.0)]}, vocals, cloning=short)
    assert reg.card_for(a, "en").by == "auto"  # type: ignore[union-attr]
    more = _cloning({a: [(0.0, 4.0, None), (6.0, 10.0, None)]}, vocals)
    assert ensure_assignments(reg, store, CFG, {a: [(0.0, 10.0)]}, vocals, cloning=more) == []
    reg.close()


# -- Task 9: name-confirmed references and stale automatic cards -----------------------------

NAMED_EP = "S01E14"


def _named_episode(
    root: Path,
    episode: str,
    turns: list[tuple[float, float, str, float]],
    clusters: list[dict],
    *,
    names: Sequence[tuple[float, float, str | None, bool]] | None = None,
    excluded: Sequence[tuple[float, float]] = (),
) -> Path:
    """An episode work dir: vocals (each turn a tone, its noise setting its SNR), a transcript
    line per turn, assignments with these refined turns and clusters, and, with ``names``,
    a used names.json with those lines (start, end, key, inner) and ``excluded`` spans."""
    from fukikae.artifacts import write_json
    from fukikae.models import Diarization, Segment, Transcript
    from fukikae.names.inputs import names_path
    from fukikae.names.models import LabelInfo, NamedLine, NamesResult
    from fukikae.names.normalize import NameClass
    from fukikae.workspace import Workspace
    from tests.cards.test_select import _vocals

    work = Workspace(root).work_dir(episode)
    for d in ("01-separate", "03-asr", "05-characters"):
        (work / d).mkdir(parents=True, exist_ok=True)
    _vocals(work / "01-separate" / "vocals.wav", [(a, b, n) for a, b, _, n in turns])
    segs = [Segment(start=a, end=b, text="これはテストのセリフです", words=[])
            for a, b, _, _ in turns]  # fmt: skip
    write_json(
        work / "03-asr" / "transcript.json",
        Transcript(language="ja", backend="x", word_timing=False, segments=segs),
    )
    write_json(
        work / "05-characters" / "assignments.json",
        Assignments(
            episode=episode, threshold=0.6,
            clusters=[ClusterAssignment.model_validate(
                {"actor_id": c.get("character_id"), "score": None, "clean_seconds": 10.0,
                 "status": "matched", **c}) for c in clusters],
            turns=[Turn(start=a, end=b, speaker=s) for a, b, s, _ in turns],
        ),
    )  # fmt: skip
    if names is not None:
        lines = [
            NamedLine(id=i, start=a, end=b, raw=(k or "A/B").upper(), key=k,
                      display=k.title() if k else None,
                      cls=NameClass.NAMED if k else NameClass.MULTI, inner=inner, text="x")
            for i, (a, b, k, inner) in enumerate(names)
        ]  # fmt: skip
        names_path(work).parent.mkdir(parents=True, exist_ok=True)
        refined = [Turn(start=a, end=b, speaker=s) for a, b, s, _ in turns]
        write_json(names_path(work).parent / "turns.json", Diarization(backend="x", turns=refined))
        labels = {
            c["cluster"]: LabelInfo(parent=c["cluster"].split("~")[0], key=c.get("name"),
                                    display=None, cls="named" if c.get("name") else "remainder")
            for c in clusters
        }  # fmt: skip
        write_json(
            names_path(work),
            NamesResult(
                episode=episode, used=True, verdict="full", reason=None,
                source={"how": "target"}, offset_s=0.0, coverage={}, lines=lines, hidden=[],
                pieces=[], labels=labels, unvoiced=[], excluded=list(excluded),
                dropped_row_seconds=0.0, suggestions=[],
            ),
        )  # fmt: skip
    return work


# S01E14: character 1 (Natsu) has three named turns, one of them under an inner-voice line,
# and a loud unnamed remainder; character 2 (Wendy) one named turn. Noise 0.001 is loud
# (high SNR), 0.05 quiet.
NATSU_TURNS = [
    (0.0, 5.0, "S0~natsu", 0.02),
    (10.0, 15.0, "S0~natsu", 0.02),
    (20.0, 25.0, "S0", 0.001),  # the remainder: loudest, but not name-confirmed
    (30.0, 35.0, "S0~wendy", 0.02),
    (40.0, 45.0, "S0~natsu", 0.001),  # an inner-voice line: never a reference
    (50.0, 55.0, "S0~natsu", 0.03),
]
NATSU_CLUSTERS = [
    {"cluster": "S0~natsu", "character_id": 1, "name": "natsu", "source": "name"},
    {"cluster": "S0", "character_id": 1},
    {"cluster": "S0~wendy", "character_id": 2, "name": "wendy", "source": "name"},
]
NATSU_NAMES = [
    (0.0, 5.0, "natsu", False), (10.0, 15.0, "natsu", False), (30.0, 35.0, "wendy", False),
    (40.0, 45.0, "natsu", True), (50.0, 55.0, "natsu", False),
]  # fmt: skip


def _ws_named(tmp_path: Path) -> tuple[Registry, list[int], Any]:
    from fukikae.workspace import Workspace

    reg, ids = _registry(tmp_path, 3)
    _named_episode(tmp_path / "ws", NAMED_EP, NATSU_TURNS, NATSU_CLUSTERS, names=NATSU_NAMES,
                   excluded=[(40.0, 45.0)])  # fmt: skip
    return reg, ids, Workspace(tmp_path / "ws")


def _inside(pick, spans) -> bool:
    return any(a <= pick.start and pick.end <= b for a, b in spans)


def test_candidates_use_only_name_confirmed_turns(tmp_path: Path):
    from fukikae.cards.auto import character_candidates

    reg, _, ws = _ws_named(tmp_path)
    got = character_candidates(ws, [NAMED_EP], 1)
    assert [(p.start, p.end) for p in got[NAMED_EP]] == [(0.0, 5.0), (10.0, 15.0), (50.0, 55.0)]
    reg.close()


def test_inner_voice_and_multi_spans_are_never_references(tmp_path: Path):
    """The 40-45 turn is Natsu's by name but an inner-voice line: excluded. A multi line
    (two names) leaves its stretch unnamed, so its turn is a remainder and never a reference
    either."""
    from fukikae.cards.auto import character_candidates

    reg, _, ws = _ws_named(tmp_path)
    picks = character_candidates(ws, [NAMED_EP], 1)[NAMED_EP]
    assert not any(p.start < 45.0 and p.end > 40.0 for p in picks)
    assert not any(p.start < 25.0 and p.end > 20.0 for p in picks)
    reg.close()


def test_without_names_reference_selection_is_unchanged(tmp_path: Path):
    """No names.json: every turn of the character's clusters is a candidate, exactly as
    `eligible_references` over its labels gives (the selection before names existed)."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import character_candidates, character_references
    from fukikae.cards.select import eligible_references, pick_references, speaker_register
    from fukikae.models import Transcript
    from fukikae.workspace import Workspace

    reg, _ = _registry(tmp_path, 3)
    work = _named_episode(tmp_path / "ws", NAMED_EP, NATSU_TURNS, NATSU_CLUSTERS)
    ws = Workspace(tmp_path / "ws")
    turns = [Turn(start=a, end=b, speaker=s) for a, b, s, _ in NATSU_TURNS]
    tr = read_json(work / "03-asr" / "transcript.json", Transcript)
    vocals = work / "01-separate" / "vocals.wav"
    register = speaker_register(vocals, [(t.start, t.end) for t in turns], (165.0, 200.0))
    want = eligible_references(turns, {"S0~natsu", "S0"}, tr, None, vocals, exclude=[],
                               register=register)  # fmt: skip
    assert character_candidates(ws, [NAMED_EP], 1)[NAMED_EP] == want
    picks = pick_references(want)
    assert picks is not None and character_references(ws, [NAMED_EP], 1) == (
        NAMED_EP, picks, False
    )  # fmt: skip
    assert any(p.start == 20.0 for p in picks)  # the loud remainder, as before
    reg.close()


def test_named_episode_preferred_over_louder_acoustic_episode(tmp_path: Path):
    from fukikae.cards.auto import character_references

    reg, _, ws = _ws_named(tmp_path)
    loud = [(float(s), s + 8.0, "S5", 0.001) for s in (0, 10, 20)]  # 24 s, all loud
    _named_episode(tmp_path / "ws", "S01E13", loud, [{"cluster": "S5", "character_id": 1}])
    got = character_references(ws, ["S01E13", NAMED_EP], 1)
    assert got is not None
    ep, picks, confirmed = got
    assert ep == NAMED_EP and confirmed
    assert all(_inside(p, [(0, 5), (10, 15), (50, 55)]) for p in picks)
    # Wendy's one named turn (5 s) is under the minimum: the acoustic episode is used, flagged
    _named_episode(tmp_path / "ws", "S01E12", loud, [{"cluster": "S5", "character_id": 2}])
    assert character_references(ws, ["S01E12", NAMED_EP], 2) == (
        "S01E12", character_references(ws, ["S01E12"], 2)[1], False  # type: ignore[index]
    )  # fmt: skip
    reg.close()


def test_a_lock_on_another_characters_key_confirms_nothing(tmp_path: Path):
    """A lock-kept sub-cluster is name-confirmed only when the locked character answers to
    its key."""
    from fukikae.cards.auto import confirmed_labels, key_owner

    reg, (one, two, _) = _registry(tmp_path, 3)
    reg.name_character(one, "Natsu", "natsu", "en")
    asg = Assignments(episode="x", threshold=0.6, clusters=[
        ClusterAssignment.model_validate({"cluster": "S0~natsu", "actor_id": 1, "character_id": one,
            "score": None, "clean_seconds": 5, "status": "matched", "name": "natsu",
            "source": "lock"}),
        ClusterAssignment.model_validate({"cluster": "S1~happy", "actor_id": 1, "character_id": one,
            "score": None, "clean_seconds": 5, "status": "matched", "name": "happy",
            "source": "lock"}),
    ], turns=[])  # fmt: skip
    assert confirmed_labels(asg, one, key_owner(reg)) == {"S0~natsu"}
    assert confirmed_labels(asg, two, key_owner(reg)) == set()
    reg.close()


def _auto_clone(reg: Registry, store: CardStore, cid: int, origins, *, by: str = "auto",
                locked: bool = False) -> str:  # fmt: skip
    """A clone card with references at ``origins`` [(episode, start, end)], assigned ``by``."""
    from fukikae.cards.card import Card, Reference, Voice

    clip = store.root / "clip.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    write_wav(clip, harmonic(200.0, 1.0))
    refs = tuple(
        Reference(clip=store.add_sidecar(clip, "clip"), transcript="x", language="ja",
                  origin={"episode": e, "start": a, "end": b})
        for e, a, b in origins
    )  # fmt: skip
    card = Card(target_language="en", voice=Voice(backend="clonefake", model_revision="rev-9",
                source="clone", seed=1234, references=refs))  # fmt: skip
    new = store.put(card)
    reg.assign_card(cid, "en", new, by=by, locked=locked)
    return new


def _cloning_ws(ws):
    """The production clone source over ``ws``, with the fake clone backend."""
    import dataclasses

    from fukikae.cards.auto import workspace_cloning

    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("fukikae.cards.auto.clone_problem", lambda *_: None)
        cloning = workspace_cloning(ws.root, cfg, NAMED_EP)
    assert cloning is not None
    from tests.cards.test_select import CloneSynth

    return dataclasses.replace(cloning, backend=CloneSynth)


def test_auto_card_with_stale_references_is_repicked(tmp_path: Path, vocals: Path):
    from fukikae.cards.auto import REPICK_NOTE, stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(NAMED_EP, 20.0, 25.0), (NAMED_EP, 30.0, 35.0)])
    stale = stale_cards(ws, reg, store, [1], "en")
    # 20-25 is Natsu's unnamed remainder: unnamed speech never makes a card stale
    assert [(r["start"], r["names"]) for r in stale[1]] == [(30.0, ["wendy"])]
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals,
                               cloning=_cloning_ws(ws), stale=stale)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.by == "auto" and not cur.locked
    assert cur.card_id != old and cur.note is not None and cur.note.startswith(REPICK_NOTE)
    assert [h.card_id for h in reg.card_history(1, "en")] == [old, cur.card_id]  # CD1
    origins = [r.origin for r in store.get(cur.card_id or "").voice.references]
    assert all(_inside_o(o, [(0, 5), (10, 15), (50, 55)]) for o in origins)
    assert len(lines) == 1 and REPICK_NOTE in lines[0]
    assert stale_cards(ws, reg, store, [1], "en") == {}  # the new card is not stale
    reg.close()


def _inside_o(o, spans) -> bool:
    return any(a <= o["start"] and o["end"] <= b for a, b in spans)


@pytest.mark.parametrize("by", ["assign", "import:Pack", "auto-locked"])
def test_assigned_imported_or_locked_card_is_flagged_not_repicked(tmp_path, vocals, by):
    from fukikae.cards.auto import stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    locked = by == "auto-locked"
    old = _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)],
                      by="auto" if locked else by, locked=locked)  # fmt: skip
    stale = stale_cards(ws, reg, store, [1], "en")
    flags: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals,
                               cloning=_cloning_ws(ws), stale=stale, flags=flags)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id == old and len(reg.card_history(1, "en")) == 1
    assert lines == []  # nothing changed: the flag goes to flags, never to the report
    (line,) = flags
    assert "voices clone" in line and "wendy" in line
    reg.close()


def test_restock_card_with_stale_references_is_repicked(tmp_path: Path, vocals: Path):
    """CD9: restock is automatic, so a stale restock clone card is re-picked."""
    from fukikae.cards.auto import REPICK_NOTE, stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)], by="restock")
    stale = stale_cards(ws, reg, store, [1], "en")
    ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals,
                       cloning=_cloning_ws(ws), stale=stale)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id != old and (cur.note or "").startswith(REPICK_NOTE)
    reg.close()


def test_a_stale_card_without_confirmed_picks_gets_a_stock_voice(tmp_path: Path, vocals: Path):
    """Character 3 has no name-confirmed turns and no acoustic-only episode, so the re-pick
    falls back the way a new card does: a stock voice, no longer the voice of the speech the
    names gave Wendy. The stock card has no references, so the next run leaves it."""
    from fukikae.cards.auto import REPICK_NOTE, stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 3, [(NAMED_EP, 30.0, 35.0)])
    stale = stale_cards(ws, reg, store, [3], "en")
    (line,) = ensure_assignments(reg, store, CFG, {3: [(30.0, 35.0)]}, vocals,
                                 cloning=_cloning_ws(ws), stale=stale)  # fmt: skip
    cur = reg.card_for(3, "en")
    assert cur is not None and cur.card_id != old and cur.by == "auto"
    assert store.get(cur.card_id or "").voice.source == "stock"
    assert REPICK_NOTE in line and "wendy" in line
    assert stale_cards(ws, reg, store, [3], "en") == {}
    reg.close()


def test_stale_references_need_10_percent_under_someone_elses_names():
    from fukikae.cards.auto import stale_references
    from fukikae.cards.card import Card, Reference, Voice

    refs = tuple(
        Reference(clip="0" * 64 + ".wav", transcript="x", language="ja",
                  origin={"episode": e, "start": a, "end": b})
        for e, a, b in [("E1", 0.0, 10.0), ("E1", 10.0, 20.0), ("E1", 20.0, 30.0), ("E2", 0, 9)]
    )  # fmt: skip
    card = Card(target_language="en", voice=Voice(backend="clonefake", model_revision="r",
                source="clone", seed=1, references=refs))  # fmt: skip
    got = stale_references(card, {"E1": [(9.1, 11.0), (21.0, 22.0)]})
    # 0-10 has 9% under another's names, 10-20 10%, 20-30 10%; E2 used no names
    assert [(r["start"], r["end"], r["overlap"]) for r in got] == [(10.0, 20.0, 0.1),
                                                                   (20.0, 30.0, 0.1)]  # fmt: skip


# -- Task 9 review: the controller's ruling on stale references, and the names guard ----------

# Character 3 speaks only in S9, an unnamed remainder of the names-used episode (24 s, loud):
# no name is said over it.
REMAINDER_TURNS = [*NATSU_TURNS, *((float(s), s + 8.0, "S9", 0.001) for s in (60, 70, 80))]
REMAINDER_CLUSTERS = [*NATSU_CLUSTERS, {"cluster": "S9", "character_id": 3}]


def _ws_remainder(tmp_path: Path, turns=REMAINDER_TURNS, names=NATSU_NAMES):
    from fukikae.workspace import Workspace

    reg, ids = _registry(tmp_path, 3)
    _named_episode(tmp_path / "ws", NAMED_EP, turns, REMAINDER_CLUSTERS, names=names,
                   excluded=[(40.0, 45.0)])  # fmt: skip
    return reg, ids, Workspace(tmp_path / "ws")


def test_unnamed_speech_never_makes_a_card_stale(tmp_path: Path):
    """The reviewer's q2b: character 3's clone from its unnamed remainder keeps its clone."""
    from fukikae.cards.auto import stale_cards

    reg, _, ws = _ws_remainder(tmp_path)
    vocals = write_wav(tmp_path / "vocals.wav", harmonic(200, 120))
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 3, [(NAMED_EP, 60.0, 68.0), (NAMED_EP, 70.0, 78.0)])
    assert stale_cards(ws, reg, store, [3], "en") == {}
    assert ensure_assignments(reg, store, CFG, {3: [(60.0, 88.0)]}, vocals,
                              cloning=_cloning_ws(ws), stale={}) == []  # fmt: skip
    assert reg.card_for(3, "en").card_id == old  # type: ignore[union-attr]
    reg.close()


def test_a_stale_card_without_name_confirmed_turns_re_picks_from_its_own_speech(tmp_path: Path):
    """Character 3's card was cloned from Wendy's named turn; it has no name-confirmed turns,
    so the re-pick takes its own acoustic speech (the S9 remainder) before a stock voice, and
    says the references are not name-confirmed. A second run is a no-op."""
    from fukikae.cards.auto import REPICK_NOTE, stale_cards

    reg, _, ws = _ws_remainder(tmp_path)
    vocals = write_wav(tmp_path / "vocals.wav", harmonic(200, 120))
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 3, [(NAMED_EP, 30.0, 35.0)])
    stale = stale_cards(ws, reg, store, [3], "en")
    assert [r["names"] for r in stale[3]] == [["wendy"]]
    cloning = _cloning_ws(ws)
    (line,) = ensure_assignments(reg, store, CFG, {3: [(60.0, 88.0)]}, vocals,
                                 cloning=cloning, stale=stale)  # fmt: skip
    cur = reg.card_for(3, "en")
    assert cur is not None and cur.card_id != old and (cur.note or "").startswith(REPICK_NOTE)
    card = store.get(cur.card_id or "")
    assert card.voice.source == "clone"
    assert all(_inside_o(r.origin, [(60, 68), (70, 78), (80, 88)]) for r in card.voice.references)
    assert "references are not name-confirmed" in line
    # second run: the new card is not stale, nothing to do
    assert stale_cards(ws, reg, store, [3], "en") == {}
    assert ensure_assignments(reg, store, CFG, {3: [(60.0, 88.0)]}, vocals, cloning=cloning,
                              stale={}) == []  # fmt: skip
    assert len(reg.card_history(3, "en")) == 2
    reg.close()


def test_acoustic_candidates_leave_out_turns_under_other_names(tmp_path: Path):
    """A remainder turn with 10% or more of it under another key's line is no candidate; one
    under 10% still is."""
    from fukikae.cards.auto import character_candidates

    names = [*NATSU_NAMES, (67.0, 69.0, "wendy", False), (79.5, 80.5, "lucy", False)]
    reg, _, ws = _ws_remainder(tmp_path, names=names)
    got = character_candidates(ws, [NAMED_EP], 3, acoustic=True)[NAMED_EP]
    # 60-68 has 1 s (12.5%) of Wendy; 80-88 0.5 s (6%) of Lucy
    assert sorted((p.start, p.end) for p in got) == [(70.0, 78.0), (80.0, 88.0)]
    assert character_candidates(ws, [NAMED_EP], 3) == {}  # no name-confirmed turns
    reg.close()


def test_the_acoustic_tier_is_searched_only_without_confirmed_picks(tmp_path: Path, monkeypatch):
    """Final review B #11: workspace_cloning reads a character's acoustic candidates only
    when its name-confirmed ones give no picks."""
    import fukikae.cards.auto as auto

    reg, _, ws = _ws_named(tmp_path)
    cloning = _cloning_ws(ws)
    real = auto.character_candidates
    calls: list[bool] = []

    def spy(*a, **kw):
        calls.append(bool(kw.get("acoustic")))
        return real(*a, **kw)

    monkeypatch.setattr(auto, "character_candidates", spy)
    refs = cloning.references(1)
    assert refs.picks and refs.confirmed and calls == [False]
    calls.clear()
    cloning.references(3)  # no name-confirmed turns: the acoustic tier is read
    assert calls == [False, True]
    reg.close()


def test_a_lock_kept_turn_is_never_stale_for_its_character(tmp_path: Path):
    """The operator locked character 1 onto S0~wendy's speech: its names say Wendy, but a
    lock keeps speech whatever the names say, so a reference there isn't stale."""
    from fukikae.cards.auto import stale_cards
    from fukikae.workspace import Workspace

    reg, _ = _registry(tmp_path, 3)
    locked = {"cluster": "S0~wendy", "character_id": 1, "name": "wendy", "source": "lock"}
    clusters = [*NATSU_CLUSTERS[:2], locked]
    _named_episode(tmp_path / "ws", NAMED_EP, NATSU_TURNS, clusters, names=NATSU_NAMES)
    ws = Workspace(tmp_path / "ws")
    store = CardStore(ws.root)
    _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)])
    assert stale_cards(ws, reg, store, [1], "en") == {}
    reg.close()


def _stale_pair(ws, how: str) -> None:
    """Make the names-used episode's assignments older than its names: other refined turns,
    or a names label its clusters don't cover."""
    from fukikae.artifacts import read_json, write_json
    from fukikae.models import Diarization
    from fukikae.names.inputs import names_path
    from fukikae.names.models import LabelInfo, NamesResult

    work = ws.work_dir(NAMED_EP)
    if how == "turns":
        path = names_path(work).parent / "turns.json"
        d = read_json(path, Diarization)
        extra = Turn(start=90, end=95, speaker="S0~lucy")
        write_json(path, d.model_copy(update={"turns": [*d.turns, extra]}))
    else:
        n = read_json(names_path(work), NamesResult)
        labels = {**n.labels, "S0~lucy": LabelInfo(parent="S0", key="lucy", display="Lucy",
                                                  cls="named")}  # fmt: skip
        write_json(names_path(work), n.model_copy(update={"labels": labels}))


@pytest.mark.parametrize("how", ["turns", "labels"])
def test_an_episode_whose_assignments_are_older_than_its_names_is_skipped(tmp_path, how):
    """The stale pair: names.json and turns.json from a newer names run than assignments.json.
    Its name decisions are stale, so neither stale detection nor candidates read it, and the
    report says why and what to run."""
    from fukikae.cards.auto import character_candidates, character_references, stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)])
    assert stale_cards(ws, reg, store, [1], "en")  # stale while the pair agrees
    _stale_pair(ws, how)
    notes: list[str] = []
    assert stale_cards(ws, reg, store, [1], "en", notes=notes) == {}
    (note,) = notes
    assert note.startswith(f"{NAMED_EP}: its assignments.json is older than its names")
    assert f"fukikae run <ws> {NAMED_EP} --from characters" in note
    assert character_candidates(ws, [NAMED_EP], 1) == {}
    assert character_candidates(ws, [NAMED_EP], 1, acoustic=True) == {}
    assert character_references(ws, [NAMED_EP], 1) is None
    reg.close()


def test_stale_cards_read_each_episode_once(tmp_path: Path, monkeypatch):
    import fukikae.artifacts as artifacts
    from fukikae.cards.auto import stale_cards

    reg, ids, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    for cid in ids:
        _auto_clone(reg, store, cid, [(NAMED_EP, 30.0, 35.0)])
    reads: list[str] = []
    real = artifacts.read_json

    def counting(path, model):
        reads.append(Path(path).name)
        return real(path, model)

    monkeypatch.setattr(artifacts, "read_json", counting)
    stale_cards(ws, reg, store, ids, "en")
    assert reads.count("assignments.json") == 1 and reads.count("names.json") == 1


def test_a_stale_card_kept_while_the_clone_backend_is_unusable(tmp_path: Path, vocals: Path):
    """Stale and unusable: the automatic card stays, and the flag says why and how to fix it;
    nothing is reported as changed."""
    from fukikae.cards.auto import stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)])
    stale = stale_cards(ws, reg, store, [1], "en")
    flags: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals,
                               cloning=_unusable(), stale=stale, flags=flags)  # fmt: skip
    assert lines == [] and reg.card_for(1, "en").card_id == old  # type: ignore[union-attr]
    (flag,) = flags
    assert "kept until the clone backend" in flag and "no worker environment" in flag
    assert "fukikae backends setup" in flag
    flags.clear()
    assert ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals, cloning=None,
                              stale=stale, flags=flags) == []  # fmt: skip
    (flag,) = flags
    assert "no [tts] clone_backend is set" in flag and "can run" not in flag
    reg.close()


def test_a_lock_kept_key_is_not_a_key_the_character_answers_to(tmp_path: Path):
    """F1: character 1's lock kept the sub-cluster S1~wendy, so that turn stays its own, but
    the lock doesn't make Wendy's other lines its speech: a Wendy line elsewhere is foreign.
    A key it took by name (natsu) is its own."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import foreign_spans
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamesResult
    from fukikae.workspace import Workspace

    root = tmp_path / "ws"
    turns = [(0.0, 4.0, "S0~natsu", 0.01), (10.0, 14.0, "S1~wendy", 0.01), (20.0, 24.0, "S2", 0.01)]
    clusters = [
        {"cluster": "S0~natsu", "character_id": 1, "name": "natsu", "source": "name"},
        {"cluster": "S1~wendy", "character_id": 1, "name": "wendy", "source": "lock"},
        {"cluster": "S2", "character_id": 2},
    ]
    names = [(0.0, 4.0, "natsu", False), (10.0, 14.0, "wendy", False), (20.0, 24.0, "wendy", False)]
    work = _named_episode(root, "S01E12", turns, clusters, names=names)
    from fukikae.models import Assignments

    asg = read_json(work / "05-characters" / "assignments.json", Assignments)
    got = foreign_spans(asg, read_json(names_path(Workspace(root).work_dir("S01E12")), NamesResult),
                        1, lambda _k: None)  # fmt: skip
    assert got == [(20.0, 24.0)]


def test_an_engine_is_refused_as_clone_backend_or_tts_backend():
    """indextts2_trt renders indextts2 cards; it makes none, so it is no clone_backend and no
    [tts] backend. The message points at [tts] engines."""
    from fukikae.cards.auto import clone_backend, tts_backend
    from fukikae.config import ConfigError

    with pytest.raises(ConfigError, match=r"\[tts\] engines"):
        clone_backend(Config(raw={"tts": {"clone_backend": "indextts2_trt"}}))
    with pytest.raises(ConfigError, match=r"\[tts\] engines"):
        tts_backend(Config(raw={"tts": {"backend": "indextts2_trt"}}))


# -- D32: joined fragments and the clone floor ------------------------------------------------

# Young Makarov in miniature: one shouted 6.3 s name-confirmed turn, and two fragments the
# names cut 0.3 s apart (4.5 s joined); Wendy has one named turn between them.
FLOOR_EP = NAMED_EP
FLOOR_TURNS = [
    (0.0, 6.3, "S0~mak", 0.01),
    (10.0, 12.0, "S0~mak", 0.01),
    (12.3, 14.5, "S0~mak", 0.01),
    (20.0, 25.0, "S0~wendy", 0.01),
]
FLOOR_CLUSTERS = [
    {"cluster": "S0~mak", "character_id": 1, "name": "makarov", "source": "name"},
    {"cluster": "S0~wendy", "character_id": 2, "name": "wendy", "source": "name"},
]
FLOOR_NAMES = [
    (0.0, 6.3, "makarov", False), (10.0, 14.5, "makarov", False), (20.0, 25.0, "wendy", False),
]  # fmt: skip


def _ws_floor(tmp_path: Path, turns=FLOOR_TURNS, names=FLOOR_NAMES):
    from fukikae.workspace import Workspace

    reg, ids = _registry(tmp_path, 2)
    _named_episode(tmp_path / "ws", FLOOR_EP, turns, FLOOR_CLUSTERS, names=names)
    return reg, ids, Workspace(tmp_path / "ws")


def test_name_confirmed_fragments_are_joined(tmp_path: Path):
    from fukikae.cards.auto import character_candidates

    reg, _, ws = _ws_floor(tmp_path)
    got = character_candidates(ws, [FLOOR_EP], 1)[FLOOR_EP]
    assert sorted((p.start, p.end) for p in got) == [(0.0, 6.3), (10.0, 14.5)]
    reg.close()


def test_a_join_over_someone_elses_named_line_falls_back(tmp_path: Path):
    """Wendy's name over the gap is 0.3 s of a 4.5 s joined turn, 6.7%: still joined. Over
    12.0-12.6 it also covers 14% of the second fragment, which is then no candidate and
    blocks the join, and the first fragment (2 s) is too short alone."""
    from fukikae.cards.auto import character_candidates

    names = [*FLOOR_NAMES, (12.0, 12.3, "wendy", False)]
    reg, _, ws = _ws_floor(tmp_path, names=names)
    got = character_candidates(ws, [FLOOR_EP], 1)[FLOOR_EP]
    assert (10.0, 14.5) in [(p.start, p.end) for p in got]
    reg.close()
    names = [*FLOOR_NAMES, (12.0, 12.6, "wendy", False)]
    reg, _, ws = _ws_floor(tmp_path / "b", names=names)
    got = character_candidates(ws, [FLOOR_EP], 1)[FLOOR_EP]
    assert [(p.start, p.end) for p in got] == [(0.0, 6.3)]
    reg.close()


def test_an_auto_card_below_the_floor_is_re_picked_when_a_better_pick_exists(
    tmp_path: Path, vocals: Path
):
    """D32: the one-clip 6.3 s card is below the floor; the join gives 2 clips and 10.8 s,
    so it is re-picked like a stale card: a new card, the old one kept in history (CD1)."""
    from fukikae.cards.auto import FLOOR_NOTE, below_floor

    reg, _, ws = _ws_floor(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)])
    short = below_floor(reg, store, [1, 2], "en")
    assert list(short) == [1] and short[1].card_id == old
    assert short[1].phrase == "1 clip and 6.3 s, under the floor of 2 clips and 10 s"
    flags: list[str] = []
    (line,) = ensure_assignments(reg, store, CFG, {1: [(0.0, 14.5)]}, vocals,
                                 cloning=_cloning_ws(ws), short=short, flags=flags)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id != old and cur.by == "auto" and not cur.locked
    assert [h.card_id for h in reg.card_history(1, "en")] == [old, cur.card_id]
    assert line.startswith(f"character 1: {FLOOR_NOTE} (1 clip and 6.3 s")
    origins = sorted((r.origin["start"], r.origin["end"])
                     for r in store.get(cur.card_id or "").voice.references
                     if r.origin is not None)  # fmt: skip
    assert origins == [(0.0, 6.3), (10.0, 14.5)]
    assert flags == [] and below_floor(reg, store, [1], "en") == {}
    reg.close()


def test_an_auto_card_below_the_floor_with_no_better_pick_is_kept_and_flagged(
    tmp_path: Path, vocals: Path
):
    from fukikae.cards.auto import below_floor

    turns = [FLOOR_TURNS[0], FLOOR_TURNS[3]]  # no fragments: the 6.3 s turn is all there is
    reg, _, ws = _ws_floor(tmp_path, turns=turns)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)])
    flags: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 6.3)]}, vocals,
                               cloning=_cloning_ws(ws),
                               short=below_floor(reg, store, [1], "en"), flags=flags)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id == old and len(reg.card_history(1, "en")) == 1
    assert lines == []
    assert flags == [
        f"character 1: clone card {old} is below the clone floor (1 clip and 6.3 s, under the "
        "floor of 2 clips and 10 s) and no better reference pick exists; kept, flagged for "
        "review: `fukikae voices clone` with a better clip replaces it"
    ]
    reg.close()


def test_a_floor_re_pick_never_falls_back_to_a_stock_voice(tmp_path: Path, vocals: Path):
    """Character 2's card is below the floor and no reference picks remain at all: kept and
    flagged, never a stock voice or another character's card."""
    from fukikae.cards.auto import below_floor

    reg, _, ws = _ws_floor(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 2, [(FLOOR_EP, 20.0, 25.0)])
    flags: list[str] = []
    ensure_assignments(reg, store, CFG, {2: [(20.0, 25.0)]}, vocals, cloning=_cloning_ws(ws),
                       short=below_floor(reg, store, [2], "en"), flags=flags)  # fmt: skip
    cur = reg.card_for(2, "en")
    assert cur is not None and cur.card_id == old
    (flag,) = flags
    assert "below the clone floor (1 clip and 5.0 s" in flag and "voices clone" in flag
    reg.close()


@pytest.mark.parametrize("by", ["assign", "import:Pack"])
def test_curated_and_imported_cards_are_never_below_the_floor(tmp_path: Path, by: str):
    from fukikae.cards.auto import below_floor

    reg, _, ws = _ws_floor(tmp_path)
    store = CardStore(ws.root)
    _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)], by=by)
    assert below_floor(reg, store, [1], "en") == {}
    reg.close()


def test_a_locked_auto_card_below_the_floor_is_flagged_not_re_picked(tmp_path: Path, vocals: Path):
    from fukikae.cards.auto import below_floor

    reg, _, ws = _ws_floor(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)], locked=True)
    flags: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 14.5)]}, vocals,
                               cloning=_cloning_ws(ws),
                               short=below_floor(reg, store, [1], "en"), flags=flags)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id == old and lines == []
    (flag,) = flags
    assert "below the clone floor" in flag and "locked" in flag and "voices clone" in flag
    reg.close()


def test_a_new_clone_below_the_floor_is_made_and_flagged(tmp_path: Path, vocals: Path):
    reg, (a,) = _registry(tmp_path, 1)
    store = CardStore(tmp_path / "ws")
    flags: list[str] = []
    (line,) = ensure_assignments(reg, store, CFG, {a: [(0.0, 6.3)]}, vocals,
                                 cloning=_cloning({a: [(0.0, 6.3, None)]}, vocals),
                                 flags=flags)  # fmt: skip
    cur = reg.card_for(a, "en")
    assert cur is not None and store.get(cur.card_id or "").voice.source == "clone"
    assert "below the clone floor (1 clip and 6.3 s" in line
    (flag,) = flags
    assert flag.startswith(f"character {a}: clone card {cur.card_id} is below the clone floor")
    reg.close()


def test_floor_lines_say_what_the_next_tts_run_does(tmp_path: Path, monkeypatch):
    """`voices list` and `names show`: a re-pick when a better pick exists, else a flag."""
    from fukikae.cards.auto import floor_lines

    monkeypatch.setattr("fukikae.cards.auto.repick_blocker", lambda _cfg: None)
    reg, _, ws = _ws_floor(tmp_path)
    store = CardStore(ws.root)
    _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)])
    old2 = _auto_clone(reg, store, 2, [(FLOOR_EP, 20.0, 25.0)])
    one, two = floor_lines(ws, reg, store, [1, 2], "en")
    assert "below the clone floor (1 clip and 6.3 s" in one
    assert one.endswith("the next tts run re-picks it (2 clips, 10.8 s)")
    assert two.startswith(f"character 2's card {old2} (auto) is below the clone floor")
    assert two.endswith("no better reference pick exists; flagged for review: "
                        "`fukikae voices clone` with a better clip replaces it")  # fmt: skip
    reg.close()


def test_the_episode_meeting_the_floor_beats_one_with_more_seconds(tmp_path: Path):
    from fukikae.cards.auto import character_references

    reg, _ = _registry(tmp_path, 1)
    from fukikae.workspace import Workspace

    one = [(0.0, 11.5, "S5", 0.01)]  # one clip, 11.5 s
    two = [(0.0, 5.0, "S5", 0.01), (10.0, 15.5, "S5", 0.01)]  # two clips, 10.5 s
    _named_episode(tmp_path / "ws", "S01E12", one, [{"cluster": "S5", "character_id": 1}])
    _named_episode(tmp_path / "ws", "S01E13", two, [{"cluster": "S5", "character_id": 1}])
    got = character_references(Workspace(tmp_path / "ws"), ["S01E12", "S01E13"], 1)
    assert got is not None and got[0] == "S01E13" and len(got[1]) == 2
    reg.close()


def test_floor_filling_never_moves_a_character_to_another_episode(tmp_path: Path):
    """Episodes that reach the floor compare by the seconds picked before any floor filling
    (neutral turns, and non-neutral ones up to 6 s), as before D32: an episode of three
    neutral clips (10.5 s) keeps winning over one whose angry turns only fill it up."""
    from fukikae.cards.auto import character_references
    from fukikae.workspace import Workspace

    reg, _ = _registry(tmp_path, 1)
    calm = [(0.0, 3.5, "S5", 0.01), (10.0, 13.5, "S5", 0.01), (20.0, 23.5, "S5", 0.01)]
    # 4 s neutral, then angry 3 s (the louder) and 6 s: 7 s before D32, 13 s filled
    loud = [(0.0, 4.0, "S5", 0.01), (10.0, 13.0, "S5", 0.002), (20.0, 26.0, "S5", 0.01)]
    _named_episode(tmp_path / "ws", "S01E11", loud, [{"cluster": "S5", "character_id": 1}])
    _named_episode(tmp_path / "ws", "S01E12", calm, [{"cluster": "S5", "character_id": 1}])
    from fukikae.artifacts import write_json
    from fukikae.models import Tags, TagSpan

    tags = Tags(backend="x", events=[], emotion_backend="emotion2vec", emotions=[
        TagSpan(start=10.0, end=13.0, label="angry", confidence=0.9),
        TagSpan(start=20.0, end=26.0, label="angry", confidence=0.9)])  # fmt: skip
    work = Workspace(tmp_path / "ws").work_dir("S01E11")
    (work / "04b-tag").mkdir(parents=True, exist_ok=True)
    write_json(work / "04b-tag" / "tags.json", tags)
    got = character_references(Workspace(tmp_path / "ws"), ["S01E11", "S01E12"], 1)
    assert got is not None and got[0] == "S01E12"
    reg.close()


# -- D32 review -----------------------------------------------------------------------------


def test_eligible_seconds_are_the_picked_episodes_in_the_tier_used(tmp_path: Path):
    """Charle's shape: the picks are name-confirmed turns of a names-used episode (11 s
    eligible), while an acoustic-only episode has 15 s the confirmed tier can't use: the
    figure is the picked episode's 11 s."""
    from fukikae.cards.auto import find_references
    from fukikae.workspace import Workspace

    reg, _ = _registry(tmp_path, 2)
    named = [(0.0, 5.0, "S0~mak", 0.01), (10.0, 16.0, "S0~mak", 0.01)]
    clusters = [{"cluster": "S0~mak", "character_id": 1, "name": "makarov", "source": "name"}]
    names = [(0.0, 5.0, "makarov", False), (10.0, 16.0, "makarov", False)]
    _named_episode(tmp_path / "ws", "S01E12", named, clusters, names=names)
    plain = [(0.0, 5.0, "S5", 0.01), (10.0, 15.0, "S5", 0.01), (20.0, 25.0, "S5", 0.01)]
    _named_episode(tmp_path / "ws", "S01E13", plain, [{"cluster": "S5", "character_id": 1}])
    got = find_references(Workspace(tmp_path / "ws"), ["S01E12", "S01E13"], 1, "ja")
    assert got.episode == "S01E12" and got.confirmed
    assert got.seconds == 11.0
    reg.close()


def test_below_the_floor_episodes_rank_by_how_close_they_come(tmp_path: Path):
    """Neither episode reaches the floor: the one with 2 clips and 9.5 s (6.5 s neutral, then
    an angry 3 s) beats 2 neutral clips of 8.5 s, though it picked less before filling."""
    from fukikae.artifacts import write_json
    from fukikae.cards.auto import character_references
    from fukikae.models import Tags, TagSpan
    from fukikae.workspace import Workspace

    reg, _ = _registry(tmp_path, 1)
    calm = [(0.0, 4.0, "S5", 0.01), (10.0, 14.5, "S5", 0.01)]
    filled = [(0.0, 6.5, "S5", 0.01), (10.0, 13.0, "S5", 0.01)]
    _named_episode(tmp_path / "ws", "S01E11", calm, [{"cluster": "S5", "character_id": 1}])
    _named_episode(tmp_path / "ws", "S01E12", filled, [{"cluster": "S5", "character_id": 1}])
    tags = Tags(backend="x", events=[], emotion_backend="emotion2vec", emotions=[
        TagSpan(start=10.0, end=13.0, label="angry", confidence=0.9)])  # fmt: skip
    work = Workspace(tmp_path / "ws").work_dir("S01E12")
    (work / "04b-tag").mkdir(parents=True, exist_ok=True)
    write_json(work / "04b-tag" / "tags.json", tags)
    got = character_references(Workspace(tmp_path / "ws"), ["S01E11", "S01E12"], 1)
    assert got is not None and got[0] == "S01E12" and len(got[1]) == 2
    reg.close()


def test_a_locked_stale_card_under_the_floor_says_both(tmp_path: Path, vocals: Path):
    from fukikae.cards.auto import below_floor, stale_cards

    reg, _, ws = _ws_named(tmp_path)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(NAMED_EP, 30.0, 35.0)], locked=True)
    stale = stale_cards(ws, reg, store, [1], "en")
    flags: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 55.0)]}, vocals,
                               cloning=_cloning_ws(ws), stale=stale,
                               short=below_floor(reg, store, [1], "en"), flags=flags)  # fmt: skip
    assert lines == [] and reg.card_for(1, "en").card_id == old  # type: ignore[union-attr]
    (flag,) = flags
    assert "wendy" in flag and "voices clone" in flag
    assert flag.endswith(
        "; below the clone floor (1 clip and 5.0 s, under the floor of 2 clips and 10 s)"
    )
    reg.close()


def test_a_re_pick_still_under_the_floor_is_flagged_and_a_second_run_changes_nothing(
    tmp_path: Path, vocals: Path
):
    """The one-clip 6.3 s card is re-picked to two clips (9.3 s), still under the floor:
    flagged. The next run finds no better pick: no new card, no report line, the same flag."""
    from fukikae.cards.auto import below_floor

    turns = [FLOOR_TURNS[0], (10.0, 13.0, "S0~mak", 0.01), FLOOR_TURNS[3]]
    names = [(0.0, 6.3, "makarov", False), (10.0, 13.0, "makarov", False),
             (20.0, 25.0, "wendy", False)]  # fmt: skip
    reg, _, ws = _ws_floor(tmp_path, turns=turns, names=names)
    store = CardStore(ws.root)
    old = _auto_clone(reg, store, 1, [(FLOOR_EP, 0.0, 6.3)])
    flags: list[str] = []
    (line,) = ensure_assignments(reg, store, CFG, {1: [(0.0, 13.0)]}, vocals,
                                 cloning=_cloning_ws(ws),
                                 short=below_floor(reg, store, [1], "en"), flags=flags)  # fmt: skip
    cur = reg.card_for(1, "en")
    assert cur is not None and cur.card_id != old
    assert "below the clone floor (2 clips and 9.3 s" in line
    (flag,) = flags
    assert flag.startswith(f"character 1: clone card {cur.card_id} is below the clone floor")
    again: list[str] = []
    lines = ensure_assignments(reg, store, CFG, {1: [(0.0, 13.0)]}, vocals,
                               cloning=_cloning_ws(ws),
                               short=below_floor(reg, store, [1], "en"), flags=again)  # fmt: skip
    assert lines == [] and reg.card_for(1, "en").card_id == cur.card_id  # type: ignore[union-attr]
    assert len(reg.card_history(1, "en")) == 2
    (flag,) = again
    assert "no better reference pick exists" in flag
    reg.close()


def test_reference_candidates_are_read_once_per_process_until_a_file_changes(
    tmp_path: Path, monkeypatch
):
    """Each episode-run's tts asks `find_references` again for a flagged character: the
    episode's candidates and song exclusions are worked out once while its files are
    unchanged, whatever order the episodes come in."""
    import os

    from fukikae.cards import select
    from fukikae.cards.auto import find_references

    reg, _, ws = _ws_floor(tmp_path)
    reg.close()
    calls = {"eligible": 0, "exclusions": 0}
    eligible, exclusions = select.eligible_references, select.reference_exclusions

    def count_eligible(*a, **kw):
        calls["eligible"] += 1
        return eligible(*a, **kw)

    def count_exclusions(*a, **kw):
        calls["exclusions"] += 1
        return exclusions(*a, **kw)

    monkeypatch.setattr(select, "eligible_references", count_eligible)
    monkeypatch.setattr(select, "reference_exclusions", count_exclusions)
    first = find_references(ws, [FLOOR_EP], 1, "ja")
    assert find_references(ws, [FLOOR_EP], 1, "ja") == first
    assert calls == {"eligible": 1, "exclusions": 1}
    tr = ws.work_dir(FLOOR_EP) / "03-asr" / "transcript.json"
    st = tr.stat()
    os.utime(tr, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    find_references(ws, [FLOOR_EP], 1, "ja")
    assert calls == {"eligible": 2, "exclusions": 1}


# -- A1: reference gates in the candidates ------------------------------------------------------

# Natsu's name-confirmed turns at 130 Hz, one loud confirmed turn at 240 Hz (someone else's
# shout the names gave him).
GATE_TURNS = [
    (0.0, 5.0, "S0~natsu", 0.02), (30.0, 35.0, "S0~natsu", 0.02),
    (60.0, 64.5, "S0~natsu", 0.0005), (90.0, 95.0, "S0~natsu", 0.02),
]  # fmt: skip
GATE_HZ = [130.0, 130.0, 240.0, 130.0]


def _ws_gated(tmp_path: Path):
    from fukikae.workspace import Workspace
    from tests.cards.test_select import _voices

    reg, ids = _registry(tmp_path, 2)
    clusters = [{"cluster": "S0~natsu", "character_id": 1, "name": "natsu", "source": "name"}]
    names = [(a, b, "natsu", False) for a, b, _, _ in GATE_TURNS]
    work = _named_episode(tmp_path / "ws", NAMED_EP, GATE_TURNS, clusters, names=names)
    _voices(
        work / "01-separate" / "vocals.wav",
        [(a, b, hz, n) for (a, b, _, n), hz in zip(GATE_TURNS, GATE_HZ, strict=True)],
        seconds=120.0,
    )
    return reg, ids, Workspace(tmp_path / "ws")


def test_name_confirmed_candidates_skip_the_register_gate(tmp_path: Path):
    """Names vouch for who speaks, so the name-confirmed tier keeps an off-register turn
    (Erza's calm alto against her shouted register, FT E13); the voicing, song and scene
    gates still run there."""
    from fukikae.cards.auto import character_candidates
    from fukikae.cards.select import Gated

    reg, _, ws = _ws_gated(tmp_path)
    gated: dict[str, Gated] = {}
    got = character_candidates(ws, [NAMED_EP], 1, gated=gated)[NAMED_EP]
    assert sorted(p.start for p in got) == [0.0, 30.0, 60.0, 90.0]
    assert all(p.voiced_fraction is not None for p in got)  # gate 2 measured them
    assert gated[NAMED_EP] == Gated()
    assert any((ws.root / ".fukikae" / "pitch-cache").glob("*.npz"))
    reg.close()


def test_acoustic_candidates_are_gated_by_the_register(tmp_path: Path):
    """The acoustic tier (no names, or the character's remainders) is where a cluster mixes
    voices: there the 240 Hz turn among 130 Hz speech is gated."""
    from fukikae.cards.auto import character_candidates
    from fukikae.cards.select import Gated

    reg, _, ws = _ws_gated(tmp_path)
    gated: dict[str, Gated] = {}
    got = character_candidates(ws, [NAMED_EP], 1, acoustic=True, gated=gated)[NAMED_EP]
    assert sorted(p.start for p in got) == [0.0, 30.0, 90.0]
    assert gated[NAMED_EP] == Gated(off_register=1, unvoiced=0, song_like=0, top=1)
    reg.close()


def test_the_clone_note_names_gated_top_candidates(tmp_path: Path):
    from fukikae.cards.auto import find_references

    reg, _, ws = _ws_gated(tmp_path)
    from fukikae.cards.select import reference_report

    names = ws.work_dir(NAMED_EP) / "04c-names" / "names.json"
    names.unlink()  # no names: the acoustic tier, where the register gate runs
    (names.parent / "turns.json").unlink()
    found = find_references(ws, [NAMED_EP], 1, "ja")
    assert found.picks is not None and not any(p.start == 60.0 for p in found.picks)
    assert found.gated is not None and found.gated.top == 1
    assert "gated: 1 off-register" in reference_report(found.picks, found.gated)
    reg.close()


def test_the_register_reads_the_pitch_bands_not_a_profile_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A workspace band of 300 Hz (the anime profile's, tuned on the old estimator) must not
    make the register the median of only the turns above 300 Hz."""
    from fukikae.cards import auto, select
    from fukikae.workspace import Workspace

    reg, _, ws = _ws_gated(tmp_path)
    gender = "[tts.gender]\nmale_below = 165.0\nfemale_above = 300.0\n"
    (ws.root / "fukikae.toml").write_text(gender)
    seen: list[tuple[float, float]] = []
    real = select.speaker_register

    def spy(vocals, spans, bands=(165.0, 200.0), root=None):
        seen.append(bands)
        return real(vocals, spans, bands, root)

    monkeypatch.setattr(select, "speaker_register", spy)
    auto.character_candidates(Workspace(ws.root), [NAMED_EP], 1)
    assert seen == [(165.0, 200.0)]
    reg.close()
