import wave
from pathlib import Path

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


def test_gender_reads_the_20th_percentile_so_shouting_does_not_lift_it(tmp_path: Path):
    """A male voice shouting most of the time: the median reads female, the low end doesn't."""
    x = np.concatenate([harmonic(300, 7), harmonic(125, 3)])
    path = write_wav(tmp_path / "shout.wav", x)
    assert estimate_gender(path, [(0.0, 10.0)]) == "m"


def test_gender_thresholds_come_from_config(vocals: Path):
    assert estimate_gender(vocals, [(6.0, 10.0)], female_above=300.0) == "u"
    assert auto.gender_thresholds(CFG) == (165.0, 180.0)
    band = Config(raw={"tts": {"backend": "kokoro", "gender": {"female_above": 300.0}}})
    assert auto.gender_thresholds(band) == (165.0, 300.0)


def test_anime_profile_has_a_wide_unknown_band():
    from fukikae.config import load_profile

    g = load_profile("anime")["tts"]["gender"]
    assert g["male_below"] == 165.0 and g["female_above"] == 300.0


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


def test_workspace_cloning_reads_the_episode_artifacts(tmp_path: Path):
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
    cfg = Config(raw={"tts": {"backend": "kokoro", "clone_backend": "indextts2"}})
    cloning = workspace_cloning(root, cfg, "S01E12")
    assert cloning is not None
    assert cloning.backend.name == "indextts2" and cloning.source_language == "ja"
    one = cloning.references(1)
    assert one.picks is not None and one.episode == "S01E12" and one.seconds == 15.0
    assert one.vocals == work / "01-separate" / "vocals.wav"
    two = cloning.references(2)
    assert two.picks is None and two.seconds == 4.0
    assert workspace_cloning(root, CFG, "S01E12") is None
