# Character Identification Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the measured identification defects of the Sol Bianca dub automatically (clone references gated by pitch, voicing, songs and scene; extras gender from pitch; a per-line gender rule on `emotion_audio`; ending songs and Whisper's outro hallucinations), then give the operator a per-episode review page whose exported decisions `fukikae review --apply` writes through the registry, with an opt-in checkpoint in `fukikae run` (lokkju/fukikae#33, operator ruling D40).

**Architecture:** A new CPU-only `fukikae.pitch` module (autocorrelation F0 track, 60 to 800 Hz, median-based gender with a confidence) replaces the 20th-percentile estimate everywhere. Part A puts it to work in `cards.select` (four reference gates), `cards.auto` (`impure_cards` re-picks, gendered extras, wider pools), `stages/tts.py` (the line gate in `delivery_passed`), `songs.py` (a repeated tail search) and a new `asrclean` stage that drops stock hallucinations over music or silence. `fukikae eval gender` measures all of it. Part B adds registry schema 5 (`gender`, `gender_by`, `kind`, `reviewed`), a `review.json` builder, an HTML client in the listening-pack style, a decisions file applied through the registry API with stage invalidation, pitch splits through the names stage's sub-cluster relabelling, and a `[review] required` gate before translate.

**Tech Stack:** Python 3.12, uv, typer, pydantic, numpy, scipy, sqlite3; existing `fukikae.cards`, `fukikae.registry`, `fukikae.names`, `fukikae.songs`, `fukikae.eval.ab` (page style). No new runtime dependencies; librosa only through `uv run --with librosa` in the calibration script. No GPU in any task except the re-render step of Task 10 (open decision 7).

**Spec:** `docs/superpowers/specs/2026-11-11-identification-workflow-design.md` (draft; this plan follows it and stops only for the open decisions at its end). Rulings R1 to R20 (`docs/superpowers/ledgers/2026-11-05-speaker-identity.md`), GT1 to GT15 (`docs/superpowers/ledgers/2026-11-08-ground-truth-rulings.md`), CD1 to CD26 and D27, D32, D34 (`docs/superpowers/ledgers/2026-11-08-character-cards-rulings.md`), D29 (`docs/superpowers/ledgers/2026-11-10-subtitle-speaker-names.md`) and D33 (CLAUDE.md) stand. Execution ledger: `docs/superpowers/ledgers/2026-11-11-identification-workflow.md` (created by the controller at Task 1's dispatch).

## Global Constraints

- `$REPO` is the checkout the plan runs in (a worktree under `~/projects/fukikae/.claude/worktrees/`). uv only, always `uv run --directory $REPO ...`; never pip, never `cd <dir> && <cmd>`; absolute paths everywhere. Pipeline and media commands through `mise -C $REPO exec -- <cmd>`.
- Tests first, with RED evidence in each task report. Markers as in CLAUDE.md; default runs stay offline, model-free and GPU-free. Pitch tests use synthetic signals written in the test (harmonic tones, noise, tone over music); no media file is committed.
- Real-data runs (Tasks 2, 3, 7, 10) work on hard-link copies (`cp -al`) under `~/fukikae-ws/t-ident/`; source media and the existing workspaces (`~/fukikae-ws/fairytail`, `bbt-en`, `kaze`, `~/fukikae-scratch/sol-bianca`) are read-only. GPU 0 only, only in Task 10's re-render step, only with open decision 7 answered yes; index 1 is vLLM.
- Commits: stage explicit paths with `git -C $REPO add <paths>`, then a plain `git -C $REPO commit -m "<conventional message>"` (no pathspec). No AI attribution, no `Co-Authored-By`, no session links. One committer at a time in a checkout.
- R1: pipeline code never imports `fukikae.eval`. `fukikae.pitch` is pipeline code; `eval/gender.py` imports it, never the reverse.
- R3: `[review]` is a settings table without `backend`, exempt like `[resolve]` and `[names]`.
- CD2, CD3, CD9, D29 #4, D32: every new re-pick reason (impure references, operator gender) re-picks only cards `replace_refusal(a, "auto")` allows (`auto`, `auto-fallback`, `restock`, unlocked); curated, imported and locked cards are flagged with a `fukikae voices clone` hint; a card with no passing pick is kept and flagged, never made stock.
- CD1: no card default changes value. Nothing in this plan adds a card field; pitch facts live in the pitch cache, never in a card id.
- CD5: the A1 gates filter candidates; the rank order among survivors is unchanged.
- CD12: the line gate lives in `delivery_passed`, the one render path the stage and `eval tts` share.
- Every threshold is a module constant named in the spec: `F0_MIN = 60`, `F0_MAX = 800`, `MALE_BELOW = 165`, `FEMALE_ABOVE = 200`, `SPEAKER_MIN_VOICED_S = 1.0`, `LINE_MIN_VOICED_S = 0.4`, `CONFIDENCE_ST = 4.0`, `REF_PITCH_ST = 4.0`, `REF_VOICED_MIN = 0.5`, `REF_SONG_SHARE = 0.5`, `SCENE_GAP_S = 60.0`, `BIMODAL_MIN_S = 5.0`, `BIMODAL_MIN_SHARE = 0.15`, `MERGE_SUGGEST_COS = 0.5`, `HALLUCINATION_MUSIC_SHARE = 0.6`, `HALLUCINATION_ACTIVE_MAX = 0.3`, `tail_window_s = 30`. None is tuned on Fairy Tail E14 or Sol Bianca S00E01; Task 2 may move the two bands only on the tuning episodes, recorded in its results doc.
- Subagents never start background watchers; a long run is one foreground command or one background command that reports on exit.

## Review Focus

1. **Operator state beats every new rule.** No gate, re-pick, gender or kind change touches a curated, imported or locked card; `--apply` refuses a stale decisions file; nothing in the page writes the registry. Pinned in Tasks 5, 6, 12 and 15.
2. **The estimator change moves every gender call.** Bands now read the median; the anime override goes; a workspace that set `[tts.gender]` gets a warning. Check the calibration numbers before the profile edit lands. Pinned in Tasks 1, 2 and 3.
3. **No wrong-gender emotion prompt on any path**, extras and `eval tts` included. Pinned in Task 6.
4. **Existing workspaces.** `asrclean` output is byte-identical to its input when nothing is dropped; the song detector's version bump reruns mix only; registry schema 4 migrates to 5 in one transaction; assignments written without an ignored character are byte-identical. Pinned in Tasks 7, 8, 11 and 12.
5. **No regression on Fairy Tail's names arm**: reference purity and cue accuracy per episode, and every changed E12/E13 card explained by a named gate. Pinned in Task 10.

---

## Decisions this plan makes

- **Module:** `src/fukikae/pitch.py` (`F0Track`, `F0Summary`, `GenderEstimate`, `f0_track`, `summarize`, `classify`, `turn_classes`, `majority_register`, `bimodal`, and a cache under `.fukikae/pitch-cache/`). `cards.select.speaker_f0` becomes a wrapper returning the median.
- **Stage:** `asrclean`, directory `03b-asrclean/`, between `asr` and `diarize` in `pipeline_stages`. `fukikae.transcripts.transcript_path(ep)` is the one way consumers find the transcript.
- **Registry schema 5:** `characters.gender TEXT` (`f`, `m`, NULL), `characters.gender_by TEXT` (`operator`, `cast`, NULL), `characters.kind TEXT NOT NULL DEFAULT 'character'` (`character`, `narrator`, `ignored`), `characters.reviewed TEXT` (ISO timestamp or NULL). `Character` gains the four fields with those defaults.
- **`ClusterAssignment.status`** gains `"ignored"`; the existing `"review"` value keeps its meaning.
- **Files:** `review/<episode>/{review.json,index.html,audio/,decisions.json,splits.json}` under the workspace.
- **Exit code 3** for "stopped for review" in `fukikae run`.

## Dependencies and gating

| Task | GPU | Waits on |
|---|---|---|
| 1 Pitch module | no | |
| 2 Gender calibration on tuning sets | no | 1 |
| 3 A2: extras gender, pools, bands | no | 1, 2 |
| 4 A1: reference gates | no | 1 |
| 5 A1: impure cards re-picked | no | 4 |
| 6 A3: line gate on emotion_audio | no | 1 |
| 7 A4: ending song tail | no | |
| 8 A4: asrclean stage | no | 7 |
| 9 `eval gender` | no | 1 |
| 10 Part A evaluation (Sol Bianca, Fairy Tail) | step 3 only (decision 7) | 3, 5, 6, 8, 9 |
| 11 Registry schema 5 and CLI | no | |
| 12 Kinds and operator gender in the pipeline | no | 3, 5, 6, 11 |
| 13 `review.json` builder | no | 9, 11 |
| 14 Review page and `--serve` | no | 13 |
| 15 Decisions and `--apply` | no | 11, 13 |
| 16 Pitch splits through the names stage | no | 15 |
| 17 `[review]` gate in `fukikae run` | no | 15 |
| 18 Cast priors from `cast.toml` (decision 8) | no | 14 |
| 19 Docs: CLAUDE.md, README | no | all |

Tasks 1, 7 and 11 can start in parallel; 4, 6 and 9 in parallel after 1. Part A (1 to 10) ships before Part B.

## File Structure

```
src/fukikae/pitch.py                 # new: F0 track, summaries, gender, cache
src/fukikae/transcripts.py           # new: transcript_path(ep)
src/fukikae/hallucinations.py        # new: phrase list, drop rule
src/fukikae/stages/asrclean.py       # new: the asrclean stage
src/fukikae/review/__init__.py       # new package
src/fukikae/review/model.py          # review.json and decisions.json schemas
src/fukikae/review/build.py          # review.json from an episode
src/fukikae/review/page.py           # index.html and audio/
src/fukikae/review/apply.py          # decisions -> registry, invalidation
src/fukikae/review/gate.py           # needs_review(ep, cfg)
src/fukikae/review/cli.py            # fukikae review
src/fukikae/eval/gender.py           # eval gender
scripts/gender-calibration/calibrate.py
scripts/gender-calibration/voice_pools.py
modified: cards/select.py cards/auto.py cards/stock.py stages/tts.py songs.py stages/characters.py
          stages/names.py names/cli.py cues.py stages/translate.py registry/store.py registry/cli.py
          config.py job.py cli.py models.py profiles/anime.toml CLAUDE.md README.md
```

---

### Task 1: Pitch module

**Files:**
- Create: `src/fukikae/pitch.py`
- Modify: `src/fukikae/cards/select.py` (`speaker_f0`, `estimate_gender`, `gender_of` delegate)
- Test: `tests/test_pitch.py`, `tests/cards/test_select.py`

**Interfaces (`fukikae.pitch`):**
- Constants as in Global Constraints, plus `VERSION = 1` (in the cache key).
- `@dataclass(frozen=True) class F0Track: f0: np.ndarray` (Hz per 20 ms hop, NaN unvoiced); `span_index: np.ndarray` (which input span each frame came from); `seconds: float`.
- `@dataclass(frozen=True) class F0Summary: median: float | None; p10; p20; p90; voiced_s: float; seconds: float; voiced_fraction: float`.
- `@dataclass(frozen=True) class GenderEstimate: gender: Literal["f","m","u"]; confidence: float; median: float | None; voiced_s: float`.
- `f0_track(vocals: Path, spans: Sequence[tuple[float,float]]) -> F0Track`: the existing autocorrelation per 40 ms frame, range `F0_MIN` to `F0_MAX`, shortest lag within 90% of the best.
- `summarize(track: F0Track, span: int | None = None) -> F0Summary`.
- `classify(summary, male_below=MALE_BELOW, female_above=FEMALE_ABOVE, min_voiced_s=SPEAKER_MIN_VOICED_S) -> GenderEstimate`: median-based; confidence `min(1, |12*log2(median/edge)| / CONFIDENCE_ST)` to the nearest edge, 0 for `u`.
- `turn_classes(track, bands, min_voiced_s=LINE_MIN_VOICED_S) -> list[GenderEstimate]`: one per input span.
- `band_seconds(track, bands) -> dict[str, float]` (m, u, f seconds by turn).
- `majority_register(track, bands) -> tuple[Literal["f","m","u"], F0Summary]`: the majority class by seconds and the summary of that class's turns only.
- `bimodal(seconds: Mapping[str,float]) -> bool`: minor of m/f at least `BIMODAL_MIN_S` and `BIMODAL_MIN_SHARE` of classified seconds.
- `cached_track(root: Path, vocals: Path, spans) -> F0Track`: npz under `<root>/.fukikae/pitch-cache/<sha256(vocals fingerprint, spans, VERSION)>.npz`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pitch.py
def test_harmonic_tone_reads_its_fundamental(tmp_path):
    for hz in (90, 220, 420, 640):            # 420 and 640 were folded under the old 400 Hz cap
        wav = write_harmonic(tmp_path, hz, seconds=2.0)
        s = summarize(f0_track(wav, [(0.0, 2.0)]))
        assert abs(12 * math.log2(s.median / hz)) < 0.5

def test_strong_second_harmonic_does_not_halve_f0(tmp_path): ...   # 200 Hz with a louder 400 Hz partial -> ~200

def test_noise_is_unvoiced(tmp_path):
    s = summarize(f0_track(write_noise(tmp_path, 2.0), [(0, 2)]))
    assert s.median is None and s.voiced_fraction < 0.1

def test_classify_bands_on_the_median():
    assert classify(summary(median=140, voiced_s=3)).gender == "m"
    assert classify(summary(median=240, voiced_s=3)).gender == "f"
    assert classify(summary(median=185, voiced_s=3)).gender == "u"
    assert classify(summary(median=240, voiced_s=0.5)).gender == "u"      # under the speaker minimum

def test_confidence_grows_with_distance_from_the_edge():
    near, far = classify(summary(median=210, voiced_s=3)), classify(summary(median=320, voiced_s=3))
    assert 0 < near.confidence < far.confidence == 1.0

def test_majority_register_ignores_the_minor_mode(tmp_path):
    # 6 turns at 130 Hz, 14 turns at 260 Hz
    g, s = majority_register(track, (165, 200))
    assert g == "f" and abs(s.median - 260) < 8

def test_bimodal_needs_seconds_and_share():
    assert bimodal({"m": 19, "u": 34, "f": 70}) and not bimodal({"m": 6, "u": 47, "f": 177})

def test_cached_track_is_reused_and_keyed_by_spans(tmp_path): ...

# tests/cards/test_select.py
def test_speaker_f0_returns_the_median_now(tmp_path):
    assert abs(speaker_f0(wav_260, [(0, 2)])[0] - 260) < 8
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/test_pitch.py tests/cards/test_select.py -v`
Expected: FAIL (no module `fukikae.pitch`; `speaker_f0` returns the 20th percentile)

- [ ] **Step 3: Implement** the module by moving `_read`, `_frames` and the autocorrelation out of `cards/select.py` and returning frames instead of a percentile; `speaker_f0`, `gender_of` and `estimate_gender` delegate. Update the stock-card note in `cards/auto.py` from "20th percentile" to "median".

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS once tests that pinned 20th-percentile numbers in `tests/cards/test_auto.py` and `tests/stages/test_tts.py` are updated to the median (list each in the task report).

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/pitch.py src/fukikae/cards/select.py src/fukikae/cards/auto.py tests/test_pitch.py tests/cards/test_select.py tests/cards/test_auto.py tests/stages/test_tts.py
git -C $REPO commit -m "feat(pitch): one F0 track estimator to 800 Hz with median-based gender and confidence"
```

---

### Task 2: Gender calibration on the tuning sets

**Files:**
- Create: `scripts/gender-calibration/calibrate.py`, `scripts/gender-calibration/truth.toml`, `docs/results/2026-11-11-gender-bands.md`, `docs/results/2026-11-11-gender-bands.jsonl`

**Interfaces:**
- `calibrate.py <ws> <episode>... --truth truth.toml [--pyin]`: per diarize cluster and per character, the A0 summary and class, the pYIN median and class (`--pyin` imports librosa; run as `uv run --directory $REPO --with librosa python scripts/gender-calibration/calibrate.py ...`), and the truth class; per cue, A0 class against pYIN class.
- `truth.toml`: `[<set>] "<character name>" = "f" | "m"` for main characters (30 s or more), voice register as heard, for Fairy Tail (E12, E13), The Big Bang Theory (en tuning episodes per `docs/results/2026-11-09-live-action-ground-truth.md`) and Kaze (tuning episodes), filled from the sets' ground-truth labels. Each entry the writer is unsure of is marked `# check` for the operator.

- [ ] **Step 1: Write the failing test** `tests/scripts/test_calibrate.py::test_acceptance_rule` on a synthetic table: a main character in the other class fails; a 12% unknown share fails; 90% line agreement passes.
- [ ] **Step 2: Run to verify it fails.** `uv run --directory $REPO pytest tests/scripts/test_calibrate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the script and run it on hard-link copies under `~/fukikae-ws/t-ident/`, CPU only. Sol Bianca S00E01 runs too, reported in its own section and excluded from the acceptance rule.
- [ ] **Step 4: Write the results doc:** per set, the histogram of main-character medians by truth class, the confusion table at 165/200, pYIN agreement per line, and the acceptance verdict. If a set fails at 165/200, report the band that passes on that set's tuning episodes and stop for the operator before Task 3 edits a profile.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add scripts/gender-calibration/calibrate.py scripts/gender-calibration/truth.toml tests/scripts/test_calibrate.py docs/results/2026-11-11-gender-bands.md docs/results/2026-11-11-gender-bands.jsonl
git -C $REPO commit -m "docs(results): gender bands on the median, calibrated on Fairy Tail, BBT and Kaze tuning episodes"
```

---

### Task 3: A2, extras gender, wider pools, the bands

**Files:**
- Modify: `src/fukikae/cards/auto.py` (`extras_voices`, `gender_thresholds`), `src/fukikae/cards/stock.py` (`STOCK_POOLS`, `pick_stock`), `src/fukikae/stages/tts.py` (`_extra_cards`, `config_key`, cue extra label), `src/fukikae/profiles/anime.toml`
- Create: `scripts/gender-calibration/voice_pools.py`
- Test: `tests/cards/test_auto.py`, `tests/cards/test_stock.py`, `tests/stages/test_tts.py`

**Interfaces:**
- `extras_voices(cues, cfg, genders: Mapping[str, GenderEstimate]) -> dict[str, str]`: label to voice. `genders` is keyed by extra label; `unknown` cues are keyed `unknown~f`, `unknown~m`, `unknown~u` by their own line class.
- `extra_label(cue, line_gender) -> str`: the label a cue's extras voice is looked up under.
- `extra_genders(ep, cues, vocals) -> dict[str, GenderEstimate]` in `stages/tts.py`: per extra cluster from its turns in `assignments.json`, per `unknown` cue from its reference span, through `pitch.cached_track`.
- `pick_stock(..., gender, used, extras, median: float | None = None)`: a `u` with a median takes the pool nearer by the geometric midpoint `sqrt(MALE_BELOW * FEMALE_ABOVE)`; alternates only without a median.
- `STOCK_POOLS["kokoro"]["en"]["extras_f"] = ["af_kore", "af_nova", "af_aoede", "bf_lily", "af_river"]`, `["extras_m"] = ["am_echo", "bm_daniel", "am_eric", "am_liam", "am_onyx"]`, minus any voice `voice_pools.py` fails (decision 5).
- `gender_thresholds(cfg)`: defaults `MALE_BELOW`, `FEMALE_ABOVE` from `fukikae.pitch`; logs once per run when the workspace (not the profile) sets `[tts.gender]`: "[tts.gender] now reads the median F0, not the 20th percentile; check male_below and female_above".
- `anime.toml`: `[tts.gender]` table removed (or set to Task 2's bands if Task 2 reported different ones and the operator approved).
- `voice_pools.py`: renders ten fixed English sentences per candidate voice on Kokoro (CPU) into a temp dir and prints each voice's render median and class; a voice outside its pool's class is reported.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cards/test_auto.py
def test_extra_cluster_gets_the_pool_of_its_pitch():
    voices = extras_voices(cues, cfg, {"SPEAKER_08": est("m", 111), "SPEAKER_09": est("f", 289)})
    assert voices["SPEAKER_08"] in STOCK_POOLS["kokoro"]["en"]["extras_m"]
    assert voices["SPEAKER_09"] in STOCK_POOLS["kokoro"]["en"]["extras_f"]

def test_unknown_label_splits_by_line_gender():
    voices = extras_voices(cues_unknown_f_and_m, cfg, genders)
    assert voices["unknown~f"] in extras_f and voices["unknown~m"] in extras_m

def test_unknown_band_with_a_median_takes_the_nearer_pool():
    assert pick_stock("kokoro", "en", "u", [], extras=True, median=193) in extras_f
    assert pick_stock("kokoro", "en", "u", [], extras=True, median=170) in extras_m

def test_workspace_gender_table_warns_about_the_median(caplog): ...

# tests/cards/test_stock.py
def test_extras_pools_are_disjoint_from_character_pools(): ...
def test_af_alloy_is_not_an_extras_voice(): ...

# tests/stages/test_tts.py
def test_extras_map_in_the_stage_key_follows_pitch(tmp_ws): ...    # SPEAKER_08 male source -> extras_m voice
def test_anime_profile_sets_no_gender_override(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/cards/test_auto.py tests/cards/test_stock.py tests/stages/test_tts.py -v`. Expected: FAIL.
- [ ] **Step 3: Run `voice_pools.py`** (CPU, Kokoro) and put its table in the task report; drop any failing voice from the pools.
- [ ] **Step 4: Implement**, then `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cards/auto.py src/fukikae/cards/stock.py src/fukikae/stages/tts.py src/fukikae/profiles/anime.toml scripts/gender-calibration/voice_pools.py tests/cards/test_auto.py tests/cards/test_stock.py tests/stages/test_tts.py
git -C $REPO commit -m "feat(cards): extras voices from the speaker's pitch, wider extras pools, median gender bands"
```

---

### Task 4: A1, reference gates

**Files:**
- Modify: `src/fukikae/cards/select.py` (`eligible_references`, `pick_references`, `reference_report`), `src/fukikae/songs.py` (`song_like_seconds`)
- Test: `tests/cards/test_select.py`, `tests/test_songs.py`

**Interfaces:**
- `songs.song_like_seconds(background, vocals, settings) -> np.ndarray`: the detector's per-second song-like mask over the whole episode (the `good` array of `find_candidates`, exposed).
- `eligible_references(..., register: tuple[Literal["f","m","u"], float | None] | None = None, song_like: np.ndarray | None = None, voicing: Callable[[float,float], float] | None = None, track: F0Track | None = None) -> list[RefPick]`: gates 1 to 3 after the existing checks: off-register over `REF_PITCH_ST`, voiced fraction under `REF_VOICED_MIN`, song-like share at or over `REF_SONG_SHARE` with voicing at or over `voiced_share`. Each refusal is logged at debug with its gate and counted.
- `RefPick` gains `median: float | None = None` (for reports).
- `@dataclass(frozen=True) class Gated: off_register: int; unvoiced: int; song_like: int; top: int` (how many would have ranked in the first three); `eligible_references` returns it through an optional `gated: list[Gated] | None` out-parameter, so existing callers are unchanged.
- `pick_references(..., scene_gap: float | None = SCENE_GAP_S)`: at most one pick per scene (starts within `scene_gap` of a taken pick) until the fill would fall short of the floor; `None` turns the gate off (narrators).
- `reference_report(picks, gated=None)` adds "gated: N off-register, N unvoiced, N song-like" when `gated.top > 0`.
- `cards.auto.character_candidates` and `character_references` compute `register` with `pitch.majority_register` over the character's turns in that episode (bands from `gender_thresholds`) and pass it with `song_like`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cards/test_select.py
def test_off_register_candidate_is_gated(tmp_path):
    # character speech at 240 Hz; a clean, loud 175 Hz turn (narration) would rank first by SNR
    picks = eligible_references(turns, "A", transcript, None, vocals, register=("f", 240.0))
    assert all(abs(12 * math.log2(p.median / 240)) <= 4 for p in picks)

def test_mixed_cluster_uses_its_majority_register(tmp_path): ...    # 2 male + 5 female turns -> only female picks

def test_unvoiced_candidate_is_gated(tmp_path): ...                  # noise-only turn with a transcript

def test_song_like_candidate_is_gated_outside_theme_spans(tmp_path):
    # a sung turn (tone over loud bed, voiced 0.9) mid-episode, not in exclude
    picks = eligible_references(..., song_like=mask)
    assert not any(p.start == sung.start for p in picks)

def test_one_scene_cannot_fill_every_slot():
    ranked = [rp(30, 37), rp(40, 46), rp(60, 66), rp(400, 405)]   # first three within 60 s
    picks = pick_references(ranked, count=3)
    assert [p.start for p in picks][:2] == [30, 400]

def test_same_scene_is_used_when_the_floor_needs_it(): ...           # only one other scene, too short

def test_scene_gate_off_for_narrators(): ...                          # scene_gap=None keeps the SNR order

def test_gates_do_not_reorder_survivors(): ...                        # CD5: survivors keep rank_key order

def test_report_names_gated_top_candidates(): ...

# tests/test_songs.py
def test_song_like_seconds_matches_find_candidates_mask(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/cards/test_select.py tests/test_songs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cards/select.py src/fukikae/cards/auto.py src/fukikae/songs.py tests/cards/test_select.py tests/test_songs.py
git -C $REPO commit -m "feat(cards): gate clone references by pitch register, voicing, song-like audio and scene"
```

---

### Task 5: A1, impure automatic cards are re-picked

**Files:**
- Modify: `src/fukikae/cards/auto.py` (`impure_cards`, `ensure_assignments`, `floor_lines`), `src/fukikae/stages/tts.py` (`_assign`), `src/fukikae/names/cli.py` (`_card_lines`), `src/fukikae/cards/cli.py` (`voices list`)
- Test: `tests/cards/test_auto.py`, `tests/cards/test_cli.py`

**Interfaces:**
- `@dataclass(frozen=True) class Impurity: card_id: str; refs: tuple[dict, ...]` (each `{episode, start, end, gate, median, voiced_fraction}`).
- `impure_cards(ws, reg, store, language) -> dict[int, Impurity]`: automatic clone cards (`chosen_by in AUTOMATIC_BY`) with a reference failing gate 1, 2 or 3 measured on its origin span; cards without an `{episode, start, end}` origin are skipped; curated, imported and locked cards are included for flagging only (the caller decides by `replace_refusal`).
- `PURITY_NOTE = "re-picked: references fail the reference gates"`.
- `ensure_assignments(..., impure: Mapping[int, Impurity] | None = None)`: an allowed card is re-picked when the current rules give gate-passing picks (note `f"{PURITY_NOTE} ({gates})"`, combined with `REPICK_NOTE` and `FLOOR_NOTE` when the card is also stale or below the floor: one re-pick, all reasons); otherwise kept and flagged; a refused card is flagged with the failing reference and `fukikae voices clone`.

- [ ] **Step 1: Write the failing tests**

```python
def test_auto_card_with_an_off_register_reference_is_repicked(ws_sb_like):
    assign_auto_clone(1, origins=[("E01", 29.0, 37.0), ("E01", 41.0, 49.0), ("E01", 60.0, 68.0)])  # 175 Hz narration
    ensure_assignments(reg, store, cfg, spans, vocals, cloning=fake_cloning, impure=impure_cards(ws, reg, store, "en"))
    a = reg.card_for(1, "en")
    assert a.note.startswith(PURITY_NOTE) and a.card_id != old

def test_impure_card_without_a_passing_pick_is_kept_and_flagged(): ...      # never stock

def test_assigned_imported_or_locked_impure_card_is_only_flagged():
    for by in ("assign", "import:pack", "auto-locked"):
        ...
        assert reg.card_for(1, "en").card_id == old and any("voices clone" in f for f in flags)

def test_voices_clone_card_is_never_judged(): ...                            # origin {"source": "voices clone"}

def test_stale_floor_and_impure_together_make_one_repick(): ...

def test_restock_card_is_repicked(): ...                                     # CD9: restock is automatic

# tests/cards/test_cli.py
def test_voices_list_shows_impure_flag(cli_ws): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/cards/test_auto.py tests/cards/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**; `TtsStage._assign` passes `impure_cards(...)` beside `stale_cards` and `below_floor`; `names show` and `voices list` print the flag.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cards/auto.py src/fukikae/stages/tts.py src/fukikae/names/cli.py src/fukikae/cards/cli.py tests/cards/test_auto.py tests/cards/test_cli.py
git -C $REPO commit -m "feat(cards): re-pick automatic clone cards whose references fail the gates; flag curated ones"
```

---

### Task 6: A3, the line gate on emotion_audio

**Files:**
- Modify: `src/fukikae/stages/tts.py` (`delivery_passed`, `prepare_cue`, `run`, `config_key`), `src/fukikae/models.py` (`TtsPlanCue`), `src/fukikae/eval/tts.py` (passes genders through the shared path)
- Test: `tests/stages/test_tts.py`, `tests/eval/test_tts.py`

**Interfaces:**
- `GATE_VERSION = 1` in the tts stage key with the bands.
- `established_gender(cid, reg, card, track, bands) -> Literal["f","m","u"]`: operator gender (after Task 11; until then none), else the card's reference clips' class (measured on the sidecar clips), else `majority_register` of the character's speech.
- `line_genders(vocals, cues, root) -> dict[str, GenderEstimate]`: each spoken cue's `delivery.reference` span, `LINE_MIN_VOICED_S`.
- `delivery_passed(c, card, cls, slot, events=(), line: GenderEstimate | None = None, voice: Literal["f","m","u"] = "u")`: when `line.gender` and `voice` are opposite known classes, no `emotion_audio`; `out["emotion_dropped"] = "gender"`. Applies to every card source, extras included.
- `TtsPlanCue` gains `source_gender: Literal["f","m","u"] | None = None` and `gender_gate: bool = False`, both omitted from JSON at their defaults.
- `run.json` flags: `f"{who}: cue {id} source is {line} ({median:.0f} Hz) against the character's {voice}; rendered without emotion audio"` for character cues (narrators skipped from Task 12 on).

- [ ] **Step 1: Write the failing tests**

```python
def test_wrong_gender_line_passes_no_emotion_audio():
    passed = delivery_passed(cue, clone_card, IndexTts2, 2.0, line=est("m", 130), voice="f")
    assert "emotion_audio" not in passed and passed["emotion_dropped"] == "gender"

def test_unknown_on_either_side_never_gates():
    assert "emotion_audio" in delivery_passed(cue, card, cls, 2.0, line=est("u", 185), voice="f")
    assert "emotion_audio" in delivery_passed(cue, card, cls, 2.0, line=est("m", 130), voice="u")

def test_gated_line_keeps_its_characters_card(tmp_ws):
    plan = run_tts(tmp_ws)
    assert plan.cue("c0290").card_id == card_of_char5 and plan.cue("c0290").gender_gate

def test_gate_flags_land_in_run_json(tmp_ws): ...

def test_extras_never_get_a_wrong_gender_prompt(tmp_ws): ...

def test_only_gated_cues_change_cache_keys(tmp_ws): ...      # other cue_keys identical to before

# tests/eval/test_tts.py
def test_eval_tts_applies_the_same_gate(): ...                # CD12
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/stages/test_tts.py tests/eval/test_tts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/stages/tts.py src/fukikae/models.py src/fukikae/eval/tts.py tests/stages/test_tts.py tests/eval/test_tts.py
git -C $REPO commit -m "feat(tts): never pass wrong-gender source audio as emotion_audio; flag the lines for review"
```

---

### Task 7: A4, the ending song tail

**Files:**
- Modify: `src/fukikae/songs.py` (`SongSettings`, `find_candidates`, `VERSION`), `scripts/song_scores.py`
- Test: `tests/test_songs.py`

**Interfaces:**
- `SongSettings` gains `tail_window_s: int = 30` and `tail_repeat: bool = True`; `VERSION = 3`.
- `find_candidates`: after the tail's best window is accepted, the tail outside accepted spans is searched again with `tail_window_s` under the same `min_score` and `voiced_share` tests, repeated until nothing passes; accepted spans within `grow_s` merge in `_spans`. Each pass's candidates are reported.

- [ ] **Step 1: Write the failing tests**

```python
def test_second_ending_stretch_is_found():
    # synthetic per-second features: song 3658-3740, dialogue 3740-3771, song 3771-3817, EOF 3830
    spans = theme_songs(bg, vo, active, settings=SongSettings(), voiced=lambda a, b: 0.85)
    assert any(s.start <= 3772 and s.end >= 3816 for s in spans)

def test_battle_in_the_tail_is_still_refused(): ...          # score passes, voiced 0.53 -> not accepted

def test_tail_repeat_off_keeps_version_2_behaviour(): ...

def test_detector_constants_carry_version_3(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/test_songs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**, then rerun `scripts/song_scores.py` on hard-link copies (CPU) for Fairy Tail E11 to E14, Kaze E01 to E03, BBT en and de, Grid and Sol Bianca S00E01; the task report lists every span before and after. Acceptance: no new span on BBT or Grid; Fairy Tail and Kaze spans unchanged or longer only into credits; Sol Bianca covers 3771 to 3817 s.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/songs.py scripts/song_scores.py tests/test_songs.py
git -C $REPO commit -m "feat(songs): search the ending again for further song stretches"
```

---

### Task 8: A4, the asrclean stage

**Files:**
- Create: `src/fukikae/hallucinations.py`, `src/fukikae/stages/asrclean.py`, `src/fukikae/transcripts.py`
- Modify: `src/fukikae/cli.py` (`pipeline_stages`), `src/fukikae/stages/characters.py`, `src/fukikae/stages/translate.py`, `src/fukikae/cards/auto.py` (`episode_artifacts`), `src/fukikae/stages/names.py` and any other reader of `03-asr/transcript.json` (grep and list them in the report)
- Test: `tests/test_hallucinations.py`, `tests/stages/test_asrclean.py`

**Interfaces:**
- `HALLUCINATION_PHRASES: dict[str, tuple[str, ...]]` by language, each entry commented with where it was seen; `fold(text) -> str` (NFKC, casefold, spaces and punctuation stripped).
- `is_stock_phrase(text, language) -> bool`: the folded text equals a phrase or is that phrase repeated.
- `drop_reason(seg, language, song_spans, song_like, active) -> str | None`: `"stock phrase over a theme song"`, `"stock phrase over music"` (song-like share at least `HALLUCINATION_MUSIC_SHARE`), `"stock phrase over no speech"` (vocals active under `HALLUCINATION_ACTIVE_MAX`), or None.
- `AsrCleanStage`: name `asrclean`; inputs `03-asr/transcript.json` and the three separation stems; outputs `03b-asrclean/transcript.json` and `03b-asrclean/dropped.json` (`[{start, end, text, reason}]`); key holds the phrase table digest, the thresholds and `songs.detector_constants`.
- `transcript_path(ep) -> Path`: `03b-asrclean/transcript.json`, falling back to `03-asr/transcript.json` with a warning when the episode predates the stage.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hallucinations.py
def test_stock_phrase_folding():
    assert is_stock_phrase("ご視聴ありがとうございました。", "ja")
    assert is_stock_phrase("ご視聴ありがとうございました ご視聴ありがとうございました", "ja")
    assert not is_stock_phrase("ありがとうございました", "ja")

def test_phrase_over_music_is_dropped(): ...
def test_phrase_over_dialogue_is_kept():                     # a character says it to camera, no bed
    assert drop_reason(seg, "ja", [], quiet_mask, active_high) is None
def test_other_text_over_music_is_kept(): ...

# tests/stages/test_asrclean.py
def test_output_is_byte_identical_when_nothing_is_dropped(tmp_ws): ...
def test_dropped_segments_are_listed_with_reasons(tmp_ws): ...
def test_consumers_read_the_cleaned_transcript(tmp_ws): ...  # cues built without the dropped segment
def test_pipeline_order_puts_asrclean_after_asr(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/test_hallucinations.py tests/stages/test_asrclean.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/hallucinations.py src/fukikae/stages/asrclean.py src/fukikae/transcripts.py src/fukikae/cli.py src/fukikae/stages/characters.py src/fukikae/stages/translate.py src/fukikae/stages/names.py src/fukikae/cards/auto.py tests/test_hallucinations.py tests/stages/test_asrclean.py
git -C $REPO commit -m "feat(asrclean): drop stock ASR hallucinations over music or silence before diarize"
```

---

### Task 9: `fukikae eval gender`

**Files:**
- Create: `src/fukikae/eval/gender.py`
- Modify: `src/fukikae/eval/cli.py`
- Test: `tests/eval/test_gender.py`

**Interfaces:**
- `@dataclass class GenderReport`: `cues`, `known`, `mismatched_strict`, `mismatched_cut` (180 Hz), per-character and per-extra rows, `extras_disagree`, `purity: dict[int, tuple[int, int]]` (passing references, total), `gated_lines`, `wrong_gender_prompts`, `dropped_hallucinations`, `stock_cues`, `similarity: dict | None`.
- `measure(ws, episode, *, similarity=False) -> GenderReport`: reads `plan.json`, `07-tts/*.wav`, cards and the vocals; all pitch through `fukikae.pitch`; `--similarity` loads WeSpeaker on the CPU.
- `fukikae eval gender <ws> <episode> [--similarity] [--json PATH] [--md PATH]`.

- [ ] **Step 1: Write the failing tests**: `test_mismatch_counts_on_a_synthetic_episode` (three cues: f source and f render, m source and f render, u source; expect known 2, strict mismatches 1); `test_purity_counts_off_register_refs`; `test_wrong_gender_prompts_reads_plan_passed`; `test_eval_never_imported_by_pipeline` (R1: grep `src/fukikae` outside `eval/` for `fukikae.eval`).
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/eval/test_gender.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**, then run it on the untouched Sol Bianca copy and compare with the diagnostic (its numbers came from pYIN; report both side by side in the task report).
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/eval/gender.py src/fukikae/eval/cli.py tests/eval/test_gender.py
git -C $REPO commit -m "feat(eval): eval gender measures cue gender agreement, reference purity and gated lines"
```

---

### Task 10: Part A evaluation

**Files:**
- Create: `docs/results/2026-11-11-identification-part-a.md`, `docs/results/2026-11-11-identification-part-a.jsonl`
- Modify: the execution ledger

- [ ] **Step 1: Before.** `eval gender` on hard-link copies of Sol Bianca S00E01 and Fairy Tail E11 to E14 (names arm workspace), plus `eval names` cue accuracy and the names ledger's reference purity on Fairy Tail. CPU only.
- [ ] **Step 2: After, CPU part.** On fresh copies at this branch's HEAD: `fukikae run --to translate`, then the tts stage's `prepare` only (card assignment, no render) through `fukikae voices list`; record every card change with its note, extras voices, gated lines and dropped segments.
- [ ] **Step 3: After, render (GPU 0, decision 7).** `fukikae run --from tts` on both copies under the GPU lock; then `eval gender --similarity`.
- [ ] **Step 4: Write the results doc**: the diagnostic's table, before and after, against the spec's targets (strict mismatches 15 or fewer; extras disagreement 3 or fewer; purity 100% on chars 1, 2, 3, 10 or flagged; char 11 gone; wrong-gender prompts 0); Fairy Tail purity and cue accuracy per episode, every changed E12/E13 card with its gate. A target missed is reported as missed, with the cue list, and becomes a ruling in the ledger; nothing is retuned on Sol Bianca or E14.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add docs/results/2026-11-11-identification-part-a.md docs/results/2026-11-11-identification-part-a.jsonl docs/superpowers/ledgers/2026-11-11-identification-workflow.md
git -C $REPO commit -m "docs(results): Part A on Sol Bianca S00E01 and the Fairy Tail names arm"
```

---

### Task 11: Registry schema 5 and CLI

**Files:**
- Modify: `src/fukikae/registry/store.py`, `src/fukikae/registry/cli.py`
- Test: `tests/registry/test_store.py`, `tests/registry/test_cli.py`

**Interfaces (`fukikae.registry.store`):**
- `SCHEMA_VERSION = 5`; `_migrate_4_to_5` adds `gender`, `gender_by`, `kind` (default `'character'`) and `reviewed` to `characters` in one transaction.
- `Character` gains `gender: str | None = None`, `gender_by: str | None = None`, `kind: str = "character"`, `reviewed: str | None = None`.
- `KINDS = {"character", "narrator", "ignored"}`; `GENDER_BY = {"operator", "cast"}`.
- `set_character_gender(cid, gender: Literal["f","m"] | None, by: str = "operator") -> None` (None clears).
- `set_character_kind(cid, kind) -> None`.
- `mark_reviewed(cid, when: str | None = None) -> None`.
- `merge_actors` keeps the kept character's gender, kind and reviewed; when only the dropped one has an operator gender, it moves; two different operator genders refuse (`ValueError` naming both).
- CLI: `fukikae registry gender <ws> <character> f|m|--clear`, `fukikae registry kind <ws> <character> character|narrator|ignored`; `registry show` prints gender (with `gender_by`), kind and reviewed.

- [ ] **Step 1: Write the failing tests**

```python
def test_v4_registry_migrates_to_5_with_defaults(tmp_path):
    with Registry(v4_file) as reg:
        c = reg.characters()[0]
        assert (c.kind, c.gender, c.reviewed) == ("character", None, None)

def test_set_gender_and_clear(): ...
def test_unknown_kind_is_refused(): ...
def test_merge_moves_the_only_operator_gender(): ...
def test_merge_with_two_operator_genders_refuses(): ...

# tests/registry/test_cli.py
def test_registry_gender_and_kind_commands(cli_ws): ...    # exit 0; show prints them
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/registry/test_store.py tests/registry/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS (existing migration tests now end at 5).
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/registry/store.py src/fukikae/registry/cli.py tests/registry/test_store.py tests/registry/test_cli.py
git -C $REPO commit -m "feat(registry): schema 5 with character gender, kind and reviewed"
```

---

### Task 12: Kinds and operator gender in the pipeline

**Files:**
- Modify: `src/fukikae/models.py` (`ClusterAssignment.status`), `src/fukikae/stages/characters.py`, `src/fukikae/cues.py` (`_who`, skip), `src/fukikae/cards/auto.py` (operator gender re-pick, narrator scene gate), `src/fukikae/stages/tts.py` (`established_gender`, narrator flag skip), `src/fukikae/registry/cli.py` (gender/kind invalidation)
- Test: `tests/stages/test_characters.py`, `tests/test_cues.py`, `tests/cards/test_auto.py`, `tests/stages/test_tts.py`

**Interfaces:**
- `ClusterAssignment.status` gains `"ignored"`, set by the characters stage when the matched character's kind is `ignored`.
- Cues of an ignored cluster: `skip = "ignored"`, `text = None` (never translated, never rendered; the mix keeps the original vocals).
- `GENDER_NOTE = "re-picked: operator gender"`: `ensure_assignments(..., genders: Mapping[int, str])` re-picks an allowed automatic stock card of the other gender from the right pool, and an allowed automatic clone card whose references' class is the other one, under the gates with the operator's class as the register; refused cards are flagged.
- Narrators: `pick_references(scene_gap=None)`; no A3 flag; never in merge suggestions (Task 13).
- `registry gender` marks tts stale for the character's episodes; `registry kind` marks characters stale for them.

- [ ] **Step 1: Write the failing tests**

```python
def test_ignored_character_clusters_get_status_ignored(tmp_ws): ...
def test_assignments_without_ignored_characters_are_byte_identical(tmp_ws): ...
def test_ignored_cues_are_skipped_and_never_rendered(tmp_ws): ...
def test_operator_gender_repicks_an_auto_stock_card_of_the_other_gender():
    reg.set_character_gender(4, "m"); ensure_assignments(..., genders={4: "m"})
    assert store.get(reg.card_for(4, "en").card_id).voice.voice in STOCK_POOLS["kokoro"]["en"]["m"]
def test_operator_gender_only_flags_a_locked_card(): ...
def test_narrator_references_may_share_a_scene(): ...
def test_narrator_lines_are_not_flagged_by_the_line_gate(): ...
def test_operator_gender_wins_over_the_estimate_in_the_line_gate(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/stages/test_characters.py tests/test_cues.py tests/cards/test_auto.py tests/stages/test_tts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/models.py src/fukikae/stages/characters.py src/fukikae/cues.py src/fukikae/cards/auto.py src/fukikae/stages/tts.py src/fukikae/registry/cli.py tests/stages/test_characters.py tests/test_cues.py tests/cards/test_auto.py tests/stages/test_tts.py
git -C $REPO commit -m "feat: ignored and narrator characters, and operator gender re-picks automatic cards"
```

---

### Task 13: The `review.json` builder

**Files:**
- Create: `src/fukikae/review/__init__.py`, `src/fukikae/review/model.py`, `src/fukikae/review/build.py`
- Modify: `src/fukikae/names/cli.py` (move `_stale_decision`, `_disagreements`, `_merge_suggestions`, `_card_lines` into `fukikae/names/flags.py`, shared by `names show` and the builder)
- Create: `src/fukikae/names/flags.py`
- Test: `tests/review/test_build.py`, `tests/names/test_cli.py` (unchanged output)

**Interfaces (`fukikae.review.model`, pydantic):**
- `Sample(start, end, transcript, clip)`; `GenderInfo(estimate, confidence, median, seconds: dict[str,float], stored, stored_by)`; `CardInfo(card_id, source, chosen_by, locked, proposed: bool, references: list[Sample])`; `Suggestion(kind: Literal["merge","split"], other: str | None, cosine: float | None, parts: dict | None)`; `Flag(kind, text)`; `Subject(id: str, cluster: str | None, character_id: int | None, kind, name, key, seconds, clean_seconds, turns, cues, samples, gender, card, suggestions, names: list[str], flags)`; `Snapshot(character_id, actor_id, name, name_by, kind, gender, locked)`; `ReviewDoc(schema_version=1, workspace, episode, built, registry_digest, subjects, snapshot, dropped: list[dict])`.
- `build_review(ws, episode, cfg, *, clips=4) -> ReviewDoc` and `write_review(ws, episode, out) -> Path` (owner-only directory, `audio/` clips at 24 kHz mono through `cards.select.cut_clip`, card references copied from sidecars).
- Proposed references for a character without a card: `cards.auto.find_references` with nothing assigned.
- Merge suggestions: three nearest by centroid cosine at or over `MERGE_SUGGEST_COS`, narrators excluded; split suggestions from `pitch.bimodal`.

- [ ] **Step 1: Write the failing tests**: `test_review_lists_every_cluster_and_character`; `test_samples_spread_over_the_episode` (four clips from four different fifths); `test_bimodal_cluster_gets_a_split_suggestion`; `test_merge_suggestion_above_threshold_only`; `test_proposed_references_do_not_assign` (registry unchanged); `test_flags_include_stale_floor_impure_gated_and_dropped`; `test_names_show_output_unchanged_after_the_move`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/review/test_build.py tests/names/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/review/__init__.py src/fukikae/review/model.py src/fukikae/review/build.py src/fukikae/names/flags.py src/fukikae/names/cli.py tests/review/test_build.py tests/names/test_cli.py
git -C $REPO commit -m "feat(review): review.json per episode with samples, gender, cards, suggestions and flags"
```

---

### Task 14: The review page and `--serve`

**Files:**
- Create: `src/fukikae/review/page.py`, `src/fukikae/review/cli.py`
- Modify: `src/fukikae/cli.py` (register `review`), `src/fukikae/eval/ab.py` (export `CSS` and `_player` as public helpers, unchanged output)
- Test: `tests/review/test_page.py`, `tests/review/test_cli.py`

**Interfaces:**
- `render_page(doc: ReviewDoc) -> str`: one HTML file, CSS from `eval.ab`, `review.json` embedded in `<script type="application/json" id="review">`, inline JS (no external URL anywhere) that renders a card per subject with players, the decision controls of the spec's table, a draft kept in `localStorage` (wrapped in try/catch), and an "Export decisions" button that downloads `decisions.json` or, when served, POSTs it.
- `fukikae review <ws> [episodes...] [--out DIR] [--clips N] [--serve] [--port 8765]`: builds pages for the named episodes, or for every episode `review.gate.needs_review` lists (Task 17; until then, every episode with assignments).
- `--serve`: `http.server` bound to 127.0.0.1 only, serving the output directory; `POST /<episode>/decisions` writes `decisions.json` there (size cap 1 MB, JSON validated against `Decisions` before writing).

- [ ] **Step 1: Write the failing tests**: `test_page_embeds_review_json_and_has_no_external_urls` (regex for `http`, `//cdn`); `test_page_has_controls_for_every_decision_op`; `test_serve_binds_loopback_and_validates_posts` (start on port 0, POST bad JSON gets 400, good JSON writes the file); `test_review_command_builds_pages` (CliRunner on a fixture workspace).
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/review -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Check the page by hand in a browser from `file://` and from `--serve` on a fixture workspace; note it in the report.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/review/page.py src/fukikae/review/cli.py src/fukikae/cli.py src/fukikae/eval/ab.py tests/review/test_page.py tests/review/test_cli.py
git -C $REPO commit -m "feat(review): fukikae review builds a local review page per episode"
```

---

### Task 15: Decisions and `--apply`

**Files:**
- Create: `src/fukikae/review/apply.py`
- Modify: `src/fukikae/review/model.py` (`Decision`, `Decisions`), `src/fukikae/review/cli.py`, `src/fukikae/registry/cli.py` (reuse `_stale`, `_remap_assignments`, `_episodes_with_key`)
- Test: `tests/review/test_apply.py`

**Interfaces:**
- `Decision` (discriminated on `op`): `name{character, name}`, `alias{character, name}`, `gender{character, gender}`, `merge{keep, drop}` (character ids), `split{episode, cluster, by: Literal["pitch","cluster"]}`, `kind{character, kind}`, `lock{character}`, `reviewed{character}`. Each carries `key` and `external_id` of the character beside the id (#26).
- `Decisions(schema_version=1, workspace, episode, built, registry_digest, snapshot, decisions)`.
- `plan_apply(reg, ws, decisions) -> tuple[list[str], list[Invalidation]]` (lines, `(episode, stage)` pairs); `check_snapshot(reg, decisions) -> list[str]` (differences).
- `apply_decisions(ws, path, *, dry_run=False, force=False) -> list[str]`: snapshot check (refuse unless `force`), then every change in one `Registry.transaction()`; a registry refusal rolls back all; then `job.mark_stale` for each invalidation; every decided character `mark_reviewed`.
- Merge maps characters to their actors and calls `merge_actors`, then `_remap_assignments`; `split by="cluster"` calls `split_cluster`; `split by="pitch"` writes `splits.json` (Task 16 reads it).
- `fukikae review --apply <file> [--dry-run] [--force]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_name_gender_kind_lock_in_one_transaction(ws): ...
def test_stale_snapshot_refuses_without_force(ws):
    reg.rename_character(3, "Someone")   # after the page was built
    with pytest.raises(ReviewError, match="character 3"): apply_decisions(ws, path)
def test_registry_refusal_rolls_back_everything(ws): ...   # merge into a locked actor
def test_merge_remaps_assignments_and_marks_characters_stale(ws): ...
def test_gender_marks_only_tts_stale(ws): ...
def test_dry_run_writes_nothing(ws): ...
def test_every_decided_character_is_marked_reviewed(ws): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/review/test_apply.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/review/apply.py src/fukikae/review/model.py src/fukikae/review/cli.py src/fukikae/registry/cli.py tests/review/test_apply.py
git -C $REPO commit -m "feat(review): apply operator decisions through the registry and invalidate the right stages"
```

---

### Task 16: Pitch splits through the names stage

**Files:**
- Modify: `src/fukikae/stages/names.py` (inputs, key, relabel), `src/fukikae/names/align.py` (row moves for operator labels), `src/fukikae/names/inputs.py`
- Test: `tests/stages/test_names.py`, `tests/names/test_align.py`

**Interfaces:**
- `operator_splits(ep) -> dict[str, PitchSplit]` from `review/<episode>/splits.json`; `PitchSplit(male_below, female_above)`.
- The names stage applies splits after subtitle names: each turn of the cluster (or of its unnamed remainder) classified by its own median; labels `<cluster>~voice-f` and `<cluster>~voice-m`; `u` turns to the larger part; `clusters.npz` rows moved by span overlap as for names. `LabelInfo.cls` gains `"voice"`. When no names were used, the stage still writes the refined files whenever a split exists.
- The split file's digest joins the names stage key; no split file keeps the key as before.

- [ ] **Step 1: Write the failing tests**: `test_pitch_split_relabels_turns_by_their_median`; `test_unknown_turns_join_the_larger_part`; `test_voice_labels_cannot_clash_with_name_slugs`; `test_no_split_file_leaves_the_key_unchanged`; `test_split_with_subtitle_names_splits_only_the_remainder`; `test_characters_stage_matches_the_parts_separately`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/stages/test_names.py tests/names/test_align.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/stages/names.py src/fukikae/names/align.py src/fukikae/names/inputs.py tests/stages/test_names.py tests/names/test_align.py
git -C $REPO commit -m "feat(names): operator pitch splits as voice sub-clusters"
```

---

### Task 17: The `[review]` gate in `fukikae run`

**Files:**
- Create: `src/fukikae/review/gate.py`
- Modify: `src/fukikae/config.py` (`ReviewSettings`, `review_settings`, `Config.review`), `src/fukikae/job.py` (gate hook), `src/fukikae/cli.py` (exit code 3)
- Test: `tests/review/test_gate.py`, `tests/test_config.py`, `tests/test_job.py`

**Interfaces:**
- `@dataclass(frozen=True) class ReviewSettings: required: bool = False; min_seconds: float = 20.0`; `review_settings(table)` rejects unknown keys; merged profile then workspace; `[review]` exempt from R3.
- `needs_review(ep, cfg, reg) -> list[str]`: reasons (a character of kind `character` or `narrator` with `min_seconds` or more in the episode and `reviewed` NULL; or a split suggestion, impure card or names disagreement newer than its `reviewed`); empty when none.
- `job.run_stages(..., gates: Mapping[str, Callable[[Episode, Config], list[str]]] = {})`: before running a stage named in `gates`, a non-empty result raises `ReviewRequired(episode, reasons)`; `run_episodes` stops only that episode.
- `fukikae run` registers the gate before `translate` when `cfg.review.required`; exits 3 printing each reason and `fukikae review <ws> <episode>`.

- [ ] **Step 1: Write the failing tests**: `test_gate_stops_before_translate_with_unreviewed_characters`; `test_reviewed_characters_pass_on_the_next_episode` (season carry); `test_short_characters_never_hold_an_episode`; `test_ignored_characters_never_hold_an_episode`; `test_multi_episode_run_continues_other_episodes`; `test_required_false_never_stops`; `test_unknown_review_key_is_refused`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/review/test_gate.py tests/test_config.py tests/test_job.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/review/gate.py src/fukikae/config.py src/fukikae/job.py src/fukikae/cli.py tests/review/test_gate.py tests/test_config.py tests/test_job.py
git -C $REPO commit -m "feat(run): opt-in review checkpoint before translate"
```

---

### Task 18: Cast priors from `cast.toml` (decision 8)

**Files:**
- Create: `src/fukikae/review/cast.py`
- Modify: `src/fukikae/review/build.py`, `src/fukikae/review/page.py`
- Test: `tests/review/test_cast.py`

**Interfaces:**
- `CastEntry(name, gender: Literal["f","m"] | None, external_id: str | None, aliases: list[str])`; `read_cast(ws) -> list[CastEntry]` from `<ws>/cast.toml` (`[[cast]]`), absent file gives `[]`.
- `ReviewDoc` gains `cast: list[CastEntry]`; the page offers cast names in the name field, shows "cast lists N speaking roles; this episode has M characters", and turns a cast gender into a one-click `gender` decision. Nothing is applied without the click; an applied cast gender is `by = "operator"`.

- [ ] **Step 1: Write the failing tests**: `test_read_cast_validates_genders`; `test_missing_cast_file_is_empty`; `test_review_doc_carries_the_cast`; `test_cast_gender_is_never_applied_without_a_decision`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/review/test_cast.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/review/cast.py src/fukikae/review/build.py src/fukikae/review/page.py tests/review/test_cast.py
git -C $REPO commit -m "feat(review): name and gender suggestions from a hand-written cast.toml"
```

---

### Task 19: Docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`, the execution ledger

- [ ] **Step 1:** CLAUDE.md gets one-line rules citing D40: pitch is `fukikae.pitch` only (median bands, 60 to 800 Hz); the reference gates and `impure_cards` join the automatic re-pick exceptions (with operator gender as the fourth); no wrong-gender `emotion_audio` on any path; `asrclean` and `transcript_path`; `review/` files, `--apply` as the only writer, `[review]` exempt from R3; gender is voice register.
- [ ] **Step 2:** README: the `fukikae review` workflow (build, open, export, apply, rerun) and `[review] required`.
- [ ] **Step 3:** `uv run --directory $REPO pytest -q`, `uv run --directory $REPO ruff check`, `uv run --directory $REPO ruff format --check`, `uv run --directory $REPO pyright`. Expected: all clean.
- [ ] **Step 4: Commit**

```bash
git -C $REPO add CLAUDE.md README.md docs/superpowers/ledgers/2026-11-11-identification-workflow.md
git -C $REPO commit -m "docs: identification workflow rules and the review checkpoint"
```
