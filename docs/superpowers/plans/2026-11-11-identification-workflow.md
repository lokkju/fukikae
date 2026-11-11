# Character Identification Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the measured identification defects of the Sol Bianca dub automatically (clone references gated by pitch, voicing, songs and scene; a clone from its own speech for every speaker and no stock voice anywhere; a per-line gender rule on `emotion_audio`; ending songs and Whisper's outro hallucinations), then give the operator a season studio, a local web app served by `fukikae`, where identity is fixed across a season and each character's cloned voice is tuned per target language until it is accepted as a curated card, with opt-in gates in `fukikae run` (lokkju/fukikae#33, operator rulings D40 and D41).

**Architecture:** A new CPU-only `fukikae.pitch` module (autocorrelation F0 track, 60 to 800 Hz, median-based gender with a confidence) replaces the 20th-percentile estimate everywhere. Part A puts it to work in `cards.select` (four reference gates and three reference tiers), `cards.auto` (`impure_cards` and stock re-picks, extras clones, no stock branch), `stages/tts.py` (the line gate in `delivery_passed`, a refusal when the clone backend can't run), `songs.py` (a repeated tail search) and a new `asrclean` stage that drops stock hallucinations over music or silence. `fukikae eval gender` measures all of it. Part B adds registry schema 5 (`gender`, `gender_by`, `kind`, `reviewed`, and `studio` as an operator source), a `Decision` model applied through the registry API with stage invalidation, pitch splits through the names stage's sub-cluster relabelling, a `season.json` index, and `fukikae studio`: a loopback `ThreadingHTTPServer` with a token-guarded JSON API, one vendored-Preact page, voice drafts, an audition worker under the GPU lock that renders through the tts stage's own path and cache, and accept; then `[review] identity` and `[review] voices` gates.

**Tech Stack:** Python 3.12, uv, typer, pydantic, numpy, scipy, sqlite3, stdlib `http.server`; existing `fukikae.cards`, `fukikae.registry`, `fukikae.names`, `fukikae.songs`, `fukikae.gpulock`, `fukikae.backends.tts.remote`, `fukikae.eval.ab` (A/B pattern). Preact and htm vendored as ES module files (open decision 6); no node toolchain, no CDN. No new Python runtime dependencies; librosa only through `uv run --with librosa` in the calibration script. No GPU in any task except the re-render step of Task 11 (open decision 8) and the manual audition check in Task 19.

**Spec:** `docs/superpowers/specs/2026-11-11-identification-workflow-design.md` (draft, revised for D41; this plan follows it and stops only for the open decisions at its end). Rulings R1 to R20 (`docs/superpowers/ledgers/2026-11-05-speaker-identity.md`), GT1 to GT15 (`docs/superpowers/ledgers/2026-11-08-ground-truth-rulings.md`), CD1 to CD26 and D27, D32, D34 (`docs/superpowers/ledgers/2026-11-08-character-cards-rulings.md`), D29 (`docs/superpowers/ledgers/2026-11-10-subtitle-speaker-names.md`), D33 (CLAUDE.md) and D41 (the spec) stand. Execution ledger: `docs/superpowers/ledgers/2026-11-11-identification-workflow.md` (created by the controller at Task 1's dispatch).

## Global Constraints

- `$REPO` is the checkout the plan runs in (a worktree under `~/projects/fukikae/.claude/worktrees/`). uv only, always `uv run --directory $REPO ...`; never pip, never `cd <dir> && <cmd>`; absolute paths everywhere. Pipeline and media commands through `mise -C $REPO exec -- <cmd>`.
- Tests first, with RED evidence in each task report. Markers as in CLAUDE.md; default runs stay offline, model-free and GPU-free. Pitch tests use synthetic signals written in the test (harmonic tones, noise, tone over music); no media file is committed. Studio tests start the server on port 0 and use a fake synthesizer (the pattern of `tests/eval/test_tts.py::FakeSynth`).
- Real-data runs (Tasks 2, 5, 8, 11) work on hard-link copies (`cp -al`) under `~/fukikae-ws/t-ident/`; source media and the existing workspaces (`~/fukikae-ws/fairytail`, `bbt-en`, `kaze`, `~/fukikae-scratch/sol-bianca`) are read-only. GPU 0 only, only in Task 11's re-render step (open decision 8 answered yes) and Task 19's manual check; index 1 is vLLM.
- Commits: stage explicit paths with `git -C $REPO add <paths>`, then a plain `git -C $REPO commit -m "<conventional message>"` (no pathspec). No AI attribution, no `Co-Authored-By`, no session links. One committer at a time in a checkout.
- R1: pipeline code never imports `fukikae.eval`. `fukikae.pitch` is pipeline code; `eval/gender.py` imports it, never the reverse. The studio is pipeline code too: it may reuse `eval.ab`'s ideas, not import it (Task 20 moves the shared A/B helper out of `eval`).
- R3: `[review]` is a settings table without `backend`, exempt like `[resolve]` and `[names]`.
- D41: no code path in `src/fukikae` outside `eval/` creates or assigns a stock card. Task 6 adds a test that pins it.
- CD2, CD3, CD9, D29 #4, D32: every re-pick reason (impure references, stock, operator gender) re-picks only cards `replace_refusal(a, "auto")` allows (`auto`, `auto-fallback`, `restock`, unlocked); curated, imported and locked cards are flagged with a hint naming the studio and `fukikae voices clone`; a card with no better pick is kept and flagged, never made stock.
- CD1: no card default changes value. Nothing in this plan adds a card field; pitch facts live in the pitch cache, never in a card id.
- CD5: the A1 gates filter candidates; the rank order among survivors is unchanged.
- CD12: the line gate lives in `delivery_passed`, the one render path the stage, `eval tts` and studio auditions share.
- CD15: the audition worker holds `gpu_lock` for as long as its model is loaded.
- CD24: nothing fetches at run time, the studio page included: no external URL in any shipped file.
- Every threshold is a module constant named in the spec: `F0_MIN = 60`, `F0_MAX = 800`, `MALE_BELOW = 165`, `FEMALE_ABOVE = 200`, `SPEAKER_MIN_VOICED_S = 1.0`, `LINE_MIN_VOICED_S = 0.4`, `CONFIDENCE_ST = 4.0`, `REF_PITCH_ST = 4.0`, `REF_VOICED_MIN = 0.5`, `REF_SONG_SHARE = 0.5`, `SCENE_GAP_S = 60.0`, `CLONE_MIN_S = 3.0`, `THIN_MAX_S = 15.0`, `BIMODAL_MIN_S = 5.0`, `BIMODAL_MIN_SHARE = 0.15`, `MERGE_SUGGEST_COS = 0.5`, `HALLUCINATION_MUSIC_SHARE = 0.6`, `HALLUCINATION_ACTIVE_MAX = 0.3`, `tail_window_s = 30`, `AUDITION_IDLE_S = 300`. None is tuned on Fairy Tail E14 or Sol Bianca S00E01; Task 2 may move the two bands only on the tuning episodes, recorded in its results doc.
- Subagents never start background watchers; a long run is one foreground command or one background command that reports on exit.

## Review Focus

1. **Operator state beats every new rule.** No gate, re-pick, gender or kind change touches a curated, imported or locked card; a stale decision is refused; nothing in the studio page writes a file. Pinned in Tasks 4, 6, 13, 14 and 20.
2. **No stock voice on any path (D41).** No automatic stock card, no extras pool, no fallback stand-in, no `voices assign stock:`; a missing clone backend stops tts. Pinned in Tasks 5 and 6.
3. **The estimator change moves every gender call.** Bands now read the median; the anime override goes; a workspace that set `[tts.gender]` gets a warning. Check the calibration numbers before the profile edit lands. Pinned in Tasks 1 and 2.
4. **No wrong-gender emotion prompt on any path**, extras, `eval tts` and auditions included. Pinned in Tasks 7 and 19.
5. **Existing workspaces.** Automatic stock cards are re-picked once and say why; `asrclean` output is byte-identical to its input when nothing is dropped; the song detector's version bump reruns mix only; registry schema 4 migrates to 5 in one transaction; assignments written without an ignored character are byte-identical. Pinned in Tasks 6, 8, 9, 12 and 13.
6. **The studio's two constraints.** Loopback and token only; one writer (`studio.apply`) with a digest check; the GPU lock held only while the audition model is loaded and released on idle. Pinned in Tasks 14, 17 and 19.
7. **No regression on Fairy Tail's names arm**: reference purity and cue accuracy per episode, and every changed E12/E13 card explained by a named gate or re-pick reason. Pinned in Task 11.

---

## Decisions this plan makes

- **Module:** `src/fukikae/pitch.py` (`F0Track`, `F0Summary`, `GenderEstimate`, `f0_track`, `summarize`, `classify`, `turn_classes`, `majority_register`, `bimodal`, and a cache under `.fukikae/pitch-cache/`). `cards.select.speaker_f0` becomes a wrapper returning the median.
- **Reference tiers:** `cards.select.Tier = Literal["floor", "short", "thin"]`; `cards.select.tiered_references(...)` is the one entry point for characters and extras.
- **Stage:** `asrclean`, directory `03b-asrclean/`, between `asr` and `diarize` in `pipeline_stages`. `fukikae.transcripts.transcript_path(ep)` is the one way consumers find the transcript.
- **Registry schema 5:** `characters.gender TEXT` (`f`, `m`, NULL), `characters.gender_by TEXT` (`operator`, `cast`, NULL), `characters.kind TEXT NOT NULL DEFAULT 'character'` (`character`, `narrator`, `ignored`), `characters.reviewed TEXT` (ISO timestamp or NULL). `Character` gains the four fields with those defaults. `OPERATOR_BY` gains `"studio"`.
- **`ClusterAssignment.status`** gains `"ignored"`; the existing `"review"` value keeps its meaning.
- **Package:** `src/fukikae/studio/` (`model.py`, `apply.py`, `season.py`, `server.py`, `drafts.py`, `audition.py`, `gate.py`, `cast.py`, `cli.py`, `static/`).
- **Files:** `<ws>/studio/{season.json,decisions.jsonl,drafts/,splits/,audio/}`.
- **Exit codes:** 3 for "stopped by a `[review]` gate" in `fukikae run`; the tts stage's refusal without a usable clone backend is a `StageError` (the existing nonzero exit).

## Dependencies and gating

| Task | GPU | Waits on |
|---|---|---|
| 1 Pitch module | no | |
| 2 Gender calibration and the bands | no | 1 |
| 3 A1: reference gates | no | 1 |
| 4 A1: impure cards re-picked | no | 3 |
| 5 A2: a clone for every speaker (tiers, extras, cluster-less lines) | no | 3 |
| 6 A2: stock retired (refusal, CLI, packs, re-picks) | no | 5 |
| 7 A3: line gate on emotion_audio | no | 1 |
| 8 A4: ending song tail | no | |
| 9 A4: asrclean stage | no | 8 |
| 10 `eval gender` | no | 1 |
| 11 Part A evaluation (Sol Bianca, Fairy Tail) | step 3 only (decision 8) | 2, 4, 6, 7, 9, 10 |
| 12 Registry schema 5 and CLI | no | |
| 13 Kinds and operator gender in the pipeline | no | 4, 6, 7, 12 |
| 14 Decisions and `fukikae studio apply` | no | 12 |
| 15 Pitch splits through the names stage | no | 14 |
| 16 The season index | no | 10, 12 |
| 17 Studio server, page shell, identity panel | no | 14, 16 |
| 18 Reference candidates and drafts | no | 5, 17 |
| 19 Audition worker | manual check only | 7, 18 |
| 20 Compare, accept, other languages | no | 19 |
| 21 `[review]` gates in `fukikae run` | no | 14, 16 |
| 22 Cast priors from `cast.toml` (decision 10) | no | 17 |
| 23 Docs: CLAUDE.md, README | no | all |

Tasks 1, 8 and 12 can start in parallel; 3, 7 and 10 in parallel after 1. Part A (1 to 11) ships before Part B, and each Part A task is shippable alone after its dependencies.

## File Structure

```
src/fukikae/pitch.py                 # new: F0 track, summaries, gender, cache
src/fukikae/transcripts.py           # new: transcript_path(ep)
src/fukikae/hallucinations.py        # new: phrase list, drop rule
src/fukikae/stages/asrclean.py       # new: the asrclean stage
src/fukikae/studio/__init__.py       # new package
src/fukikae/studio/model.py          # Decision, Decisions, SeasonDoc, Draft schemas
src/fukikae/studio/apply.py          # decisions -> registry, digests, invalidation, decisions.jsonl
src/fukikae/studio/season.py         # season.json from the workspace
src/fukikae/studio/server.py         # loopback HTTP server, token, JSON API, static files
src/fukikae/studio/drafts.py         # reference candidates, draft versions, draft cards
src/fukikae/studio/audition.py       # audition worker: GPU lock, idle unload, queue, shared render path
src/fukikae/studio/gate.py           # needs_identity(ep, ...), needs_voices(ep, ...)
src/fukikae/studio/cast.py           # cast.toml
src/fukikae/studio/cli.py            # fukikae studio, fukikae studio apply
src/fukikae/studio/static/           # index.html, app.js, vendor/ (preact, htm, VERSIONS)
src/fukikae/abpair.py                # new: blind A/B helper moved out of eval.ab (R1)
src/fukikae/names/flags.py           # new: names show's flag lines, shared with the season index
src/fukikae/eval/gender.py           # eval gender
scripts/gender-calibration/calibrate.py
modified: cards/select.py cards/auto.py cards/stock.py cards/cli.py cards/packs.py stages/tts.py
          songs.py stages/characters.py stages/names.py names/cli.py cues.py stages/translate.py
          registry/store.py registry/cli.py config.py job.py cli.py models.py eval/ab.py
          profiles/default.toml profiles/anime.toml profiles/live-action.toml CLAUDE.md README.md
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

### Task 2: Gender calibration and the bands

**Files:**
- Create: `scripts/gender-calibration/calibrate.py`, `scripts/gender-calibration/truth.toml`, `docs/results/2026-11-11-gender-bands.md`, `docs/results/2026-11-11-gender-bands.jsonl`
- Modify: `src/fukikae/cards/auto.py` (`gender_thresholds`), `src/fukikae/profiles/anime.toml`
- Test: `tests/scripts/test_calibrate.py`, `tests/cards/test_auto.py`, `tests/stages/test_tts.py`

**Interfaces:**
- `calibrate.py <ws> <episode>... --truth truth.toml [--pyin]`: per diarize cluster and per character, the A0 summary and class, the pYIN median and class (`--pyin` imports librosa; run as `uv run --directory $REPO --with librosa python scripts/gender-calibration/calibrate.py ...`), and the truth class; per cue, A0 class against pYIN class.
- `truth.toml`: `[<set>] "<character name>" = "f" | "m"` for main characters (30 s or more), voice register as heard, for Fairy Tail (E12, E13), The Big Bang Theory (en tuning episodes per `docs/results/2026-11-09-live-action-ground-truth.md`) and Kaze (tuning episodes), filled from the sets' ground-truth labels. Each entry the writer is unsure of is marked `# check` for the operator.
- `gender_thresholds(cfg)`: defaults `MALE_BELOW`, `FEMALE_ABOVE` from `fukikae.pitch`; logs once per run when the workspace (not the profile) sets `[tts.gender]`: "[tts.gender] now reads the median F0, not the 20th percentile; check male_below and female_above".
- `anime.toml`: `[tts.gender]` table removed (or set to this task's bands if the calibration reported different ones and the operator approved, open decision 11).

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_calibrate.py
def test_acceptance_rule():
    assert not accept(table(main_in_other_class=1))
    assert not accept(table(unknown_share=0.12))
    assert accept(table(line_agreement=0.90))

# tests/cards/test_auto.py
def test_default_bands_come_from_pitch():
    assert gender_thresholds(cfg_without_gender) == (MALE_BELOW, FEMALE_ABOVE)

def test_workspace_gender_table_warns_about_the_median(caplog): ...
def test_profile_gender_table_does_not_warn(caplog): ...

# tests/stages/test_tts.py
def test_anime_profile_sets_no_gender_override(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/scripts/test_calibrate.py tests/cards/test_auto.py tests/stages/test_tts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the script** and run it on hard-link copies under `~/fukikae-ws/t-ident/`, CPU only. Sol Bianca S00E01 runs too, reported in its own section and excluded from the acceptance rule.
- [ ] **Step 4: Write the results doc:** per set, the histogram of main-character medians by truth class, the confusion table at 165/200, pYIN agreement per line, and the acceptance verdict. If a set fails at 165/200, report the band that passes on that set's tuning episodes and stop for the operator before Step 5 edits a profile.
- [ ] **Step 5: Implement the bands** (`gender_thresholds`, the warning, the anime profile), then `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 6: Commit**

```bash
git -C $REPO add scripts/gender-calibration/calibrate.py scripts/gender-calibration/truth.toml tests/scripts/test_calibrate.py docs/results/2026-11-11-gender-bands.md docs/results/2026-11-11-gender-bands.jsonl src/fukikae/cards/auto.py src/fukikae/profiles/anime.toml tests/cards/test_auto.py tests/stages/test_tts.py
git -C $REPO commit -m "feat(pitch): median gender bands calibrated on Fairy Tail, BBT and Kaze tuning episodes"
```

---

### Task 3: A1, reference gates

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

### Task 4: A1, impure automatic cards are re-picked

**Files:**
- Modify: `src/fukikae/cards/auto.py` (`impure_cards`, `ensure_assignments`, `floor_lines`), `src/fukikae/stages/tts.py` (`_assign`), `src/fukikae/names/cli.py` (`_card_lines`), `src/fukikae/cards/cli.py` (`voices list`)
- Test: `tests/cards/test_auto.py`, `tests/cards/test_cli.py`

**Interfaces:**
- `@dataclass(frozen=True) class Impurity: card_id: str; refs: tuple[dict, ...]` (each `{episode, start, end, gate, median, voiced_fraction}`).
- `impure_cards(ws, reg, store, language) -> dict[int, Impurity]`: automatic clone cards (`chosen_by in AUTOMATIC_BY`) with a reference failing gate 1, 2 or 3 measured on its origin span; cards without an `{episode, start, end}` origin are skipped; curated, imported and locked cards are included for flagging only (the caller decides by `replace_refusal`).
- `PURITY_NOTE = "re-picked: references fail the reference gates"`.
- `ensure_assignments(..., impure: Mapping[int, Impurity] | None = None)`: an allowed card is re-picked when the current rules give gate-passing picks (note `f"{PURITY_NOTE} ({gates})"`, combined with `REPICK_NOTE` and `FLOOR_NOTE` when the card is also stale or below the floor: one re-pick, all reasons); otherwise kept and flagged; a refused card is flagged with the failing reference, `fukikae studio` and `fukikae voices clone`.

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

### Task 5: A2, a clone for every speaker

**Files:**
- Modify: `src/fukikae/cards/select.py` (`tiered_references`, `Tier`, `CLONE_MIN_S`, `THIN_MAX_S`), `src/fukikae/cards/auto.py` (`find_references`, `ensure_assignments`, `CharacterRefs`, `floor_lines`, new `extras_cards`, new `thin_lines`), `src/fukikae/stages/tts.py` (`_extra_cards`, `config_key`, cue lookup for `unknown`), `src/fukikae/names/cli.py` (`_card_lines`), `src/fukikae/cards/cli.py` (`voices list`)
- Test: `tests/cards/test_select.py`, `tests/cards/test_auto.py`, `tests/stages/test_tts.py`, `tests/cards/test_cli.py`

**Interfaces:**
- `Tier = Literal["floor", "short", "thin"]`; `CLONE_MIN_S = 3.0` (open decision 3); `THIN_MAX_S = 15.0` (IndexTTS2's prompt limit).
- `tiered_references(turns, label, transcript, tags, vocals, *, register, song_like, track, theme_spans, foreign, scene_gap=SCENE_GAP_S) -> tuple[Tier, list[RefPick]]`: tier `floor` when the gated eligible picks reach `FLOOR_CLIPS` and `FLOOR_S`; `short` when they reach `CLONE_MIN_S` (after D32's fragment join); else `thin`: the label's turns by length, longest first, up to `THIN_MAX_S`, excluding only spans overlapping `theme_spans` and spans with `STALE_OVERLAP` or more under `foreign`; when that leaves nothing, the label's turns regardless. Never returns an empty pick list for a label with any turn.
- `CharacterRefs` gains `tier: Tier`. `MIN_CLONE_S` is removed; `find_references` returns picks for every character with a turn.
- `THIN_NOTE = "thin reference"`; `thin_lines(reg, store, ids, language) -> list[str]`: `f"character {label}: card {id} is a thin-reference clone ({seconds:.1f} s from {n} turns); listen in `fukikae studio`, or `fukikae voices clone` with a better clip"`; printed by `voices list`, `names show` and `run.json` flags beside `floor_lines`.
- `extras_cards(ep, cues, cfg, store, cloning) -> tuple[dict[str, str], list[str]]` in `cards.auto`: extras label to clone card id, plus flag lines. An extras cluster's card comes from `tiered_references` over its turns in `assignments.json` with its own `majority_register`. Each `unknown` cue gets the label `unknown~<cue id>` and a card cloned from its own `delivery.reference` span (tier `thin`); one flag line per episode counts them (open decision 4). Cards go in the store; no registry row.
- `TtsStage._extra_cards` calls `extras_cards`; the stage key's `extras` entry holds the label to card id map (card ids are content hashes, so the key is stable while the turns are).
- The extras tier and flags are in `run.json`; `plan.json` per cue gains nothing new (the card id already names the voice).

- [ ] **Step 1: Write the failing tests**

```python
# tests/cards/test_select.py
def test_floor_tier_when_the_floor_is_reached(): ...
def test_short_tier_between_three_seconds_and_the_floor():
    tier, picks = tiered_references(turns_4s_clean, ...)
    assert tier == "short" and sum(p.seconds for p in picks) >= CLONE_MIN_S

def test_thin_tier_sets_the_gates_aside_but_not_songs_or_foreign_names():
    # 2.1 s of speech: one off-register turn, one turn in a theme span, one under another's name
    tier, picks = tiered_references(...)
    assert tier == "thin"
    assert not any(overlaps(p, theme) or overlaps(p, foreign) for p in picks)
    assert any(p.start == off_register.start for p in picks)

def test_thin_tier_never_returns_nothing_for_a_speaker_with_turns(): ...   # all turns in a song
def test_thin_tier_stops_at_fifteen_seconds(): ...

# tests/cards/test_auto.py
def test_character_under_six_seconds_gets_a_clone_not_stock(ws):
    ensure_assignments(..., cloning=fake_cloning)
    card = store.get(reg.card_for(7, "en").card_id)
    assert card.voice.source == "clone"

def test_thin_card_is_flagged_in_thin_lines(ws): ...
def test_extras_cluster_gets_a_clone_from_its_own_turns(tmp_ws):
    cards, _ = extras_cards(ep, cues, cfg, store, fake_cloning)
    assert store.get(cards["SPEAKER_08"]).voice.source == "clone"
    assert all(ref_origin_in(cards["SPEAKER_08"], "SPEAKER_08 turns"))

def test_unknown_cue_is_cloned_from_its_own_span(tmp_ws):
    cards, flags = extras_cards(...)
    assert origin_of(cards["unknown~c0101"]) == span_of("c0101")
    assert any("cloned from themselves" in f for f in flags)

def test_extras_cards_are_stable_across_runs(tmp_ws): ...   # same turns, same card ids

# tests/stages/test_tts.py
def test_stage_key_extras_map_holds_clone_card_ids(tmp_ws): ...
def test_extras_render_with_the_clone_backend(tmp_ws): ...

# tests/cards/test_cli.py
def test_voices_list_shows_the_thin_flag(cli_ws): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/cards/test_select.py tests/cards/test_auto.py tests/stages/test_tts.py tests/cards/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The stock branch of `ensure_assignments` is left in place for Task 6 to remove; after this task it is unreachable while a clone backend is usable (pinned by `test_character_under_six_seconds_gets_a_clone_not_stock`).
- [ ] **Step 4: Measure** on a hard-link copy of Sol Bianca S00E01, CPU only: `fukikae voices list` after the tts stage's `prepare` (no render); the task report lists every speaker's tier and seconds, extras included.
- [ ] **Step 5: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 6: Commit**

```bash
git -C $REPO add src/fukikae/cards/select.py src/fukikae/cards/auto.py src/fukikae/stages/tts.py src/fukikae/names/cli.py src/fukikae/cards/cli.py tests/cards/test_select.py tests/cards/test_auto.py tests/stages/test_tts.py tests/cards/test_cli.py
git -C $REPO commit -m "feat(cards): clone every speaker from its own speech, extras and thin speakers included"
```

---

### Task 6: A2, stock voices retired

**Files:**
- Modify: `src/fukikae/cards/auto.py` (remove the stock branch, `FALLBACK_BY` writes, `extras_voices`; add `STOCK_NOTE`, `stock_cards`), `src/fukikae/cards/stock.py` (remove `STOCK_POOLS` and `pick_stock`; keep `stock_card`, `parse_stock_ref` for reading and eval), `src/fukikae/stages/tts.py` (`_assign`: stock re-picks, refusal without a usable clone backend), `src/fukikae/config.py` (`[tts] backend` warning), `src/fukikae/cards/cli.py` (`voices assign` refuses `stock:`; `voices restock` removed), `src/fukikae/cards/packs.py` (refuse stock cards on import), `src/fukikae/profiles/default.toml`, `src/fukikae/profiles/anime.toml`, `src/fukikae/profiles/live-action.toml` (drop `backend = "kokoro"` and its comment)
- Test: `tests/cards/test_auto.py`, `tests/stages/test_tts.py`, `tests/cards/test_cli.py`, `tests/cards/test_packs.py`, `tests/test_config.py`, `tests/test_no_stock.py`

**Interfaces:**
- `STOCK_NOTE = "re-picked: stock voices are retired (D41)"`; `stock_cards(reg, store, ids, language) -> dict[int, str]`: characters whose assigned card's `voice.source == "stock"`, with the card id.
- `ensure_assignments(..., stock: Mapping[int, str] | None = None)`: an allowed card (`replace_refusal(a, "auto")` is None) is re-picked through `tiered_references`, noted `STOCK_NOTE` (combined with `REPICK_NOTE`, `FLOOR_NOTE` and `PURITY_NOTE` in one re-pick); a refused card (curated, imported, locked) is kept and flagged: `f"character {label} speaks with a stock card ({by}{', locked' if locked else ''}); stock voices are retired (D41): accept a clone in `fukikae studio` or use `fukikae voices clone`"`.
- `TtsStage.prepare`/`_assign`: when `[tts] clone_backend` is unset, or `workspace_cloning(...).unavailable` is set, and any spoken cue needs a card made or re-picked (characters or extras), raise `StageError(f"the clone backend {name} can't run here ({why}); run `fukikae backends setup {name}`; fukikae renders no stock voices (D41)")` (or "set [tts] clone_backend") before any model loads. An episode whose every card already exists and checks out renders as before.
- `config`: a workspace `[tts] backend` logs once per run "[tts] backend no longer chooses any card (D41); clone_backend makes every voice". The key stays accepted.
- `voices assign <ws> <character> stock:...` exits 1: "stock voices are retired (D41); `fukikae voices clone` or `fukikae studio` makes a clone" (open decision 5). `voices restock` is removed from the CLI. `packs` import of a card whose source is `stock` raises `PackError` with the same wording.
- `tests/test_no_stock.py::test_pipeline_never_creates_stock_cards`: greps `src/fukikae` outside `eval/` and `cards/stock.py` for `stock_card(` and `pick_stock`, and fails on any hit.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cards/test_auto.py
def test_auto_stock_card_is_repicked_as_a_clone(ws):
    assign_auto_stock(4, "af_heart")
    ensure_assignments(..., cloning=fake_cloning, stock=stock_cards(reg, store, [4], "en"))
    a = reg.card_for(4, "en")
    assert store.get(a.card_id).voice.source == "clone" and a.note.startswith(STOCK_NOTE)

def test_auto_fallback_card_is_repicked_as_a_clone(ws): ...
def test_restock_card_is_repicked_as_a_clone(ws): ...                         # CD9: automatic
def test_assigned_or_locked_stock_card_is_kept_and_flagged(ws):
    for by, locked in (("assign", True), ("import:pack", False), ("auto", True)):
        ...
        assert reg.card_for(4, "en").card_id == old and any("D41" in f for f in flags)

def test_stock_and_floor_together_make_one_repick(ws): ...

# tests/stages/test_tts.py
def test_missing_clone_backend_refuses_before_any_model_loads(tmp_ws):
    with pytest.raises(StageError, match="backends setup indextts2"):
        prepare_tts(tmp_ws, cloning_unavailable="worker env missing")
    assert not fake_synth.loaded

def test_unset_clone_backend_refuses(tmp_ws): ...
def test_episode_with_every_card_present_still_renders_without_cloning(tmp_ws): ...

# tests/cards/test_cli.py
def test_voices_assign_refuses_stock(cli_ws): ...      # exit 1, message names D41
def test_voices_restock_is_gone(cli_ws): ...

# tests/cards/test_packs.py
def test_pack_import_refuses_a_stock_card(tmp_path): ...

# tests/test_config.py
def test_workspace_tts_backend_warns(caplog): ...
def test_profiles_set_no_tts_backend(): ...

# tests/test_no_stock.py
def test_pipeline_never_creates_stock_cards(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/cards tests/stages/test_tts.py tests/test_config.py tests/test_no_stock.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Remove tests that pinned stock behaviour (`extras_voices`, pools, the under-6 s stock card, `auto-fallback` creation, `voices restock`) and list each removed test in the task report with the D41 reason.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cards/auto.py src/fukikae/cards/stock.py src/fukikae/stages/tts.py src/fukikae/config.py src/fukikae/cards/cli.py src/fukikae/cards/packs.py src/fukikae/profiles/default.toml src/fukikae/profiles/anime.toml src/fukikae/profiles/live-action.toml tests/cards/test_auto.py tests/stages/test_tts.py tests/cards/test_cli.py tests/cards/test_packs.py tests/test_config.py tests/test_no_stock.py
git -C $REPO commit -m "feat(cards)!: retire stock voices; re-pick automatic stock cards as clones (D41)"
```

---

### Task 7: A3, the line gate on emotion_audio

**Files:**
- Modify: `src/fukikae/stages/tts.py` (`delivery_passed`, `prepare_cue`, `run`, `config_key`), `src/fukikae/models.py` (`TtsPlanCue`), `src/fukikae/eval/tts.py` (passes genders through the shared path; the studio's auditions use the same path from Task 19)
- Test: `tests/stages/test_tts.py`, `tests/eval/test_tts.py`

**Interfaces:**
- `GATE_VERSION = 1` in the tts stage key with the bands.
- `established_gender(cid, reg, card, track, bands) -> Literal["f","m","u"]`: operator gender (after Task 12; until then none), else the card's reference clips' class (measured on the sidecar clips), else `majority_register` of the character's speech.
- `line_genders(vocals, cues, root) -> dict[str, GenderEstimate]`: each spoken cue's `delivery.reference` span, `LINE_MIN_VOICED_S`.
- `delivery_passed(c, card, cls, slot, events=(), line: GenderEstimate | None = None, voice: Literal["f","m","u"] = "u")`: when `line.gender` and `voice` are opposite known classes, no `emotion_audio`; `out["emotion_dropped"] = "gender"`. Applies to every card, extras clones included; a cluster-less line cloned from its own span never triggers it (same audio).
- `TtsPlanCue` gains `source_gender: Literal["f","m","u"] | None = None` and `gender_gate: bool = False`, both omitted from JSON at their defaults.
- `run.json` flags: `f"{who}: cue {id} source is {line} ({median:.0f} Hz) against the character's {voice}; rendered without emotion audio"` for character cues (narrators skipped from Task 13 on).

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

### Task 8: A4, the ending song tail

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

### Task 9: A4, the asrclean stage

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

### Task 10: `fukikae eval gender`

**Files:**
- Create: `src/fukikae/eval/gender.py`
- Modify: `src/fukikae/eval/cli.py`
- Test: `tests/eval/test_gender.py`

**Interfaces:**
- `@dataclass class GenderReport`: `cues`, `known`, `mismatched_strict`, `mismatched_cut` (180 Hz), per-character and per-extra rows, `extras_disagree`, `purity: dict[int, tuple[int, int]]` (passing references, total), `tiers: dict[str, tuple[Tier, float]]` (speaker to tier and seconds, extras included), `gated_lines`, `wrong_gender_prompts`, `dropped_hallucinations`, `stock_cues` (must be 0 after Task 6), `similarity: dict | None` (with the cues that were stock-voiced before reported apart, given `--before <plan.json>`).
- `measure(ws, episode, *, similarity=False) -> GenderReport`: reads `plan.json`, `07-tts/*.wav`, cards and the vocals; all pitch through `fukikae.pitch`; `--similarity` loads WeSpeaker on the CPU.
- `fukikae eval gender <ws> <episode> [--similarity] [--before PATH] [--json PATH] [--md PATH]`.

- [ ] **Step 1: Write the failing tests**: `test_mismatch_counts_on_a_synthetic_episode` (three cues: f source and f render, m source and f render, u source; expect known 2, strict mismatches 1); `test_purity_counts_off_register_refs`; `test_wrong_gender_prompts_reads_plan_passed`; `test_stock_cues_and_tiers_are_counted`; `test_former_stock_cues_are_reported_apart_with_before`; `test_eval_never_imported_by_pipeline` (R1: grep `src/fukikae` outside `eval/` for `fukikae.eval`).
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/eval/test_gender.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**, then run it on the untouched Sol Bianca copy and compare with the diagnostic (its numbers came from pYIN; report both side by side in the task report).
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/eval/gender.py src/fukikae/eval/cli.py tests/eval/test_gender.py
git -C $REPO commit -m "feat(eval): eval gender measures cue gender agreement, reference purity, tiers, stock cues and gated lines"
```

---

### Task 11: Part A evaluation

**Files:**
- Create: `docs/results/2026-11-11-identification-part-a.md`, `docs/results/2026-11-11-identification-part-a.jsonl`
- Modify: the execution ledger

- [ ] **Step 1: Before.** `eval gender` on hard-link copies of Sol Bianca S00E01 and Fairy Tail E11 to E14 (names arm workspace), plus `eval names` cue accuracy and the names ledger's reference purity on Fairy Tail. CPU only.
- [ ] **Step 2: After, CPU part.** On fresh copies at this branch's HEAD: `fukikae run --to translate`, then the tts stage's `prepare` only (card assignment, no render) through `fukikae voices list`; record every card change with its note (stock re-picks included), every speaker's tier, the extras cards, gated lines and dropped segments.
- [ ] **Step 3: After, render (GPU 0, decision 8).** `fukikae run --from tts` on both copies under the GPU lock (extras now render on IndexTTS2 too); then `eval gender --similarity --before <the before plan.json>`.
- [ ] **Step 4: Write the results doc**: the diagnostic's table, before and after, against the spec's targets (strict mismatches 15 or fewer; extras disagreement 3 or fewer; stock-voiced cues 0; purity 100% on chars 1, 2, 3, 10 or flagged; char 11 gone; wrong-gender prompts 0), and the former stock cues' cosine against 0.06 before and the clones' 0.29, reported with no target; Fairy Tail purity and cue accuracy per episode, every changed E12/E13 card with its gate or re-pick reason. A target missed is reported as missed, with the cue list, and becomes a ruling in the ledger; nothing is retuned on Sol Bianca or E14.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add docs/results/2026-11-11-identification-part-a.md docs/results/2026-11-11-identification-part-a.jsonl docs/superpowers/ledgers/2026-11-11-identification-workflow.md
git -C $REPO commit -m "docs(results): Part A on Sol Bianca S00E01 and the Fairy Tail names arm"
```

---

### Task 12: Registry schema 5 and CLI

**Files:**
- Modify: `src/fukikae/registry/store.py`, `src/fukikae/registry/cli.py`
- Test: `tests/registry/test_store.py`, `tests/registry/test_cli.py`

**Interfaces (`fukikae.registry.store`):**
- `SCHEMA_VERSION = 5`; `_migrate_4_to_5` adds `gender`, `gender_by`, `kind` (default `'character'`) and `reviewed` to `characters` in one transaction.
- `Character` gains `gender: str | None = None`, `gender_by: str | None = None`, `kind: str = "character"`, `reviewed: str | None = None`.
- `KINDS = {"character", "narrator", "ignored"}`; `GENDER_BY = {"operator", "cast"}`; `OPERATOR_BY` gains `"studio"` (an accepted studio card is curated, CD2).
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
def test_studio_is_an_operator_source(): ...     # assign_card(by="studio") beats auto; auto can't replace it

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

### Task 13: Kinds and operator gender in the pipeline

**Files:**
- Modify: `src/fukikae/models.py` (`ClusterAssignment.status`), `src/fukikae/stages/characters.py`, `src/fukikae/cues.py` (`_who`, skip), `src/fukikae/cards/auto.py` (operator gender re-pick, narrator scene gate), `src/fukikae/stages/tts.py` (`established_gender`, narrator flag skip), `src/fukikae/registry/cli.py` (gender/kind invalidation)
- Test: `tests/stages/test_characters.py`, `tests/test_cues.py`, `tests/cards/test_auto.py`, `tests/stages/test_tts.py`

**Interfaces:**
- `ClusterAssignment.status` gains `"ignored"`, set by the characters stage when the matched character's kind is `ignored`.
- Cues of an ignored cluster: `skip = "ignored"`, `text = None` (never translated, never rendered; the mix keeps the original vocals).
- `GENDER_NOTE = "re-picked: operator gender"`: `ensure_assignments(..., genders: Mapping[int, str])` re-picks an allowed automatic clone card whose references' class is the other one, through `tiered_references` with the operator's class as the register; refused cards are flagged. (No stock card exists to re-pick after Task 6.)
- Narrators: `pick_references(scene_gap=None)`; no A3 flag; never in merge suggestions (Task 16).
- `registry gender` marks tts stale for the character's episodes; `registry kind` marks characters stale for them.

- [ ] **Step 1: Write the failing tests**

```python
def test_ignored_character_clusters_get_status_ignored(tmp_ws): ...
def test_assignments_without_ignored_characters_are_byte_identical(tmp_ws): ...
def test_ignored_cues_are_skipped_and_never_rendered(tmp_ws): ...   # and get no extras clone
def test_operator_gender_repicks_an_auto_clone_with_other_class_references():
    reg.set_character_gender(4, "m"); ensure_assignments(..., genders={4: "m"})
    a = reg.card_for(4, "en")
    assert a.note.startswith(GENDER_NOTE) and refs_class(a.card_id) == "m"
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

### Task 14: Decisions and `fukikae studio apply`

**Files:**
- Create: `src/fukikae/studio/__init__.py`, `src/fukikae/studio/model.py`, `src/fukikae/studio/apply.py`, `src/fukikae/studio/cli.py`
- Modify: `src/fukikae/cli.py` (register `studio`), `src/fukikae/registry/cli.py` (move `_stale`, `_remap_assignments`, `_episodes_with_key` into `fukikae.registry.invalidate` so the studio reuses them; CLI output unchanged)
- Create: `src/fukikae/registry/invalidate.py`
- Test: `tests/studio/test_apply.py`, `tests/registry/test_cli.py` (unchanged output)

**Interfaces (`fukikae.studio.model`, pydantic):**
- `Decision` (discriminated on `op`): `name{character, name}`, `alias{character, name}`, `gender{character, gender: "f"|"m"|None}`, `merge{keep, drop}` (character ids), `split{episode, cluster, by: Literal["pitch","cluster"]}`, `kind{character, kind}`, `lock{character}`, `reviewed{character}`, `accept{character, language, card_id, draft, version, references, overrides}`. Each carries `key` and `external_id` of the character beside the id (sharing later), and `digest`: the sha256 of the character's registry rows it was made against (character, actor, the language's card assignment for `accept`).
- `Decisions(schema_version=1, workspace, decisions: list[Decision])`.

**Interfaces (`fukikae.studio.apply`):**
- `row_digest(reg, character_id, language=None) -> str`.
- `plan_decision(reg, ws, d) -> tuple[list[str], list[Invalidation]]` (lines, `(episode, stage)` pairs), used for the page's "this reruns ..." preview.
- `apply_decision(ws, d, *, dry_run=False, force=False) -> ApplyResult(lines, invalidations)`: digest check (refuse with `StaleDecision` naming the difference unless `force`), then one `Registry.transaction()`; a registry refusal rolls the decision back and raises `DecisionRefused` with the registry's message; then `job.mark_stale` for each invalidation; identity decisions `mark_reviewed` their characters; `accept` assigns `chosen_by="studio"`, `locked=True`, note naming draft, version, references, overrides and audition count. Each applied decision is appended to `<ws>/studio/decisions.jsonl` with time and result lines.
- `apply_file(ws, path, *, dry_run, force) -> list[ApplyResult]`: in order; the first refusal stops the file, earlier decisions stay applied (each is its own transaction), and the error names how many were applied.
- Merge maps characters to their actors and calls `merge_actors`, then `_remap_assignments`; `split by="cluster"` calls `split_cluster`; `split by="pitch"` writes `<ws>/studio/splits/<episode>.json` (Task 15 reads it).
- `fukikae studio apply <ws> <file> [--dry-run] [--force]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_name_gender_kind_lock_each_apply_in_one_transaction(ws): ...
def test_stale_digest_refuses_without_force(ws):
    d = decision_for(ws, "name", character=3, name="Janny")
    reg.rename_character(3, "Someone")     # after the page loaded the row
    with pytest.raises(StaleDecision, match="character 3"):
        apply_decision(ws, d)

def test_registry_refusal_changes_nothing(ws): ...           # merge into a locked actor
def test_merge_remaps_assignments_and_marks_characters_stale(ws): ...
def test_gender_marks_only_tts_stale(ws): ...
def test_pitch_split_writes_the_split_file(ws): ...
def test_accept_assigns_a_locked_studio_card(ws):
    apply_decision(ws, accept(character=1, language="en", card_id=cid, draft="1.en", version=4))
    a = reg.card_for(1, "en")
    assert (a.card_id, a.chosen_by, a.locked) == (cid, "studio", True)

def test_accept_over_a_newer_assignment_is_stale(ws): ...    # a run re-picked meanwhile
def test_every_applied_decision_is_logged(ws): ...
def test_dry_run_writes_nothing(ws): ...
def test_apply_file_stops_at_the_first_refusal(ws): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_apply.py tests/registry/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/__init__.py src/fukikae/studio/model.py src/fukikae/studio/apply.py src/fukikae/studio/cli.py src/fukikae/cli.py src/fukikae/registry/cli.py src/fukikae/registry/invalidate.py tests/studio/test_apply.py tests/registry/test_cli.py
git -C $REPO commit -m "feat(studio): decisions as data, applied through the registry with digests and invalidation"
```

---

### Task 15: Pitch splits through the names stage

**Files:**
- Modify: `src/fukikae/stages/names.py` (inputs, key, relabel), `src/fukikae/names/align.py` (row moves for operator labels), `src/fukikae/names/inputs.py`
- Test: `tests/stages/test_names.py`, `tests/names/test_align.py`

**Interfaces:**
- `operator_splits(ep) -> dict[str, PitchSplit]` from `<ws>/studio/splits/<episode>.json`; `PitchSplit(male_below, female_above)`.
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

### Task 16: The season index

**Files:**
- Create: `src/fukikae/studio/season.py`, `src/fukikae/names/flags.py`
- Modify: `src/fukikae/studio/model.py` (`SeasonDoc` and parts), `src/fukikae/names/cli.py` (move `_stale_decision`, `_disagreements`, `_merge_suggestions`, `_card_lines` into `names/flags.py`, shared by `names show` and the index)
- Test: `tests/studio/test_season.py`, `tests/names/test_cli.py` (unchanged output)

**Interfaces (`fukikae.studio.model`):**
- `Sample(episode, start, end, transcript, clip)`; `GenderInfo(estimate, confidence, median, seconds: dict[str,float], per_episode: dict[str,str], stored, stored_by)`; `CardInfo(language, card_id, source, chosen_by, locked, tier: Tier | None, flags: list[str])`; `Suggestion(kind: Literal["merge","split"], other: int | None, cosine: float | None, episode: str | None, cluster: str | None, parts: dict | None)`; `Flag(kind, text, episode: str | None)`; `CharacterEntry(id, actor_id, name, key, external_id, kind, locked, reviewed, digest, episodes: dict[str, EpisodeSpeech], seconds, clean_seconds, turns, cues, samples, gender, cards: list[CardInfo], suggestions, names: list[str], flags)`; `ExtraEntry(episode, label, seconds, tier, card_id, samples)`; `EpisodeEntry(id, last_stage, ready: bool, command: str | None)`; `SeasonDoc(schema_version=1, workspace, built, inputs_digest, episodes, characters, extras, queue: list[int])`.

**Interfaces (`fukikae.studio.season`):**
- `build_season(ws, cfg, *, clips=6) -> SeasonDoc`: every episode with `05-characters` done is ready; others are listed with the `fukikae run ... --to characters` command. Samples spread over episodes and over each episode's turns (longest clean turn per slice). Gender from `pitch.cached_track` over the season and per episode. Merge suggestions from registry centroids at or over `MERGE_SUGGEST_COS`, narrators excluded; split suggestions from `pitch.bimodal` per (episode, cluster). Flags: thin and below the floor, impure, stale, stock kept, gated lines (from each episode's `07-tts/run.json` when present), dropped hallucinations (`03b-asrclean/dropped.json`), names disagreements, bimodal.
- `queue`: character ids ordered by flag severity (thin, stock kept, impure, bimodal, gated lines, names disagreements, below floor), then unreviewed, then seconds descending.
- `season_path(ws) -> Path` (`<ws>/studio/season.json`); `load_or_build(ws, cfg) -> SeasonDoc`: rebuilt when `inputs_digest` (digests of each ready episode's assignments, turns, transcript, names and tts run files, plus the registry's `data_version`) changes.
- `cut_sample(ws, episode, start, end) -> Path`: `<ws>/studio/audio/<sha>.wav`, 24 kHz mono through `cards.select.cut_clip`, owner-only directory.

- [ ] **Step 1: Write the failing tests**: `test_season_lists_every_character_across_episodes`; `test_episode_without_characters_is_listed_with_its_command`; `test_samples_spread_over_episodes_and_turns`; `test_bimodal_cluster_gets_a_split_suggestion`; `test_merge_suggestion_above_threshold_only_and_never_with_a_narrator`; `test_flags_include_thin_stock_impure_gated_and_dropped`; `test_queue_orders_flags_before_unreviewed_before_seconds`; `test_index_is_rebuilt_only_when_inputs_change`; `test_building_the_index_writes_nothing_to_the_registry`; `test_names_show_output_unchanged_after_the_move`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_season.py tests/names/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/season.py src/fukikae/studio/model.py src/fukikae/names/flags.py src/fukikae/names/cli.py tests/studio/test_season.py tests/names/test_cli.py
git -C $REPO commit -m "feat(studio): season index of characters, samples, suggestions and flags"
```

---

### Task 17: Studio server, page shell and identity panel

**Files:**
- Create: `src/fukikae/studio/server.py`, `src/fukikae/studio/static/index.html`, `src/fukikae/studio/static/app.js`, `src/fukikae/studio/static/vendor/preact.module.js`, `src/fukikae/studio/static/vendor/hooks.module.js`, `src/fukikae/studio/static/vendor/htm.module.js`, `src/fukikae/studio/static/vendor/VERSIONS`
- Modify: `src/fukikae/studio/cli.py` (`fukikae studio <ws>`), `pyproject.toml` (package data for `studio/static`)
- Test: `tests/studio/test_server.py`, `tests/studio/test_static.py`

**Interfaces:**
- `fukikae studio <ws> [--port 8765] [--clips 6] [--no-open]`: builds or loads the season index, starts the server, prints `http://127.0.0.1:<port>/?t=<token>` and `ssh -L <port>:127.0.0.1:<port> <hostname>`; Ctrl-C stops it and unloads any audition worker.
- `StudioServer(ws, cfg, *, port, token)`: `ThreadingHTTPServer` bound to `127.0.0.1` only (refuses any other bind address); every request needs the token (`?t=` on the page load sets an `HttpOnly`, `SameSite=Strict` cookie; API calls send it as `X-Fukikae-Token`), else 403; `Host` must be `127.0.0.1:<port>` or `localhost:<port>` (DNS-rebinding guard); request bodies capped at 1 MB; JSON validated by the pydantic models.
- Endpoints in this task: `GET /` (the page), `GET /static/...` (package files only, no path traversal), `GET /api/season`, `GET /api/characters/<id>`, `GET /audio/<sha>.wav` (only files the index or a draft produced), `POST /api/decisions/preview` (`plan_decision`), `POST /api/decisions` (`apply_decision`; 409 with the difference on `StaleDecision`, 422 on `DecisionRefused`), then the character's entry is rebuilt and returned.
- `static/vendor/VERSIONS`: each vendored file's package, version and sha256; `test_static.py` checks the hashes.
- The page: the queue (home), a character view with samples, gender, cards per language, suggestions, names evidence, flags, and the identity panel (every `Decision` op but `accept`), each action previewed ("reruns translate on E02, E05") before it is posted. No external URL in any static file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/studio/test_server.py
def test_binds_loopback_only(studio): assert studio.server_address[0] == "127.0.0.1"
def test_requests_without_the_token_are_refused(studio): ...        # 403
def test_foreign_host_header_is_refused(studio): ...                # 403
def test_season_endpoint_serves_the_index(studio): ...
def test_decision_post_applies_and_returns_the_rebuilt_character(studio): ...
def test_stale_decision_is_409_with_the_difference(studio): ...
def test_static_paths_cannot_escape_the_package(studio): ...       # /static/../server.py -> 404
def test_body_over_one_megabyte_is_413(studio): ...

# tests/studio/test_static.py
def test_no_external_urls_in_static_files(): ...                   # regex for http(s)://, //cdn, @import url(
def test_vendored_files_match_their_recorded_hashes(): ...
def test_page_has_a_control_for_every_identity_op(): ...
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_server.py tests/studio/test_static.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Vendor Preact and htm from their published npm tarballs at pinned versions (record version, tarball URL and sha256 in `VERSIONS`; the fetch is a one-time step in this task, never at run time). Check the page by hand through `ssh -L` on a fixture workspace and note it in the report.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/server.py src/fukikae/studio/static src/fukikae/studio/cli.py pyproject.toml tests/studio/test_server.py tests/studio/test_static.py
git -C $REPO commit -m "feat(studio): fukikae studio serves a loopback season app with identity editing"
```

---

### Task 18: Reference candidates and drafts

**Files:**
- Create: `src/fukikae/studio/drafts.py`
- Modify: `src/fukikae/studio/model.py` (`Candidate`, `Draft`, `DraftVersion`), `src/fukikae/studio/server.py` (endpoints), `src/fukikae/studio/static/app.js` (voice panel: references and parameters), `src/fukikae/cards/select.py` (`eligible_references` reports every refused candidate with its reasons through the existing `gated` out-parameter, extended to a per-candidate list)
- Test: `tests/studio/test_drafts.py`, `tests/cards/test_select.py`

**Interfaces:**
- `Candidate(episode, start, end, seconds, transcript, snr, median, gender, rank: int | None, verdicts: dict[str, bool], reasons: list[str], clip)`: every turn of the character across ready episodes; `rank` is the CD5 rank among eligible turns, None for a refused turn; `reasons` names every check it failed (overlap, length, song, foreign names, sparse text, register, voicing, song-like, scene).
- `candidates(ws, cfg, character_id) -> list[Candidate]`.
- `DraftVersion(n, backend, references: list[RefSpan], params: dict[str, Any], card_id, overrides: list[RefSpan], created, auditions: list[Audition])`; `Draft(schema_version=1, character_id, language, base: Literal["card","derived","proposed"], versions)`.
- `open_draft(ws, cfg, character_id, language) -> Draft`: version 1 from the current card for the language; else derived from an accepted card in another language (`voices derive`'s rule); else the automatic proposal (`tiered_references`, nothing assigned).
- `draft_card(ws, cfg, draft, references, backend, params) -> str`: builds the clone card exactly as `cards.auto._clone` does (sidecar clips from the reference spans, D32's joins, `origin` per clip), puts it in the store, returns its id. `params` is validated against the backend class's `voice_keys`; an unknown key is refused.
- `floor_meter(references) -> dict` (`clips`, `seconds`, `floor_clips`, `floor_s`, `prompt_s`, `tier`).
- `save_draft(ws, draft)`: `<ws>/studio/drafts/<character>.<language>.json`, owner-only, atomic.
- Endpoints: `GET /api/characters/<id>/candidates`, `GET /api/drafts/<id>/<lang>` (opens one when missing), `POST /api/drafts/<id>/<lang>/versions` (`{references, backend, params}` -> new version with its card id and meter), `GET /api/backends` (installed clone backends and each one's `voice_keys` with type and range).
- The voice panel: candidate list with play buttons and verdicts, a "show gated" toggle, the selected clips in order with remove and reorder, the meter, the backend choice, and one control per voice key from `/api/backends`. A gated pick is recorded in `overrides`.

- [ ] **Step 1: Write the failing tests**: `test_candidates_cover_every_turn_with_reasons_for_refused_ones`; `test_eligible_candidates_keep_cd5_rank`; `test_draft_opens_from_the_current_card`; `test_draft_for_a_new_language_derives_from_an_accepted_card`; `test_draft_without_cards_opens_from_the_proposal_and_assigns_nothing`; `test_draft_card_matches_the_automatic_clone_for_the_same_references` (same card id as `cards.auto._clone`); `test_unknown_voice_key_is_refused`; `test_gated_pick_is_recorded_as_an_override`; `test_floor_meter_reports_the_tier`; `test_backends_endpoint_lists_voice_keys`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_drafts.py tests/cards/test_select.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/drafts.py src/fukikae/studio/model.py src/fukikae/studio/server.py src/fukikae/studio/static/app.js src/fukikae/cards/select.py tests/studio/test_drafts.py tests/cards/test_select.py
git -C $REPO commit -m "feat(studio): reference candidates with their gate verdicts, and voice drafts"
```

---

### Task 19: The audition worker

**Files:**
- Create: `src/fukikae/studio/audition.py`
- Modify: `src/fukikae/stages/tts.py` (expose `render_line(card_id, text, delivery, cfg, store, cache, synth) -> RenderedLine` built from `prepare_cue`, `delivery_passed`, `render_guarded` and the pause cap, used by the stage and by auditions; stage output unchanged), `src/fukikae/studio/server.py`, `src/fukikae/studio/static/app.js` (audition list and GPU status)
- Test: `tests/studio/test_audition.py`, `tests/stages/test_tts.py`

**Interfaces:**
- `AuditionLine` (discriminated on `kind`): `typed{text, emotion_from: CueRef | None}`, `real{episode, cue_id}`.
- `AuditionWorker(ws, cfg)`: one thread, a FIFO queue of `(draft version, line)` jobs. On the first job it loads the version's backend through `loader` and `WorkerSynthesizer` under `gpu_lock(<device>)` (the backend's device per CD14); between jobs it keeps the model; after `AUDITION_IDLE_S` with an empty queue, or on `release()`, it unloads and the lock goes with the worker process. A job for another backend unloads the current one first.
- `status() -> GpuStatus(loaded_backend, device, holder: str | None, queued: int)`: `holder` names the lock file's pid and job while another job holds the card (the page shows "GPU 0 held by fukikae run (pid 4121)").
- Real lines: the cue from `06-translate/cues.<language>.json`, its `delivery.reference` as `emotion_audio` under the card's `emotion_strength` and A3's gate, rendered through `render_line` into the workspace's tts cache under the stage's own `cue_key`. Typed lines: no `emotion_audio` unless `emotion_from` names a cue, whose source span is then passed under the same gate. Each audition records `cue_key`, seconds, `suspect` and `passed` on the draft version.
- Endpoints: `POST /api/auditions` (`{character, language, version, line}` -> job id), `GET /api/auditions/<job>` (queued, running, done with wav URL and the source clip for real lines, failed with the message), `GET /api/gpu`, `POST /api/gpu/release`.

- [ ] **Step 1: Write the failing tests**

```python
def test_real_line_audition_uses_the_stage_cue_key(studio_ws):
    job = worker.submit(version, real("E01", "c0042")); worker.wait(job)
    assert audition(job).cue_key == stage_cue_key(studio_ws, "E01", "c0042", version.card_id)

def test_audition_render_is_reused_by_the_next_tts_run(studio_ws): ...   # cache hit, zero renders
def test_wrong_gender_source_is_never_an_emotion_prompt_in_auditions(studio_ws): ...
def test_typed_line_passes_no_emotion_audio_by_default(studio_ws): ...
def test_worker_takes_the_gpu_lock_while_loaded_and_releases_on_idle(studio_ws, fake_clock): ...
def test_release_unloads_and_frees_the_lock(studio_ws): ...
def test_busy_gpu_is_reported_with_its_holder(studio_ws): ...            # lock held by another process
def test_jobs_run_one_at_a_time_in_order(studio_ws): ...
def test_switching_backend_unloads_the_previous_one(studio_ws): ...

# tests/stages/test_tts.py
def test_stage_output_unchanged_after_render_line_refactor(tmp_ws): ...  # plan.json byte-identical
```

- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_audition.py tests/stages/test_tts.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then one manual check on GPU 0 (IndexTTS2) on a hard-link copy of a Fairy Tail workspace, with no other GPU job queued: three auditions of one character, then `fukikae run --from tts` on that episode reports cache hits for those lines; note it in the report.
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/audition.py src/fukikae/stages/tts.py src/fukikae/studio/server.py src/fukikae/studio/static/app.js tests/studio/test_audition.py tests/stages/test_tts.py
git -C $REPO commit -m "feat(studio): auditions through the tts render path under the GPU lock"
```

---

### Task 20: Compare, accept, and other languages

**Files:**
- Create: `src/fukikae/abpair.py` (the blind pair and key logic moved out of `eval/ab.py`; `eval/ab.py` imports it, output unchanged)
- Modify: `src/fukikae/eval/ab.py`, `src/fukikae/studio/drafts.py`, `src/fukikae/studio/server.py`, `src/fukikae/studio/static/app.js`
- Test: `tests/studio/test_accept.py`, `tests/eval/test_ab.py` (unchanged output)

**Interfaces:**
- `abpair.blind_pair(a, b, seed) -> tuple[Pair, Key]`, shared by `eval tts-ab` and the studio.
- `GET /api/drafts/<id>/<lang>/compare?a=<n>&b=<n>&line=<...>[&blind=1]`: the two versions' auditions of the same line (submitting the missing one), blind with a key revealed by `POST .../reveal`.
- `POST /api/drafts/<id>/<lang>/accept` (`{version}`): builds an `accept` decision (digest of the language's current assignment taken when the draft version was made) and applies it through `apply_decision`; the response names the episodes whose tts goes stale and which of their lines are already cached.
- The voice panel: version list with a play-on-line button per version, A/B side by side or blind, and Accept on a version; after accept the character's card in the queue shows `studio, locked`.
- A second language: `open_draft` for a language with no card starts from the accepted card of another language (Task 18); the panel says which.

- [ ] **Step 1: Write the failing tests**: `test_accept_writes_a_curated_locked_card_through_apply`; `test_accept_after_a_run_repicked_the_card_is_409`; `test_accepted_card_survives_the_next_tts_run` (no automatic re-pick: CD2); `test_later_episode_uses_the_accepted_card` (a second episode matching the character renders with it); `test_blind_compare_hides_the_versions_until_reveal`; `test_second_language_draft_starts_from_the_accepted_card`; `test_eval_ab_output_unchanged_after_the_move`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_accept.py tests/eval/test_ab.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/abpair.py src/fukikae/eval/ab.py src/fukikae/studio/drafts.py src/fukikae/studio/server.py src/fukikae/studio/static/app.js tests/studio/test_accept.py tests/eval/test_ab.py
git -C $REPO commit -m "feat(studio): compare versions, accept a curated card, start other languages from it"
```

---

### Task 21: The `[review]` gates in `fukikae run`

**Files:**
- Create: `src/fukikae/studio/gate.py`
- Modify: `src/fukikae/config.py` (`ReviewSettings`, `review_settings`, `Config.review`), `src/fukikae/job.py` (gate hook), `src/fukikae/cli.py` (exit code 3)
- Test: `tests/studio/test_gate.py`, `tests/test_config.py`, `tests/test_job.py`

**Interfaces:**
- `@dataclass(frozen=True) class ReviewSettings: identity: bool = False; voices: bool = False; min_seconds: float = 20.0`; `review_settings(table)` rejects unknown keys (the first draft's `required` included, with a message naming `identity` and `voices`); merged profile then workspace; `[review]` exempt from R3. No profile turns either gate on (open decision 2).
- `main_characters(ep, reg, cfg) -> list[int]`: kind `character` or `narrator`, `min_seconds` or more in the episode.
- `needs_identity(ep, cfg, reg) -> list[str]`: a main character with `reviewed` NULL, or a bimodal split suggestion, impure card or names disagreement newer than its `reviewed`.
- `needs_voices(ep, cfg, reg) -> list[str]`: a main character without a curated, imported or locked card for `cfg.target_language` (`replace_refusal(a, "auto") is not None`).
- `job.run_stages(..., gates: Mapping[str, Callable[[Episode, Config], list[str]]] = {})`: before running a stage named in `gates`, a non-empty result raises `ReviewRequired(episode, stage, reasons)`; `run_episodes` stops only that episode.
- `fukikae run` registers `needs_identity` before `translate` and `needs_voices` before `tts` when enabled; exits 3 printing each reason and `fukikae studio <ws>`.

- [ ] **Step 1: Write the failing tests**: `test_identity_gate_stops_before_translate`; `test_voices_gate_stops_before_tts`; `test_reviewed_character_passes_on_the_next_episode` (season carry); `test_accepted_card_passes_the_voices_gate_on_the_next_episode`; `test_short_and_ignored_characters_never_hold_an_episode`; `test_new_flag_after_review_reopens_the_identity_gate`; `test_multi_episode_run_continues_other_episodes`; `test_gates_off_never_stop`; `test_old_required_key_is_refused_with_a_pointer`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_gate.py tests/test_config.py tests/test_job.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/gate.py src/fukikae/config.py src/fukikae/job.py src/fukikae/cli.py tests/studio/test_gate.py tests/test_config.py tests/test_job.py
git -C $REPO commit -m "feat(run): opt-in identity and voices gates for the studio"
```

---

### Task 22: Cast priors from `cast.toml` (decision 10)

**Files:**
- Create: `src/fukikae/studio/cast.py`
- Modify: `src/fukikae/studio/season.py`, `src/fukikae/studio/model.py`, `src/fukikae/studio/static/app.js`
- Test: `tests/studio/test_cast.py`

**Interfaces:**
- `CastEntry(name, gender: Literal["f","m"] | None, external_id: str | None, aliases: list[str])`; `read_cast(ws) -> list[CastEntry]` from `<ws>/cast.toml` (`[[cast]]`), absent file gives `[]`.
- `SeasonDoc` gains `cast: list[CastEntry]`; the identity panel offers cast names in the name field, the queue header shows "cast lists N speaking roles; this season has M characters", and a cast gender is a one-click `gender` decision. Nothing is applied without the click; an applied cast gender is `gender_by = "operator"`.

- [ ] **Step 1: Write the failing tests**: `test_read_cast_validates_genders`; `test_missing_cast_file_is_empty`; `test_season_doc_carries_the_cast`; `test_cast_gender_is_never_applied_without_a_decision`.
- [ ] **Step 2: Run to verify they fail.** `uv run --directory $REPO pytest tests/studio/test_cast.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/studio/cast.py src/fukikae/studio/season.py src/fukikae/studio/model.py src/fukikae/studio/static/app.js tests/studio/test_cast.py
git -C $REPO commit -m "feat(studio): name and gender suggestions from a hand-written cast.toml"
```

---

### Task 23: Docs

**Files:**
- Modify: `CLAUDE.md`, `README.md`, the execution ledger

- [ ] **Step 1:** CLAUDE.md gets one-line rules citing D40 and D41: pitch is `fukikae.pitch` only (median bands, 60 to 800 Hz); no stock voice on any pipeline path, every speaker cloned through the three reference tiers, a missing clone backend stops tts; the reference gates, `impure_cards`, stock and operator gender join the automatic re-pick reasons; no wrong-gender `emotion_audio` on any path; `asrclean` and `transcript_path`; `studio/` files, `studio.apply` as the only writer, `chosen_by = "studio"` cards are curated and locked, the audition worker holds the GPU lock only while loaded; `[review]` exempt from R3; gender is voice register.
- [ ] **Step 2:** README: the season workflow (`fukikae run --to characters`, `fukikae studio`, identity, `--to translate`, voices, the rest), the ssh tunnel line, `[review] identity` and `voices`, and the D41 note that stock voices are gone.
- [ ] **Step 3:** `uv run --directory $REPO pytest -q`, `uv run --directory $REPO ruff check`, `uv run --directory $REPO ruff format --check`, `uv run --directory $REPO pyright`. Expected: all clean.
- [ ] **Step 4: Commit**

```bash
git -C $REPO add CLAUDE.md README.md docs/superpowers/ledgers/2026-11-11-identification-workflow.md
git -C $REPO commit -m "docs: no stock voices, the identification fixes and the season studio"
```
