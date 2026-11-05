# Speaker Identity Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and improve cross-episode speaker identity against subtitle ground truth, and ship tested default profiles (anime, live-action, general, plus any language variant that earns its place).

**Architecture:** An offline identity harness reads existing pipeline artifacts plus ground truth (ASS speaker names, SDH speaker prefixes, colour-coded subtitles), computes segment embeddings once per (segmentation, embedder) into a cache, and simulates the registry's episode-by-episode matching in memory, so embedder, segmentation and matcher experiments are cheap and separable. Winners move into the pipeline as backends and config profiles.

**Tech Stack:** Python 3.12, uv, existing fukikae packages; new optional extras per model family (e.g. `wespeaker`/3D-Speaker models, `transformers` WavLM speaker models, `audio-separator` dialogue models), scikit-learn (LDA), numpy/scipy.

**Spec:** `docs/superpowers/specs/2026-11-05-speaker-identity-design.md` (results it starts from: `docs/results/2026-11-05-registry-fairytail.md`; foundation ledger: `docs/superpowers/ledgers/2026-11-04-foundation-and-registry.md`).

## Global Constraints

- The bar, per profile, on its held-out episode: a recurring character has ≥ 20 s of labeled speech in each episode compared; pass when ≥ 90% of recurring characters keep ≥ 80% of their speech on one character id and no two recurring characters share a modal id. Report the strict per-clip metric too.
- Spans labeled `?` and overlapped speech are excluded from every metric; every run reports coverage.
- Nothing is chosen by looking at a held-out episode. Fairy Tail: tune E12→E13, hold out E14 (resolved against the E12-E13 registry); adaptation trains on E12-E13 only.
- Each audio language is its own workspace and registry.
- Profiles resolve: workspace explicit > per-language override > profile > built-in default.
- Every model is a backend behind an entry point in its own optional extra; `uv sync --all-extras` must stay resolvable (transformers < 5, huggingface-hub < 1.0 today).
- No model library imported at module load in core code; GPU work on nvidia-smi index 0 (index 1 is vLLM); media never committed; read-only on source media; argument-list subprocess calls only.
- TDD with RED evidence; Conventional Commits; no AI attribution.
- Every keep/drop decision after a comparison is a ledger ruling with its numbers.

## Review Focus

1. **Subtitle timing that doesn't match the audio** (a different release, an intro offset): ground truth silently misaligned. Expect the harness to measure label/speech alignment (share of labeled lines overlapping detected speech, best global offset within ±10 s) and refuse to score below 0.6 alignment unless `--force`. Pinned in Task 2.
2. **No recurring characters** (episodes share no character with ≥ 20 s each): expect verdict `insufficient`, never a division by zero or a misleading pass. Pinned in Task 3.
3. **Two audio tracks in the same language** (DAN DA DAN's es-ES and es-419): expect `extract.audio_track` or a title match (`extract.audio_title = "Castilian"`) to pick the intended track, and the choice recorded in the manifest key. Pinned in Task 1.
4. **Stale embedding cache** after a re-separation or a changed embedder option: expect a cache miss, never reuse. Pinned in Task 3.
5. **Profile and override precedence**: a workspace that sets a value explicitly must win over its profile and over per-language overrides. Pinned in Task 1.

---

## File Structure

```
src/fukikae/
  config.py                      # modify: profile layer in resolve()
  profiles/                      # new: built-in presets (default.toml, anime.toml, live-action.toml, + language variants)
  media.py                       # modify: audio title match in choose_audio_track
  eval/groundtruth/              # new package
    base.py                      # GroundTruthLine, alignment check
    ass_names.py                 # moved from eval/autolabel.py (keeps auto-label working)
    sdh_prefix.py                # SDH/CC speaker prefixes (Latin, Cyrillic, Hangul, CJK)
    colour.py                    # colour-coded WebVTT / EBU-TT classes
  eval/identity/                 # new package
    segments.py                  # segmentation variants -> list[Segment(ep, start, end, label)]
    cache.py                     # embedding cache keyed by (segmentation, embedder, audio hash)
    simulate.py                  # in-memory episode-order registry simulation
    metrics.py                   # majority (bar), strict, purity, coverage, verdict
    matchers.py                  # centroid, knn, lda-adapted, joint-cluster
    cli.py                       # `fukikae eval identity`
  backends/embed/<new>.py        # new embedders
  backends/separation/<new>.py   # dialogue separation
  stages/characters.py           # modify: matcher option, adaptation projection
  registry/store.py              # modify: per-series projection storage
docs/results/                    # one results note per comparison
```

---

### Task 1: Profiles, audio-track selection by title

**Files:** Modify `src/fukikae/config.py`, `src/fukikae/media.py`, `src/fukikae/stages/extract.py`; Create `src/fukikae/profiles/default.toml`, `anime.toml`, `live-action.toml`, `src/fukikae/profiles/__init__.py`; Test `tests/test_profiles.py`, `tests/test_media.py`.

**Interfaces:**
- Produces: `Config.profile: str` (default `"default"`); `load_profile(name: str) -> dict` reading package data (`importlib.resources`); `Config.resolve(kind, language=None, pair=None)` now layers profile table under the per-language/pair overrides and the workspace's explicit tables (order in Global Constraints). Unknown profile name raises `ConfigError` listing available ones. Profiles may set any top-level table (`[embed]`, `[diarize]`, `[registry]`, `[identity]`...); `Config.registry` values also take the profile's `[registry]` when the workspace doesn't set them.
- Produces: `choose_audio_track(probe, source_language, override, title: str | None = None)`: `override` > first track whose language matches and whose title contains `title` (case-insensitive) > previous rules. `ExtractStage.config_key` includes `audio_title`.

- [ ] **Step 1: Tests first.** `test_profile_layers_under_language_and_workspace`: profile `anime` sets `[embed] backend="x"`, `[embed.lang.ko] backend="y"` in the workspace; resolve("embed","ja") == "x", resolve("embed","ko") == "y"; a workspace `[embed] backend="z"` beats the profile for ja (Review Focus 5). `test_unknown_profile_lists_available`. `test_registry_threshold_from_profile_unless_workspace_sets_it`. `test_choose_track_by_title`: two `spa` tracks titled "Spanish (Latin America)" and "Spanish (Castilian)"; `title="castilian"` picks the second (Review Focus 3).
- [ ] **Step 2:** Run, capture RED.
- [ ] **Step 3:** Implement; `default.toml` mirrors today's built-in defaults exactly (no behavior change for a workspace without `profile`); `anime.toml` and `live-action.toml` start as copies of default with a comment saying Task 9 fills them.
- [ ] **Step 4:** Full suite GREEN.
- [ ] **Step 5:** Commit `feat(config): profiles and audio track selection by title`.

### Task 2: Ground-truth readers

**Files:** Create `src/fukikae/eval/groundtruth/{__init__,base,ass_names,sdh_prefix,colour}.py`; Modify `src/fukikae/eval/autolabel.py` (becomes a thin caller of `ass_names`, behavior unchanged), `src/fukikae/eval/cli.py` (`auto-label --source ass|sdh|colour`); Test `tests/eval/test_groundtruth.py`.

**Interfaces:**
- Produces: `GroundTruthLine(start: float, end: float, speaker: str)` (speaker already canonicalized and aliased); `read_ground_truth(path: Path, kind: Literal["ass","sdh","colour"], aliases: dict[str,str]) -> list[GroundTruthLine]`; `alignment(lines, speech_turns: list[Turn]) -> AlignmentReport(overlap_share: float, best_offset_s: float, share_at_offset: float)`.
- `sdh_prefix`: a line starting with `NAME:`, `(NAME)`, `[NAME]`, `- NAME:` after stripping ASS/HTML tags; NAME is 1-4 words of letters in any script (Latin, Cyrillic, Hangul, CJK, with spaces, dots or hyphens), ≤ 24 characters; reject a sound-description list (configurable, defaults include music/музыка/음악/音乐/laughs/смеётся/웃음/笑) and anything in lowercase Latin (`(sighs)`). A tagged speaker carries over to following lines of the same cue until a dash starts a new speaker.
- `colour`: WebVTT `<c.vtt_yellow>`/`<c.yellow>` classes and EBU-TT styles (`textYellow`, `tts:color`); speaker label = `"<colour>@<episode>"` unless a name tag co-occurs with that colour in ≥ 3 lines in the episode, then that name; white/default is never a speaker class.
- The alignment check (Review Focus 1): `eval auto-label` and `eval identity` refuse when `share_at_offset < 0.6` without `--force`, and apply `best_offset_s` when it improves the share by ≥ 0.1, reporting it.

- [ ] **Step 1: Tests first.** ASS behavior unchanged (existing tests stay green). SDH: `"КОСТЯ: Привет"` → Костя; `"(새하) 뭐야"` → 새하; `"[music]"` and `"(sighs)"` → none; a dash-split cue `"- ARISU: Go!\n- KUINA: Wait."` → two lines with the cue's time split proportionally to text length. Colour: a VTT sample with yellow/cyan cues and `(LOLA)` on three yellow cues → yellow lines labeled LOLA, cyan lines `cyan@S04E01`. Alignment: lines shifted by +3 s against turns → `best_offset_s ≈ 3`, refusal below 0.6.
- [ ] **Step 2–4:** RED, implement, GREEN; check the readers on the library samples in `<scratch>/lib-probe/subs/` (Cyberfarm, Grid, Parallels) and the RTVE VTTs in `<scratch>/liveaction-survey/rtve/`; record counts in the task report.
- [ ] **Step 5:** Commit `feat(eval): sdh-prefix and colour ground-truth readers`.

### Task 3: Identity harness

**Files:** Create `src/fukikae/eval/identity/{__init__,segments,cache,simulate,metrics,matchers,cli}.py`; Modify `src/fukikae/cli.py`; Test `tests/eval/identity/`.

**Interfaces:**
- `Segment(episode: str, start: float, end: float, label: str | None)`.
- `segments(ws, episode, kind: Literal["diarization","subtitles"], gt: list[GroundTruthLine]) -> list[Segment]`: `diarization` = clean spans from `04-diarize` (labels from ground-truth overlap ≥ 0.6, else `None`); `subtitles` = ground-truth lines themselves (overlaps between lines removed), min length 1.0 s.
- `EmbeddingCache(root: Path)`: `get_or_compute(episode, segs, embedder_cfg: BackendConfig, audio: Path) -> np.ndarray`; key = sha256 of (segment list, embedder name + options, audio content hash). Lives in `<workspace>/.fukikae/cache/embeddings/`.
- `Matcher` protocol: `fit_episode(registry_state, episode, embs, segs) -> list[int]` returning a character id per segment, mutating an in-memory registry state; built-ins in `matchers.py`: `centroid` (today's logic: cluster-level mean, threshold, min_new_seconds; operates on the diarization clusters or, for subtitle segmentation, on per-segment clustering with agglomerative cosine at the same threshold), more in Task 6.
- `simulate(order: list[str], matcher, ...) -> dict[(episode, seg_idx), char_id]`: episodes in order, each resolved against the state built from the earlier ones.
- `metrics.score(assign, segs, min_speech=20.0, keep=0.8, share=0.9) -> IdentityScore(recurring, passing, fraction, merges, strict_fraction, purity_by_id, coverage, verdict)`; verdict `pass` / `fail` / `insufficient` (no recurring characters, Review Focus 2).
- CLI: `fukikae eval identity <ws> --tune S01E12,S01E13 [--holdout S01E14] --gt ass|sdh|colour --segmentation ... --embedder <backend> [--embedder-option k=v] --matcher ... [--threshold T] [--json OUT]`. Prints a table per episode and the verdict; `--holdout` scores the held-out episode against the state built from `--tune` and is reported separately.

- [ ] **Step 1: Tests first.** Metrics on synthetic assignments: the results doc's Fairy Tail numbers reproduce (build fixture assignments from a small table: Laxus 89% one id → passing; Touka 15% → failing); `insufficient` when nothing recurs; merges counted when two recurring characters share a modal id. Cache: same inputs hit; changed audio bytes or embedder option miss (Review Focus 4). Simulation: three fake episodes with fixed embeddings, centroid matcher reproduces `registry.matching.match` decisions on the same data.
- [ ] **Step 2–4:** RED, implement, GREEN. Then run the baseline on Fairy Tail (ECAPA, diarization segmentation, centroid, threshold 0.6, tune E12→E13 only; no holdout yet) and confirm it reproduces the results doc's 70% speech-weighted / 0.17 strict within rounding.
- [ ] **Step 5:** Commit `feat(eval): identity harness with cached embeddings and offline matching`.

### Task 4: Test-set workspaces and pipeline runs

**Files:** Create `docs/results/2026-11-xx-identity-testsets.md`; workspaces outside the repo.

- [ ] **Step 1:** For each set the copy agent verified (Fairy Tail ja and en-dub; The Cyberfarm S02E01-E04 ru; Grid pair ko if found; DAN DA DAN es-ES and es-419; Link Click zh; the Russian-dub series ru), create `~/fukikae-ws/<set>/fukikae.toml` with `source`, `source_language`, `extract.audio_track`/`audio_title` where needed, ASR language default, `gpu_device = 0`, and a `labels-aliases.yaml` from the auto-label `--report` suggestions you accept (note each).
- [ ] **Step 2:** Run `fukikae run <ws> <ep> --to diarize` for every episode (background; one GPU job at a time).
- [ ] **Step 3:** `fukikae eval auto-label <ws> --source <kind> --report` per set; record per-set coverage, alignment and recurring-character counts in the test-sets doc; a set with no recurring characters or alignment < 0.6 is dropped and noted.
- [ ] **Step 4:** Commit the doc `docs: identity test sets`.

### Task 5: Embedder backends and comparison

**Files:** Create `src/fukikae/backends/embed/<name>.py` for at least three new embedders, chosen from CAM++ / ERes2NetV2 (3D-Speaker or wespeaker packaging), a WavLM-based speaker-verification model (`transformers`), pyannote's WeSpeaker ResNet34, TitaNet-L (NeMo) — pick by installability alongside the existing extras; plus a `fusion` embedder that concatenates two configured embedders' normalized outputs. Modify `pyproject.toml` (one extra per family), `src/fukikae/backends/loader.py` (EXTRAS). Test `tests/backends/test_embedders.py`. Create `docs/results/2026-11-xx-embedders.md`.

**Interfaces:** each satisfies `Embedder.embed(audio, spans) -> (n, dim)` L2-normalized, `dim`, `vram_mb` measured.

- [ ] **Step 1: Tests first** (fakes for shape/normalization; `@gpu` test per backend on the Fairy Tail clip: shape, unit norm, same-speaker pairs score higher than cross-speaker pairs on 10 labeled pairs from E12).
- [ ] **Step 2–4:** RED, implement, GREEN.
- [ ] **Step 5:** Comparison on ground-truth segments (`--segmentation subtitles`, centroid matcher) for every tuning set from Task 4: per embedder, majority fraction, strict fraction, merges, nearest-centroid accuracy; a table per genre. Write the results note; ledger ruling: which embedder(s) go forward per genre.
- [ ] **Step 6:** Commit.

### Task 6: Segmentation and matching variants

**Files:** Modify `src/fukikae/backends/diarize/pyannote.py` (expose `clustering_threshold`, `min_cluster_size`, `exclude_overlap` options), `src/fukikae/eval/identity/segments.py` (add `asr`: segments from `03-asr` sentences, split where consecutive word-window embeddings differ by cosine distance > a threshold), `src/fukikae/eval/identity/matchers.py` (add `knn` (k nearest exemplars, vote), `multi-centroid` (up to 3 k-means centroids per actor), `lda` (scikit-learn LDA fit on tuning episodes' labeled segments, applied before centroid matching), `joint` (agglomerative clustering over all episodes' segments at once with cannot-link for overlapping segments in one episode)). Test `tests/eval/identity/test_matchers.py`, `test_segments.py`. Create `docs/results/2026-11-xx-segmentation-matching.md`.

- [ ] **Step 1: Tests first** on synthetic embeddings: two close clusters plus labels → `lda` separates them where `centroid` merges; `knn` keeps a bimodal speaker on one id; `joint` never links two overlapping segments.
- [ ] **Step 2–4:** RED, implement, GREEN.
- [ ] **Step 5:** With the Task 5 winners, compare segmentation (pyannote default, pyannote tuned grid over 3×2 options, subtitles, asr) and matchers on the tuning sets; results note; ledger rulings per genre.
- [ ] **Step 6:** Commit.

### Task 7: Pipeline integration of the winners

**Files:** Modify `src/fukikae/stages/characters.py` (option `matcher` selecting the Task 6 implementations through a small registry; `lda` reads a projection), `src/fukikae/registry/store.py` (table `projections(series TEXT, kind TEXT, matrix BLOB, mean BLOB, created TEXT)`; `save_projection`, `load_projection`), `src/fukikae/registry/cli.py` (`registry adapt <ws> --episodes ... --gt ass|sdh|colour` fits and stores the projection from labeled episodes). Test `tests/registry/test_adapt.py`, `tests/stages/test_characters.py`.

- [ ] **Step 1: Tests first:** a stored projection is applied by the stage when `matcher = "lda"` and absent-projection falls back to `centroid` with a logged warning; `registry adapt` refuses episodes outside the workspace and writes one projection per series (replacing the previous one).
- [ ] **Step 2–4:** RED, implement, GREEN; pinned/resumable behavior of the characters stage unchanged (existing tests).
- [ ] **Step 5:** Commit.

### Task 8: Dialogue separation comparison

**Files:** Create `src/fukikae/backends/separation/dialogue.py` (a dialogue/music/effects model via `audio-separator`, e.g. a Mel-Band Roformer dialogue checkpoint; option `model`), extra `dialogue`; Test `tests/backends/test_dialogue_sep.py`; Create `docs/results/2026-11-xx-separation.md`.

- [ ] **Step 1–4:** Tests first (fake + `@gpu` shape/energy test as in foundation Task 8), RED, implement, GREEN.
- [ ] **Step 5:** Re-run separate→diarize for the tuning sets in copies of their workspaces with `[separation] backend="dialogue"`, re-score with the current best embedder/segmentation/matcher; results note; ruling.
- [ ] **Step 6:** Commit.

### Task 9: Threshold sweep, profiles, held-out confirmation

**Files:** Modify `src/fukikae/profiles/{anime,live-action,default}.toml` (+ language variants that earn it); Create `docs/results/2026-11-xx-identity-profiles.md`.

- [ ] **Step 1:** Per genre, sweep the match threshold (and `knn` k / LDA dims if chosen) on the tuning episodes only; pick per genre.
- [ ] **Step 2:** Write the profiles; a language variant only if it beats its genre profile on that language's tuning set by ≥ 5 points of majority fraction.
- [ ] **Step 3:** Held-out confirmation per profile, once: Fairy Tail E14, the live-action set's held-out episode, each dub set's last episode. Report bar verdicts. A profile that fails its held-out episode ships anyway as the best measured, with its failure stated in the results and a ledger ruling (no re-tuning on held-out data).
- [ ] **Step 4:** Make `profile` selection documented in README and CLAUDE.md; commit `feat(profiles): tested anime, live-action and default identity profiles`.
