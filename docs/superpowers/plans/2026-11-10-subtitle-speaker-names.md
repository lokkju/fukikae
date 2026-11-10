# Subtitle Speaker Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When an episode's subtitles name speakers (Crunchyroll ASS), attribute diarize turns to those names, split merged clusters into per-name sub-clusters before the characters stage, map names onto registry characters without overriding operator state, pick clone references from name-confirmed turns only, and measure the gain on Fairy Tail E11 to E14 with a held-out protocol (operator ruling D27).

**Architecture:** A new CPU-only `names` stage between `diarize` and `characters` reads the subtitle track CD25 already chooses (or another language's named track, or an operator file), decides whether its Name column is usable, aligns named lines to diarize turns, cuts the few turns that hold two names, relabels turns into `<cluster>~<slug>` sub-clusters and moves the existing `clusters.npz` rows accordingly (no re-embedding), writing `04c-names/{names.json,turns.json,clusters.npz}`. The characters stage, `registry resolve`, `registry reembed`, `registry adapt` and `eval identity` read identity inputs through one helper that returns the refined files when names were used. Registry schema 4 adds `characters.name_by` and a `character_aliases` table; the characters stage maps each key to a character (known key, rename an unnamed uncurated character, create, or extra) under lock and operator-name protections. Reference selection and the subtitle path read the name decisions from `assignments.json`. A new `eval names` command and the existing identity harness score two held-out protocols and write ledger rows.

**Tech Stack:** Python 3.12, uv, typer, pydantic, numpy, sqlite3; existing `fukikae.subtitles`, `fukikae.media`, registry, cards and eval modules. No new dependencies, no GPU in any task except the optional en-track diarize run in Task 11.

**Spec:** `docs/superpowers/specs/2026-11-10-subtitle-speaker-names-design.md` (draft; this plan follows it and stops for nothing but the operator's open decisions listed at its end). Rulings R1 to R20 (`docs/superpowers/ledgers/2026-11-05-speaker-identity.md`), GT1 to GT15 (`docs/superpowers/ledgers/2026-11-08-ground-truth-rulings.md`) and CD1 to CD26 (`docs/superpowers/ledgers/2026-11-08-character-cards-rulings.md`) stand. Execution ledger: `docs/superpowers/ledgers/2026-11-10-subtitle-speaker-names.md` (created by the controller at Task 1's dispatch).

## Global Constraints

- `$REPO` is the checkout the plan runs in (a worktree under `~/projects/fukikae/.claude/worktrees/`). uv only, always `uv run --directory $REPO ...`; never pip, never `cd <dir> && <cmd>`; absolute paths everywhere. Pipeline and media commands through `mise -C $REPO exec -- <cmd>`.
- Tests first, with RED evidence in each task report. Markers as in CLAUDE.md; default runs stay offline, model-free and GPU-free. Test fixtures are synthetic ASS text written in the test; no release subtitle file is committed.
- Real-data runs (Task 11) work on hard-link copies (`cp -al`) under `~/fukikae-ws/t-names/`; source media and the existing workspaces are read-only. GPU 0 only if the operator approves the en E11 run (open decision 7); index 1 is vLLM.
- Commits: stage explicit paths with `git -C $REPO add <paths>`, then a plain `git -C $REPO commit -m "<conventional message>"` (no pathspec). No AI attribution, no `Co-Authored-By`, no session links. One committer at a time in a checkout (pre-commit stashes other agents' unstaged edits).
- R1: pipeline code never imports `fukikae.eval`. Name canonicalisation and the alignment check move into `fukikae.names`; `eval/groundtruth/base.py` re-exports them so eval's imports keep working.
- R3: `[names]` is a settings table without `backend`; the config check exempts it as it does `[resolve]`. Profiles don't set `[names]`.
- CD2, CD3, CD9: no names-driven action replaces a locked, operator-chosen, imported or restock-protected card; only `auto`, `auto-fallback` and `restock` cards are re-picked.
- GT2: ledger rows add fields, never change them; `stage = "names"` is a new value, not a schema bump.
- Every threshold (0.5 coverage, 0.6 turn share, 0.5 s cut side, 0.3 s minimum piece, 5 s new character, 90% containment, 80% dominance, 0.2 s unvoiced) is a module constant named in the spec. None is tuned on E14; a defect seen there becomes a ruling in the execution ledger.
- Subagents never start background watchers; a long run is one foreground command or one background command that reports on exit.

## Review Focus

1. **Operator state beats names.** A sub-cluster under a locked actor stays with it; an operator's name is never replaced; an operator alias is never re-pointed by subtitles; curated and imported cards are never re-picked. Pinned in Tasks 5, 6 and 9.
2. **A Name column that isn't speakers** (one value everywhere, style names, a track that doesn't line up with the speech) never drives identity under `use = "auto"`. Pinned in Task 2.
3. **No grading on the run's own labels.** `holdout_episodes` and hidden lines are invisible to every pipeline consumer (characters, resolve, translate's cue attribution, reference picking), and the eval reports the ceiling arm only under its label. Pinned in Tasks 4, 8 and 10.
4. **Existing workspaces.** Without usable names, a rerun reproduces `assignments.json` byte for byte (so nothing downstream reruns), and schema 3 registries migrate in one transaction. Pinned in Tasks 5 and 6.
5. **Sub-cluster labels everywhere a cluster label goes**: pinning, `registry split`, resolve, reembed, adapt, the identity harness. Pinned in Task 7.

---

## Decisions this plan makes

- **Stage name and directory:** `names`, `04c-names/` (the work dir already has `04b-tag`, which runs after characters; numbering is by family, not order). Pipeline order: extract, separate, language, asr, diarize, names, characters, [tag], translate, tts, timefit, mix.
- **Package:** `src/fukikae/names/` holds `normalize.py` (keys, classes, display), `detect.py` (track choice and verdict), `align.py` (offset check moved from eval, turn attribution, cuts, row moves), `models.py` (the `names.json` schema), `inputs.py` (the identity-inputs helper), `cli.py` (`fukikae names`). The stage is `src/fukikae/stages/names.py`.
- **`names.json` is always written** (verdict `off`, `none`, `holdout`, `partial` or `full`); `turns.json` and `clusters.npz` only when used. The characters stage's `inputs()` reads `names.json` to choose its other inputs.
- **`ClusterAssignment` gains** `name: str | None = None` (the key) and `source: Literal["voice", "name", "lock"] = "voice"`. Both are omitted from JSON at their defaults, so assignments written without names stay byte-identical (Review Focus 4).
- **Registry schema 4:** `characters.name_by TEXT` (`operator` | `subs` | NULL) and `character_aliases(alias PRIMARY KEY, character_id, by, language, created)`. The 3-to-4 migration sets `name_by = 'operator'` wherever `name` is set and adds each such name's fold as an operator alias (first character wins a collision; the loser is logged).
- **Evaluation mechanics live in `[names]`** (`holdout_episodes`, `hide_fraction`, `hide_seed`) and in the stage's config key, so arms are reproducible from the workspace's `fukikae.toml` alone.

## Dependencies and gating

| Task | GPU | Waits on |
|---|---|---|
| 1 Normalisation and the moves out of eval | no | |
| 2 Track choice and detection | no | 1 |
| 3 Alignment, cuts and sub-clusters | no | 1 |
| 4 The `names` stage and `[names]` config | no | 2, 3 |
| 5 Registry schema 4, aliases, `registry alias` | no | |
| 6 Characters stage with names | no | 4, 5 |
| 7 Identity inputs helper: resolve, reembed, adapt, eval identity | no | 6 |
| 8 Subtitle path attribution by name | no | 6 |
| 9 Name-confirmed references and stale auto cards | no | 6 |
| 10 `fukikae names` CLI and `eval names` | no | 7, 8, 9 |
| 11 Fairy Tail evaluation, results doc, ledger | optional (en E11 diarize, open decision 7) | 10 |

Tasks 1 and 5 can run in parallel (different files); everything else is sequential.

## File Structure

```
src/fukikae/
  names/
    __init__.py
    normalize.py   # name_key, classify, display_name, NameClass; moved: canonical_name (from eval)
    detect.py      # TrackChoice, Verdict, assess, choose_track
    align.py       # moved: alignment, check_alignment, AlignmentReport; attribute, cut_turns, refine_clusters
    models.py      # NamedLine, Piece, NamesResult (names.json)
    inputs.py      # identity_inputs(work) -> (turns_path, clusters_path, names_used)
    cli.py         # `fukikae names scan|show`
  stages/names.py  # NamesStage
  stages/characters.py   # name mapping, protections, inputs via names.json
  registry/store.py      # schema 4, aliases, name_by, merge/prune alias handling
  registry/cli.py        # `registry alias`, show aliases, rename by operator
  registry/resolve.py, registry/reembed.py   # identity_inputs; name anchors in resolve
  cues.py                # assign_subtitle_speakers(name_map=, hidden=)
  stages/translate.py    # names.json input, name map from assignments
  cards/auto.py          # name-confirmed candidates, stale auto re-pick
  models.py              # ClusterAssignment.name, .source
  config.py              # [names] validation, R3 exemption
  eval/groundtruth/base.py   # re-exports from fukikae.names
  eval/identity/segments.py  # segments from assignments.turns with --from-assignments
  eval/names.py          # `fukikae eval names`
tests/
  names/test_normalize.py, test_detect.py, test_align.py
  stages/test_names.py, test_characters.py (extended)
  registry/test_store.py, test_cli.py, test_resolve.py, test_reembed.py (extended)
  cards/test_auto.py (extended), test_cues.py (extended)
  eval/test_names.py, eval/identity/ (extended)
docs/results/2026-11-1x-subtitle-speaker-names.md   # Task 11
```

---

### Task 1: Name normalisation, and the moves out of eval

**Files:**
- Create: `src/fukikae/names/__init__.py`, `src/fukikae/names/normalize.py`, `src/fukikae/names/align.py` (moved code only in this task)
- Modify: `src/fukikae/eval/groundtruth/base.py` (re-export `canonical_name`, `alignment`, `check_alignment`, `AlignmentReport`, `AlignmentError` from `fukikae.names`)
- Test: `tests/names/__init__.py`, `tests/names/test_normalize.py`

**Interfaces (`fukikae.names.normalize`):**
- `canonical_name(name: str) -> str` (moved unchanged)
- `class NameClass(StrEnum): NAMED, ROLE, MULTI, IGNORED`
- `@dataclass(frozen=True) class ParsedName: raw: str; key: str | None; display: str | None; cls: NameClass; inner: bool`
- `parse_name(raw: str) -> ParsedName`: strips `/I|/OS|/OC|/VO|(V.O.)|(O.S.)` (sets `inner`), canonicalises, folds (NFKC, casefold); `IGNORED` for `subtitles.SIGN_NAMES | SONG_NAMES` and empty; `MULTI` for `/`, `&`, `,`, word `and`, or `GROUP_WORDS = {"all", "everyone", "both", "crowd", "together", "all together"}`; `ROLE` when the canonical name ends in one letter or up to two digits after a space; else `NAMED`.
- `display_name(canonical: str) -> str`: title case per word when the name has no lower-case letter.
- `fold(text: str) -> str` (the registry's `_fold`, moved here; `registry.store._fold` imports it).
- `suggest_aliases(counts: Mapping[str, int]) -> list[tuple[str, str]]`: (rare, common) pairs at Levenshtein distance 1, both keys at least 5 characters, rare under 5 lines and common at least 5 times the rare's count.

- [ ] **Step 1: Write the failing tests**

```python
# tests/names/test_normalize.py
def test_inner_voice_suffix_is_stripped_and_flagged():
    p = parse_name("WENDY/I")
    assert (p.key, p.inner, p.cls) == ("wendy", True, NameClass.NAMED)

def test_possession_spellings_share_one_key():
    assert parse_name("MAKAROV(WRAITH)").key == parse_name("MAKAROV (WRAITH)").key == "makarov (wraith)"

def test_slash_between_two_names_is_multi():
    assert parse_name("NATSU/ZEREF").cls is NameClass.MULTI
    assert parse_name("MAKAROV/WRAITH").cls is NameClass.MULTI

def test_role_labels():
    for raw in ("TOWNSPEOPLE A", "WIZARD B", "Guard 2"):
        assert parse_name(raw).cls is NameClass.ROLE

def test_sign_and_song_names_are_ignored_but_ed_is_a_speaker():
    assert parse_name("SIGN").cls is NameClass.IGNORED
    assert parse_name("Ed").cls is NameClass.NAMED

def test_width_and_case_fold_together():
    assert parse_name("ＮＡＴＳＵ").key == parse_name("Natsu").key == "natsu"

def test_display_title_cases_all_caps_only():
    assert display_name("YOUNG MAKAROV") == "Young Makarov"
    assert display_name("McGonagall") == "McGonagall"

def test_group_words_are_multi():
    assert parse_name("EVERYONE").cls is NameClass.MULTI

def test_typo_suggestion():
    assert suggest_aliases({"nebaru": 51, "nabaru": 3}) == [("nabaru", "nebaru")]
    assert suggest_aliases({"lucy": 80, "luce": 2}) == []   # under 5 characters

def test_eval_still_imports_moved_names():
    from fukikae.eval.groundtruth.base import canonical_name as c, check_alignment
    assert c("ERZA/I") == "ERZA"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/names/test_normalize.py -v`
Expected: FAIL (`fukikae.names` does not exist)

- [ ] **Step 3: Implement** `normalize.py`; move `canonical_name`, `alignment`, `check_alignment`, `AlignmentReport`, `AlignmentError` and their constants into `fukikae.names` (`align.py` for the alignment pieces) with eval re-exporting them; point `registry.store._fold` at `fukikae.names.normalize.fold`.

- [ ] **Step 4: Run the whole suite and the R1 check**

Run: `uv run --directory $REPO pytest -q` and `grep -rn "fukikae.eval" $REPO/src/fukikae --include=*.py | grep -v "^$REPO/src/fukikae/eval/"`
Expected: PASS; the grep prints nothing new (only the existing CLI registration lines, if any).

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/names src/fukikae/eval/groundtruth/base.py src/fukikae/registry/store.py tests/names
git -C $REPO commit -m "feat(names): subtitle name keys and classes; move name canonicalisation and alignment out of eval"
```

---

### Task 2: Track choice and detection

**Files:**
- Create: `src/fukikae/names/detect.py`, `src/fukikae/names/models.py` (`NamedLine` only here)
- Test: `tests/names/test_detect.py`

**Interfaces:**
- `@dataclass(frozen=True) class NamedLine: id: int; start: float; end: float; raw: str; key: str | None; display: str | None; cls: NameClass; inner: bool; text: str` (pydantic in `models.py` so it serialises into `names.json`)
- `dialogue_lines(cues: list[Cue], rules=RULES) -> list[NamedLine]`: `merge_layers`, drop ruby, `filter_cue` hits and empty `spoken_text`; two events with the same span and different keys both become `MULTI`.
- `@dataclass(frozen=True) class Verdict: level: Literal["full", "partial", "none"]; reason: str | None; dialogue: int; named: int; share: float; distinct: int; top_share: float; offset_s: float; align_share: float`
- `assess(lines: list[NamedLine], styles: set[str], speech: list[Turn], *, min_coverage=0.5, use="auto", align=True) -> tuple[Verdict, list[NamedLine]]`: returns the lines moved back by the applied offset. Constants `FULL_SHARE = 0.9`, `MAX_TOP_SHARE = 0.8`, `MAX_STYLE_SHARE = 0.5`.
- `@dataclass(frozen=True) class TrackChoice: file: Path | None; track: SubTrack | None; language: str | None; how: Literal["file", "track", "target", "other-language"]`
- `candidate_tracks(source: Path, probe: ProbeResult, cfg: Config) -> list[TrackChoice]`: `[names] file` (with `{episode}`), `[names] track`, then `media.find_target_subtitles(...)` with `[subs]` settings, then `media.dialogue_track(probe, lang)` for each other language in probe order. Non-ASS candidates are kept (they assess as `none`, reason "format has no Name field").

- [ ] **Step 1: Write the failing tests**

```python
# tests/names/test_detect.py  (ASS built in the test from (start, end, style, name, text) tuples)
def test_crunchyroll_style_track_is_full():
    v, _ = assess(dialogue_lines(cues(NAMED_EPISODE)), STYLES, speech_under(NAMED_EPISODE))
    assert v.level == "full" and v.distinct >= 2

def test_blank_names_are_none():
    v, _ = assess(dialogue_lines(cues(BLANK_EPISODE)), STYLES, SPEECH)
    assert v.level == "none" and "share" in v.reason

def test_one_name_everywhere_is_refused():
    v, _ = assess(dialogue_lines(cues(all_named("Default"))), STYLES, SPEECH)
    assert v.level == "none" and "dominant" in v.reason

def test_names_equal_to_styles_are_refused():
    v, _ = assess(dialogue_lines(cues(names_are_styles())), {"Main", "Flashback"}, SPEECH)
    assert v.level == "none" and "style" in v.reason

def test_misaligned_track_is_refused_unless_align_off():
    shifted = shift(NAMED_EPISODE, 30.0)          # beyond the 10 s search
    assert assess(dialogue_lines(cues(shifted)), STYLES, SPEECH)[0].level == "none"
    assert assess(dialogue_lines(cues(shifted)), STYLES, SPEECH, use="on", align=False)[0].level != "none"

def test_small_offset_is_applied_to_lines():
    v, lines = assess(dialogue_lines(cues(shift(NAMED_EPISODE, 1.5))), STYLES, speech_under(NAMED_EPISODE))
    assert v.offset_s == pytest.approx(1.5, abs=0.1) and lines[0].start == pytest.approx(NAMED_EPISODE[0][0], abs=0.1)

def test_partial_between_half_and_ninety_percent():
    assert assess(dialogue_lines(cues(named_fraction(0.6))), STYLES, SPEECH)[0].level == "partial"

def test_signs_and_next_episode_title_are_not_dialogue():
    lines = dialogue_lines(cues([(1, 4, "White-Standard", "SIGN", "{\\pos(320,238)}Title"), *NAMED_EPISODE]))
    assert all(ln.raw != "SIGN" for ln in lines)

def test_simultaneous_different_names_become_multi():
    lines = dialogue_lines(cues([(10, 12, "Italics", "CHARLE", "x"), (10, 12, "Italics", "TOUKA", "x")]))
    assert {ln.cls for ln in lines} == {NameClass.MULTI}

def test_candidate_order_prefers_target_dialogue_track_then_other_languages(probe_with_en_and_de):
    got = candidate_tracks(SOURCE, probe_with_en_and_de, cfg(target_language="de"))
    assert [c.how for c in got] == ["target", "other-language"] and got[1].language == "en"

def test_srt_candidate_assesses_none(tmp_path): ...   # reason names the format
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/names/test_detect.py -v`
Expected: FAIL (no `fukikae.names.detect`)

- [ ] **Step 3: Implement** `detect.py` and `NamedLine`. The style set comes from the ASS `[V4+ Styles]` section (`subtitles.parse_ass_styles(text) -> set[str]`, a small addition beside `parse_ass_cues`).

- [ ] **Step 4: Run** `uv run --directory $REPO pytest tests/names tests/test_subtitles.py tests/test_media.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/names/detect.py src/fukikae/names/models.py src/fukikae/subtitles.py tests/names/test_detect.py
git -C $REPO commit -m "feat(names): choose a named subtitle track and decide whether its names are usable"
```

---

### Task 3: Alignment, cuts and sub-clusters

**Files:**
- Modify: `src/fukikae/names/align.py`, `src/fukikae/names/models.py` (`Piece`, `LabelInfo`)
- Test: `tests/names/test_align.py`

**Interfaces:**
- `class Piece(BaseModel): start: float; end: float; cluster: str; label: str; key: str | None; line_ids: list[int]`
- `class LabelInfo(BaseModel): parent: str; key: str | None; display: str | None; cls: Literal["named", "role", "remainder"]`
- `attribute(turns: list[Turn], lines: list[NamedLine], *, turn_share=0.6, cut_side_s=0.5, min_piece_s=0.3) -> tuple[list[Piece], list[int]]`: pieces covering every turn (named or remainder) and the ids of unvoiced lines (under `UNVOICED_S = 0.2` of turn).
- `label_for(cluster: str, key: str) -> str`: `<cluster>~<slug>`; slug is `[a-z0-9-]` from the key when it is ASCII, else `n` plus the first 8 hex of sha256(key).
- `refine_clusters(clusters_npz: Path, pieces: list[Piece], turns_out: list[Turn], out: Path, *, contain=0.9) -> float`: writes the refined npz in the diarize stage's format, returns seconds of rows dropped across cuts; `clean_seconds` per label from `stages.diarize.clean_spans(turns_out, label)`.
- `excluded_spans(lines: list[NamedLine]) -> list[tuple[float, float]]`: inner-voice and multi lines' spans, for reference selection.

- [ ] **Step 1: Write the failing tests**

```python
# tests/names/test_align.py
def test_turn_covered_by_one_name_takes_it():
    pieces, _ = attribute([T(10, 14, "SPEAKER_12")], [L(10.1, 13.9, "natsu")])
    assert [(p.label, p.key) for p in pieces] == [("SPEAKER_12~natsu", "natsu")]

def test_turn_under_share_stays_remainder():
    pieces, _ = attribute([T(10, 20, "SPEAKER_12")], [L(10, 14, "natsu")])   # 40%
    assert pieces[0].label == "SPEAKER_12" and pieces[0].key is None

def test_two_names_in_one_turn_cut_at_gap_midpoint():
    pieces, _ = attribute([T(1300.3, 1305.9, "SPEAKER_12")], [L(1300.4, 1302.8, "happy"), L(1303.2, 1305.8, "natsu")])
    assert [(p.key, round(p.end, 2)) for p in pieces] == [("happy", 1303.0), ("natsu", 1305.9)]

def test_sliver_piece_joins_neighbour():
    pieces, _ = attribute([T(0, 5, "S")], [L(0, 4.8, "lucy"), L(4.8, 5.0, "cana")])
    assert len(pieces) == 1 and pieces[0].key == "lucy"

def test_multi_line_never_names_and_never_cuts():
    pieces, _ = attribute([T(0, 4, "S")], [L(0, 4, None, cls=NameClass.MULTI)])
    assert pieces[0].key is None

def test_overlapping_turns_one_line_names_the_larger_overlap_only():
    pieces, _ = attribute([T(0, 4, "A"), T(3, 5, "B")], [L(0.2, 3.9, "lucy")])
    assert {p.cluster: p.key for p in pieces} == {"A": "lucy", "B": None}

def test_unvoiced_line_reported():
    _, unvoiced = attribute([T(0, 2, "A")], [L(0.1, 1.9, "lucy", id=0), L(50, 52, "erza", id=1)])
    assert unvoiced == [1]

def test_inner_voice_line_names_its_key():
    pieces, _ = attribute([T(0, 3, "A")], [L(0, 3, "wendy", inner=True)])
    assert pieces[0].key == "wendy"

def test_non_ascii_key_gets_hashed_slug():
    assert re.fullmatch(r"SPEAKER_01~n[0-9a-f]{8}", label_for("SPEAKER_01", "なつ"))

def test_rows_move_with_their_turn_and_cut_rows_drop(tmp_path):
    # npz with SPEAKER_12 rows for spans inside turn 1 (named natsu) and across the cut in turn 2
    dropped = refine_clusters(npz, pieces, refined_turns, tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz") as z:
        assert "SPEAKER_12~natsu" in [str(x) for x in z["labels"]]
        assert z["SPEAKER_12~natsu/embeddings"].shape[0] == 1
    assert dropped == pytest.approx(5.4, abs=0.01)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/names/test_align.py -v`
Expected: FAIL

- [ ] **Step 3: Implement** in `align.py`.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest tests/names -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/names/align.py src/fukikae/names/models.py tests/names/test_align.py
git -C $REPO commit -m "feat(names): attribute diarize turns to named lines, cut two-name turns, split cluster rows"
```

---

### Task 4: The `names` stage and `[names]` config

**Files:**
- Create: `src/fukikae/stages/names.py`, `src/fukikae/names/inputs.py`
- Modify: `src/fukikae/names/models.py` (`NamesResult`), `src/fukikae/config.py` (`[names]` validation, R3 exemption), `src/fukikae/cli.py` (`pipeline_stages`)
- Test: `tests/stages/test_names.py`, `tests/test_config.py`

**Interfaces:**
- `class NamesResult(BaseModel): schema_version: int = 1; episode: str; used: bool; verdict: Literal["off", "none", "holdout", "partial", "full"]; reason: str | None; source: dict | None; offset_s: float; coverage: dict[str, float | int]; lines: list[NamedLine]; hidden: list[int]; pieces: list[Piece]; labels: dict[str, LabelInfo]; unvoiced: list[int]; excluded: list[tuple[float, float]]; dropped_row_seconds: float; suggestions: list[tuple[str, str]]`
- `NamesStage(name="names")`: `inputs(ep)` = `04-diarize/turns.json`, `04-diarize/clusters.npz`, plus the names file when `[names] file` is set; `outputs(ep)` = `[04c-names/names.json]`; `config_key` = the resolved `[names]` settings (with `holdout` true when `ep.id` is in `holdout_episodes`), the chosen source (file path or track index and language) and `NAMES_VERSION`. `run` writes `names.json` always, and `turns.json` plus `clusters.npz` only when used.
- Hidden lines: when `hide_fraction > 0`, per key with at least 5 named lines, `round(fraction * n)` line ids drawn with `random.Random(f"{hide_seed}:{episode}:{key}")`; hidden lines keep their key in `lines` but act as unnamed in `attribute`.
- `identity_inputs(work: Path) -> tuple[Path, Path, NamesResult | None]` (`fukikae.names.inputs`): the refined pair and the result when `names.json` says used, else `04-diarize`'s pair and the result (or None when the stage never ran).
- `[names]` keys: `use` (auto|on|off), `file`, `track`, `min_coverage`, `turn_share`, `min_new_seconds`, `align`, `holdout_episodes`, `hide_fraction` (0 to 0.9), `hide_seed`; unknown keys are a `ConfigError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/stages/test_names.py  (a workspace fixture with 04-diarize outputs and a sidecar .en.ass)
def test_named_sidecar_writes_refined_turns_and_clusters(ws_named):
    NamesStage().run(ep, cfg)
    r = read_json(ep.work / "04c-names" / "names.json", NamesResult)
    assert r.used and r.verdict == "full"
    assert (ep.work / "04c-names" / "turns.json").exists()

def test_blank_names_write_only_names_json(ws_blank):
    NamesStage().run(ep, cfg)
    assert not read_json(names_json, NamesResult).used
    assert not (ep.work / "04c-names" / "turns.json").exists()

def test_use_off_reads_no_subtitles(ws_named, monkeypatch):
    monkeypatch.setattr(media, "extract_subtitle", explode)
    NamesStage().run(ep, cfg_with(use="off"))
    assert read_json(names_json, NamesResult).verdict == "off"

def test_holdout_episode_is_unused_and_reads_no_names(ws_named):
    NamesStage().run(ep, cfg_with(holdout_episodes=[ep.id]))
    r = read_json(names_json, NamesResult)
    assert r.verdict == "holdout" and not r.used and r.lines == []

def test_hidden_lines_are_stable_and_act_unnamed(ws_named):
    a = run_hidden(seed=0); b = run_hidden(seed=0)
    assert a.hidden == b.hidden and a.hidden
    assert all(p.key is None for p in a.pieces if set(p.line_ids) <= set(a.hidden))

def test_config_key_changes_with_holdout_and_hide(ws_named):
    assert key(cfg) != key(cfg_with(holdout_episodes=[ep.id])) != key(cfg_with(hide_fraction=0.2))

def test_identity_inputs_follow_names_json(ws_named, ws_blank):
    assert identity_inputs(named_work)[0].parent.name == "04c-names"
    assert identity_inputs(blank_work)[0].parent.name == "04-diarize"

def test_pipeline_runs_names_between_diarize_and_characters():
    names = [s.name for s in pipeline_stages(cfg)]
    assert names.index("names") == names.index("diarize") + 1 == names.index("characters") - 1

# tests/test_config.py
def test_names_table_rejects_unknown_key(): ...
def test_names_table_needs_no_backend(): ...      # R3 exemption
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/stages/test_names.py tests/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement** the stage, the result model, `identity_inputs`, the config validation and the pipeline insertion.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS (existing `test_cli.py` stage-order assertions updated in this task).

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/stages/names.py src/fukikae/names/inputs.py src/fukikae/names/models.py src/fukikae/config.py src/fukikae/cli.py tests/stages/test_names.py tests/test_config.py tests/test_cli.py
git -C $REPO commit -m "feat(names): names stage between diarize and characters, with [names] settings and evaluation hooks"
```

---

### Task 5: Registry schema 4, aliases and `registry alias`

**Files:**
- Modify: `src/fukikae/registry/store.py`, `src/fukikae/registry/cli.py`
- Test: `tests/registry/test_store.py`, `tests/registry/test_cli.py`

**Interfaces (`fukikae.registry.store`):**
- `SCHEMA_VERSION = 4`; `_migrate_3_to_4`: adds `characters.name_by`, creates `character_aliases`, sets `name_by = 'operator'` for named characters and adds their folds as operator aliases, in one transaction.
- `Character` gains `name_by: str | None`.
- `@dataclass(frozen=True) class Alias: alias: str; character_id: int; by: str; language: str | None; created: str`
- `character_for_key(key: str) -> Character | None`: alias table first, then a character whose folded name equals the key (ValueError naming both when two characters answer).
- `aliases(character_id: int | None = None) -> list[Alias]`
- `add_alias(character_id, key, by: Literal["subs", "operator"], language: str | None = None, replace: bool = False) -> Alias`: a `subs` alias never replaces any existing alias; an operator alias replaces a `subs` one and refuses another operator's unless `replace` (`AliasConflictError`).
- `remove_alias(key: str) -> None`
- `name_character(character_id, display, key, language) -> None`: subs naming; refuses when the character is named (`name_by` set) or curated by lock.
- `rename_character(character_id, name, by="operator")`: sets `name_by`, adds the fold as an operator alias unless another character holds it (returns a warning string).
- `merge_actors` moves the dropped characters' aliases to the target (an operator alias wins a key collision); `prune_actors` deletes pruned characters' aliases.
- CLI: `fukikae registry alias <ws> <character> <NAME> [--remove] [--replace]` (NAME is folded with `parse_name`; refuses MULTI and IGNORED names); `registry show` prints `name_by` and aliases.

- [ ] **Step 1: Write the failing tests**

```python
# tests/registry/test_store.py
def test_v3_registry_migrates_names_to_operator_aliases(tmp_path):
    # schema-3 file built with sqlite3: character 9 named "Natsu"
    with Registry(p) as reg:
        assert reg.characters()[0].name_by == "operator"
        assert reg.character_for_key("natsu").id == 9

def test_subs_alias_never_replaces_an_operator_alias():
    reg.add_alias(1, "natsu", by="operator")
    with pytest.raises(AliasConflictError):
        reg.add_alias(2, "natsu", by="subs")

def test_operator_alias_replaces_subs_alias():
    reg.add_alias(1, "nabaru", by="subs"); reg.add_alias(2, "nabaru", by="operator")
    assert reg.character_for_key("nabaru").id == 2

def test_name_character_refuses_an_operator_named_character():
    reg.rename_character(1, "Natsu Dragneel")
    with pytest.raises(ValueError):
        reg.name_character(1, "Natsu", "natsu", "en")

def test_operator_rename_keeps_the_subs_key():
    reg.name_character(1, "Natsu", "natsu", "en"); reg.rename_character(1, "Natsu Dragneel")
    assert reg.character_for_key("natsu").id == 1 and reg.character_for_key("natsu dragneel").id == 1

def test_merge_moves_aliases_and_prune_deletes_them(): ...

# tests/registry/test_cli.py
def test_registry_alias_sets_and_removes(cli_ws): ...       # exit 0; show lists it; --remove clears
def test_registry_alias_refuses_multi_names(cli_ws): ...    # "NATSU/ZEREF" exits 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/registry/test_store.py tests/registry/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement** the migration, the methods and the CLI.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS (the existing schema-1 and schema-2 migration tests now end at version 4).

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/registry/store.py src/fukikae/registry/cli.py tests/registry/test_store.py tests/registry/test_cli.py
git -C $REPO commit -m "feat(registry): schema 4 with name provenance, character aliases and registry alias"
```

---

### Task 6: Characters stage with names

**Files:**
- Modify: `src/fukikae/stages/characters.py`, `src/fukikae/models.py` (`ClusterAssignment.name`, `.source`, excluded at defaults), `src/fukikae/registry/matching.py` (`match(..., name_claimed: set[int])`)
- Test: `tests/stages/test_characters.py`, `tests/test_models.py`

**Interfaces:**
- `CharactersStage.inputs(ep)`: `04c-names/names.json` plus `identity_inputs(ep.work)[:2]`.
- `CharactersStage.config_key`: adds `names_min_new_seconds` only when names were used for the episode (the no-names key is unchanged).
- `resolve_names(reg, cfg, result: NamesResult, evidence, rows, pinned, locked_actors) -> tuple[dict[str, ClusterAssignment], list[str]]`: the spec's order (lock, known key, role or short, rename, create), returning assignments for every named label and report lines.
- Constants: `RENAME_SHARE = 0.6` (both directions), `NAMES_MIN_NEW_S = 5.0` (overridden by `[names] min_new_seconds`).
- Remainder clusters go through the existing pin and match path; `match` treats `name_claimed` actors as unclaimed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/stages/test_characters.py  (fake names.json, refined turns and npz)
def test_two_names_in_one_cluster_become_two_characters(ws):
    run_characters(ws, names={"SPEAKER_12~wendy": "wendy", "SPEAKER_12~natsu": "natsu"})
    by = {a.cluster: a for a in read_assignments(ws).clusters}
    assert by["SPEAKER_12~wendy"].character_id != by["SPEAKER_12~natsu"].character_id
    assert by["SPEAKER_12~natsu"].source == "name"

def test_known_key_takes_several_clusters_of_one_episode(ws):
    reg_with_alias("natsu", character=9)
    run_characters(ws, names={"SPEAKER_04~natsu": "natsu", "SPEAKER_09~natsu": "natsu"})
    assert {a.character_id for a in named(read_assignments(ws))} == {9}
    assert not any(a.status == "review" for a in named(read_assignments(ws)))

def test_unnamed_character_is_renamed_when_one_name_dominates(ws):
    previous_run_pins("SPEAKER_08", character=2)                 # unnamed, auto card only
    run_characters(ws, names=e14_speaker_08())                   # wendy 129 s of 208 s
    assert registry(ws).character_by_ref("2").name == "Wendy"
    assert registry(ws).characters_named("Happy")                # split out, created

def test_curated_character_is_not_renamed(ws):
    lock_card(character=2)
    run_characters(ws, names=e14_speaker_08())
    assert registry(ws).character_by_ref("2").name is None

def test_locked_actor_keeps_its_cluster_and_reports(ws, caplog):
    previous_run_pins("SPEAKER_08", character=2); lock_actor_of(2)
    run_characters(ws, names=e14_speaker_08())
    assert {a.character_id for a in sub_clusters_of("SPEAKER_08")} == {2}
    assert {a.source for a in sub_clusters_of("SPEAKER_08")} == {"lock"}
    assert "kept by lock" in caplog.text

def test_operator_named_character_is_never_renamed_by_subs(ws): ...
def test_role_label_and_short_names_are_extras_with_name(ws):
    run_characters(ws, names={"SPEAKER_00~townspeople-b": "townspeople b", "SPEAKER_03~loke": "loke"})
    assert {a.status for a in named(read_assignments(ws))} == {"extra"}
    assert registry_character_count(ws) == before

def test_remainder_still_matches_acoustically(ws): ...
def test_without_names_assignments_are_byte_identical(ws_blank):
    before = (ws_blank.assignments).read_bytes(); rerun_characters(ws_blank)
    assert ws_blank.assignments.read_bytes() == before

# tests/test_models.py
def test_cluster_assignment_defaults_are_omitted_from_json(): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/stages/test_characters.py tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implement** `resolve_names`, the inputs switch and the model fields. Renaming and creating happen in the stage's existing registry transaction.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/stages/characters.py src/fukikae/models.py src/fukikae/registry/matching.py tests/stages/test_characters.py tests/test_models.py
git -C $REPO commit -m "feat(characters): map subtitle names onto characters under lock and operator-name protections"
```

---

### Task 7: Identity inputs everywhere: resolve, reembed, adapt, eval identity

**Files:**
- Modify: `src/fukikae/registry/resolve.py` (name anchors), `src/fukikae/registry/reembed.py`, `src/fukikae/registry/cli.py` (`adapt`, `split` accepts sub-cluster labels), `src/fukikae/eval/identity/segments.py` and `cli.py` (`--from-assignments` segments from `assignments.turns`)
- Test: `tests/registry/test_resolve.py`, `tests/registry/test_reembed.py`, `tests/registry/test_cli.py`, `tests/eval/identity/test_cli.py`

**Interfaces:**
- Every reader of `04-diarize/turns.json` or `clusters.npz` named in the spec calls `identity_inputs(work)`.
- `resolve_episodes`: named labels (`ClusterAssignment.source == "name"` or `"lock"` in the episode's current assignments) are not seeds; their rows join the anchor of their character (a character created by names that has no other exemplars becomes an anchor seeded by those rows); unnamed clusters group as before. The written assignments keep each named label's character, `name` and `source`.
- `segments_for(kind="diarization", ..., turns: list[Turn] | None = None)`: with `--from-assignments` the harness passes `assignments.turns`, so labels match `assignments.clusters`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/registry/test_resolve.py
def test_named_labels_keep_their_character_through_resolve(ws_two_named_episodes):
    resolve_episodes(ws, eps, threshold=0.6)
    for e in eps:
        assert all(a.character_id == natsu_id for a in named_as(e, "natsu"))

def test_two_keys_never_share_a_group(ws_two_named_episodes): ...   # wendy and charle stay apart at threshold 0.0

# tests/registry/test_reembed.py
def test_reembed_finds_rows_of_sub_cluster_labels(ws_named): ...

# tests/registry/test_cli.py
def test_split_accepts_a_sub_cluster_label(cli_ws_named): ...

# tests/eval/identity/test_cli.py
def test_from_assignments_scores_sub_cluster_labels(ws_named):
    run = identity(ws, tune=["S01E12"], from_assignments=True)
    assert run.episodes[0].labeled > 0 and all(s is not None for s in assigned_labels(run))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/registry tests/eval/identity -v`
Expected: FAIL on the new tests

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/registry/resolve.py src/fukikae/registry/reembed.py src/fukikae/registry/cli.py src/fukikae/eval/identity tests/registry tests/eval/identity
git -C $REPO commit -m "feat(registry): resolve, reembed, adapt and eval identity read name-refined clusters"
```

---

### Task 8: Subtitle path attribution by name

**Files:**
- Modify: `src/fukikae/cues.py` (`assign_subtitle_speakers`), `src/fukikae/stages/translate.py` (inputs, name map)
- Test: `tests/test_cues.py`, `tests/stages/test_translate.py`

**Interfaces:**
- `name_map(assignments: Assignments) -> dict[str, tuple[int | None, str | None]]`: key to (character id, extra label), from each `ClusterAssignment` with `name` set.
- `assign_subtitle_speakers(..., name_map: Mapping[str, tuple[int | None, str | None]] | None = None, hidden: Collection[tuple[float, float]] = ())`: a cue whose `parse_name(cue.name).key` is in `name_map` and whose span isn't hidden takes that speaker; otherwise `_subtitle_cluster` as today.
- `TranslateStage.inputs` adds `04c-names/names.json` when it exists; `run` builds `name_map` from the assignments and `hidden` from `names.json` (the hidden lines' spans).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cues.py
def test_cue_name_beats_turn_overlap():
    cues = [Cue(1300.3, 1302.8, "I'd even settle for Jellal!", name="NATSU")]
    out = assign_subtitle_speakers(cues, turns_all_speaker_12, asg_speaker_12_is_char2, None, None, {}, "en", 1.0, RULES,
                                   name_map={"natsu": (9, None)})
    assert out[0].character_id == 9

def test_hidden_cue_falls_back_to_turn_overlap():
    out = assign_subtitle_speakers(..., name_map={"natsu": (9, None)}, hidden=[(1300.3, 1302.8)])
    assert out[0].character_id == 2

def test_unknown_or_multi_name_falls_back(): ...
def test_named_extra_gets_its_extra_label(): ...

# tests/stages/test_translate.py
def test_translate_key_changes_when_names_json_changes(ws_named): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/test_cues.py tests/stages/test_translate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cues.py src/fukikae/stages/translate.py tests/test_cues.py tests/stages/test_translate.py
git -C $REPO commit -m "feat(translate): a subtitle cue's own speaker name picks its character"
```

---

### Task 9: Name-confirmed references and stale automatic cards

**Files:**
- Modify: `src/fukikae/cards/auto.py` (`character_candidates`, `character_references`, `ensure_assignments`), `src/fukikae/cards/cli.py` (`voices list` flag)
- Test: `tests/cards/test_auto.py`, `tests/cards/test_cli.py`

**Interfaces:**
- `EpisodeArtifacts` gains `names: NamesResult | None` (from `identity_inputs`).
- `character_candidates`: in a names-used episode, `speakers` = the character's labels with `source in {"name", "lock"}`; `exclude` adds `names.excluded`.
- `character_references(...) -> tuple[str, list[RefPick], bool] | None`: the third value says the picks are name-confirmed; names-used episodes are tried first and acoustic-only ones only when none yields picks.
- `confirmed_spans(ws, character_id) -> dict[str, list[tuple[float, float]]]`: per names-used episode, the character's name-confirmed pieces.
- `stale_references(card: Card, confirmed: dict[str, list[tuple[float, float]]]) -> list[dict]`: reference origins in names-used episodes outside confirmed spans (90% containment).
- `ensure_assignments` re-picks a character whose current card's `chosen_by` is in `AUTOMATIC_BY` and whose card has stale references (new `by = "auto"` row, note `re-picked: references not name-confirmed`); curated and imported cards get a report line only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/cards/test_auto.py
def test_candidates_use_only_name_confirmed_turns(ws_named):
    got = character_candidates(ws, ["S01E14"], natsu_id)
    assert all(inside(p, natsu_pieces) for p in got["S01E14"])

def test_inner_voice_and_multi_spans_are_never_references(ws_named): ...

def test_named_episode_preferred_over_louder_acoustic_episode(ws_mixed):
    ep, picks, confirmed = character_references(ws, ["S01E13", "S01E14"], cid)
    assert ep == "S01E14" and confirmed

def test_auto_card_with_stale_references_is_repicked(ws_named):
    assign_auto_clone(character=2, origins=[("S01E14", 1300.3, 1305.9)])   # now named happy+natsu
    lines = ensure_assignments(reg, store, cfg, spans, vocals, cloning=fake_cloning)
    assert reg.card_for(2, "en").note == "re-picked: references not name-confirmed"

def test_assigned_or_imported_card_is_flagged_not_repicked(ws_named):
    assign_card(2, by="assign"); lines = ensure_assignments(...)
    assert reg.card_for(2, "en").by == "assign" and any("voices clone" in ln for ln in lines)

def test_restock_card_with_stale_references_is_repicked(ws_named): ...   # CD9: restock is automatic
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/cards/test_auto.py tests/cards/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement.** The tts stage's existing `prepare` calls `ensure_assignments`, so no stage change is needed beyond passing names-aware spans.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/cards/auto.py src/fukikae/cards/cli.py tests/cards/test_auto.py tests/cards/test_cli.py
git -C $REPO commit -m "feat(cards): clone references from name-confirmed turns; re-pick automatic cards the names contradict"
```

---

### Task 10: `fukikae names` CLI and `eval names`

**Files:**
- Create: `src/fukikae/names/cli.py`, `src/fukikae/eval/names.py`
- Modify: `src/fukikae/cli.py` (register `names_app`), `src/fukikae/eval/cli.py` (register `names`), `README.md`, `CLAUDE.md` (one bullet under Rules)
- Test: `tests/names/test_cli.py`, `tests/eval/test_names.py`

**Interfaces:**
- `fukikae names scan <ws> [episodes...]`: per episode, the candidate tracks tried, the chosen one, its verdict line and suggestions; read-only (extracted tracks cached under `.fukikae/cache/subtitles/`).
- `fukikae names show <ws> <episode>`: the spec's list (verdict, track, offset, coverage, per-key seconds and labels, cuts, unvoiced, dropped seconds, each key's character and how, voice disagreements lowest first with cosine to the centroid built from other episodes' exemplars, suggestions with the `registry alias` command, stale and flagged cards, characters still holding exemplars from episodes run without names).
- `fukikae eval names <ws> --tune E,... [--holdout E] --gt ass [--gt-file] [--alias] [--test-set] [--tag] [--no-ledger] [--json]`:
  - `NamesScore(cue_accuracy: float, cue_seconds: float, per_character: dict[str, float], reference_purity: float, purity_per_character: dict[str, float], hidden_accuracy: float | None, coverage: dict[str, float])`
  - cue attribution recomputed with `assign_subtitle_speakers` from each episode's assignments, `name_map` and `hidden` (so held-out and hidden lines never see their names); modal names from the tuning episodes' speech; reference picks from `character_references` (no rendering).
  - Ledger rows: `stage = "names"`, `backend = "subtitle-names"`, `options` = the resolved `[names]` table, splits `tune`, `holdout`, `hidden`.
- README: the names stage, `[names]`, the manual fallback, the migration commands. CLAUDE.md: one Rules bullet (names win over acoustics for covered speech; locks and operator names win over names; `[names]` exempt from R3; pipeline reads registry aliases, eval reads `labels-aliases.yaml`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/names/test_cli.py  (CliRunner)
def test_scan_prints_verdict_and_changes_nothing(cli_ws_named):
    before = snapshot(cli_ws_named); r = runner.invoke(app, ["names", "scan", str(ws)])
    assert r.exit_code == 0 and "full" in r.stdout and snapshot(cli_ws_named) == before

def test_show_lists_split_keys_and_typo_suggestion(cli_ws_named):
    r = runner.invoke(app, ["names", "show", str(ws), "S01E12"])
    assert "natsu" in r.stdout and "registry alias" in r.stdout

# tests/eval/test_names.py
def test_holdout_cue_accuracy_never_reads_holdout_names(ws_holdout):
    s = names_eval(ws, tune=["S01E12"], holdout="S01E14")
    assert s.holdout.cue_accuracy == pytest.approx(expected_from_turn_overlap_only)

def test_reference_purity_counts_wrong_speaker_seconds(ws_merged_refs):
    assert names_eval(ws, tune=["S01E12"]).tune.purity_per_character["Natsu"] == pytest.approx(0.0)

def test_hidden_accuracy_scores_only_hidden_lines(ws_hidden): ...
def test_ledger_row_has_stage_names_and_split(ws_named, tmp_ledger): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory $REPO pytest tests/names/test_cli.py tests/eval/test_names.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**, then the README and CLAUDE.md text.

- [ ] **Step 4: Run** `uv run --directory $REPO pytest -q`, `uv run --directory $REPO ruff check`, `uv run --directory $REPO ruff format --check`, `uv run --directory $REPO pyright`. Expected: all clean.

- [ ] **Step 5: Commit**

```bash
git -C $REPO add src/fukikae/names/cli.py src/fukikae/eval/names.py src/fukikae/cli.py src/fukikae/eval/cli.py README.md CLAUDE.md tests/names/test_cli.py tests/eval/test_names.py
git -C $REPO commit -m "feat(names): names scan and show, eval names with held-out and hidden-line scoring"
```

---

### Task 11: Fairy Tail evaluation, results doc and ledger

**Files:**
- Create: `docs/results/2026-11-1x-subtitle-speaker-names.md` (dated the day it is written), `scripts/names-eval/run_arms.sh` (the exact commands, committed for reproducibility)
- Modify: the execution ledger (rulings with numbers)

No code changes. CPU only, except the optional en E11 run.

- [ ] **Step 1: Workspaces.** `cp -al ~/fukikae-scratch/t12/fairytail ~/fukikae-ws/t-names/ft-ja-<arm>` for arms `baseline`, `names`, `ceiling`, `hidden`, `hidden-base`. Stage outputs are written by rename, so hard links are safe for them; the files changed in place are not. In each copy, replace `.fukikae/registry`, `.fukikae/hashcache.json` and `fukikae.toml` with real copies (`cp -a --remove-destination <t12 file> <copy file>`) and check `stat -c %h` prints 1 for each before anything runs. Then set `[names]` per the spec's protocols in the copy's `fukikae.toml`.
- [ ] **Step 2: Scan.** `mise -C $REPO exec -- uv run --directory $REPO fukikae names scan ~/fukikae-ws/t-names/ft-ja-names` records each episode's verdict (expect `full` on E11 to E14, E11 via its forced-flagged full track). Paste the table into the results doc.
- [ ] **Step 3: Run each arm.** Per arm: `fukikae run <ws> S01E11 S01E12 S01E13 S01E14 --from names --to characters`, then `fukikae registry resolve <ws> --episodes S01E11,S01E12,S01E13`, then `fukikae registry resolve <ws> --episodes S01E14`.
- [ ] **Step 4: Score.** Per arm: `fukikae eval identity <ws> --from-assignments --tune S01E11,S01E12,S01E13 --holdout S01E14 --gt ass --test-set fairytail --tag names-<arm>`; `fukikae eval names <ws> --tune S01E11,S01E12,S01E13 --holdout S01E14 --gt ass --test-set fairytail --tag names-<arm>`. Hidden arms: `eval names` without `--holdout`, reporting `hidden_accuracy`.
- [ ] **Step 5: en track.** If open decision 7 approves: E11 through diarize on GPU 0 in a copy of `~/fukikae-ws/t9hold/fairytail-en` (`fukikae run <ws> S01E11 --to diarize`), then Steps 3 and 4. Otherwise the en rows use `--tune S01E12,S01E13 --holdout S01E14`, labelled as such.
- [ ] **Step 6: Results doc.** Tables per set: protocol A (baseline, names, ceiling: identity harness held-out row, cue accuracy, reference purity), protocol B (hidden accuracy names vs baseline), coverage (named share, labels, cuts, dropped seconds, unvoiced, suggestions), the character-2 case by name (who now owns Natsu's E14 lines and E12's c0331, and what character 2's re-picked references are), runtime of the names stage per episode. State that the ceiling arm grades on its own input and that the ja and en sets are the same episodes.
- [ ] **Step 7: Rulings.** Every surprise is a ruling in the execution ledger with its numbers; no threshold changes on E14.
- [ ] **Step 8: Commit**

```bash
git -C $REPO add docs/results/2026-11-1x-subtitle-speaker-names.md scripts/names-eval/run_arms.sh
git -C $REPO commit -m "docs(results): subtitle speaker names on Fairy Tail E11-E14, held-out and hidden-line protocols"
```

A listening check (re-rendering E12 to E14's affected cues with IndexTTS2 on GPU 0) is not part of this task; it waits on open decision 9.
