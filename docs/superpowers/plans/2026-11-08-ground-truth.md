# Live-Action Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speaker-identity ground truth for live action in Japanese, English, German and Spanish, a results ledger that records every evaluation by language and genre, and a live-action profile confirmed or replaced on that evidence (Ruling R16).

**Architecture:** Every evaluator appends rows to one user-level JSONL ledger, and `fukikae eval report` renders them as a matrix. Four new ground-truth paths all end in a format the existing identity harness already reads: Japanese SDH prefixes go through the `sdh` reader, broadcast colour captions through the `colour` reader with a series-wide colour map, fan transcripts become named ASS files through a text aligner, and a subtitle-to-audio retimer fixes captions captured from a different broadcast. A clip-set evaluator scores embedders by Ruling R8's nearest-centroid accuracy without any episode media. The profile re-pick then reuses Task 9's sweep and selection rule on the new tuning sets.

**Tech Stack:** Python 3.12, uv, typer, numpy, stdlib `difflib`, `html.parser`, `fcntl`; existing fukikae ground-truth readers, identity harness, embed backends and `fukikae registry resolve`; `huggingface-cli` for the clip set; ffmpeg from mise.

**Spec:** No separate spec document; the operator approved this scope in D9, D14 and D16, and the brief is the spec. The bar and metrics come from `docs/superpowers/specs/2026-11-05-speaker-identity-design.md`; the consumer of the ledger is `docs/superpowers/specs/2026-11-08-character-cards-and-dubbing-design.md` (`eval tts` writes rows in this plan's schema, and its `[series]` table grows from this plan's). Rulings R1 to R20 in `docs/superpowers/ledgers/2026-11-05-speaker-identity.md` stand. The brief's points:

1. **Results ledger first.** Append-only JSONL. One row per evaluation run: stage (asr, identity, embedder, later tts), backend, model and revision, options hash, language, genre, test set, episode split (tune or holdout), metrics, fukikae git commit, date. `eval asr` and `eval identity` append rows. `fukikae eval report` renders a matrix: rows grouped by stage, columns by language and genre, filters on stage, backend and set. Genre comes from a new minimal `[series]` table in `fukikae.toml` (`title`, `genre`, genre defaulting to the profile name), designed so the cards plan can add external ids to it.
2. **Japanese SDH speaker prefixes.** The SDH labeller learns （名前） and (名前) and strips ASS override blocks such as `{\fscx50}` before matching. Test set: Alice in Borderland (Netflix Japanese SDH, in the library); about 27% of cues carry a name, and the metric scores only labelled speech.
3. **Broadcast colour labels** (ARIB .ass, NHK asadora 風、薫る "Kaze, Kaoru"). Per-line `\c` colours mark the main cast; （名前） prefixes name some lines. Map each colour to a name from the named lines that carry it; unmapped colours stay anonymous but consistent, and the plan says whether the metric can use them. Extend `colour.py`, don't duplicate it. The captions come from a different capture than the audio (a WEB-DL, whose own .srt has names only), so a subtitle-to-audio alignment tool finds offset and linear drift between two subtitle files by matching cue onsets and text, or against speech onsets when there is no second file, and rewrites the labelled file's timing. BBT reuses it. Files land in `~/fukikae-ws/subs/kaze-kaoru/`; media comes through the library.
4. **Transcript-to-subtitle aligner.** Speaker-named, untimed transcripts are aligned by fuzzy text alignment to the release's own timed English subtitle track (Everingham/Tapaswi) and written in a format the identity harness reads. Test set: The Big Bang Theory S01 (17 episodes, English, German and Spanish audio); English-side labels score all three tracks, which share the video timeline. The plan picks the transcript source and says why.
5. **Per-character clip set for embedder checks.** `noncegeek/cantonese-drama-voice-fine-grained` (ungated): 人民的名义 (Mandarin, 23 characters, 2,410 clips), 神雕侠侣 (Cantonese) and others. An eval computes R8's nearest-centroid accuracy per embedder from a folder-per-character clip set and writes ledger rows.
6. **Baselines and the live-action profile.** Run the live-action and default profiles on BBT (en, de, es), Alice in Borderland (ja) and Kaze, Kaoru (ja). Tuning and held-out episodes are fixed in this plan with numbers. Re-pick the live-action profile by Task 9's rule (most clean characters, then fewest merges, then fraction) on tuning episodes only; score held-out once with no re-tuning; record per-language results; a language variant is allowed when the ledger shows a clear difference. Write `docs/results/2026-11-xx-live-action-ground-truth.md`, honest about coverage bias: SDH names skew towards off-screen speakers, and colour covers only the main cast.

## Global Constraints

- uv only, always `uv run --directory ~/projects/fukikae ...`; never pip, never `cd <dir> && <cmd>`; absolute paths everywhere.
- ffmpeg and ffprobe come from mise: wrap pipeline and extraction commands as `mise -C ~/projects/fukikae exec -- <cmd>`; subprocesses take argument lists.
- Source media is read-only and never committed. Transcripts, subtitle files, clip sets and the user-level ledger stay out of git; the only ledger content committed is Task 8's JSONL export beside the results doc.
- Media from the NAS media share arrives only through a read-only Docker NFS volume, removed after the copy, as the identity plan's Task 4 did (`<nas>` and `<media-export>` are the NAS and its media export):
  `docker volume create --driver local --opt type=nfs --opt o=addr=<nas>,ro,nfsvers=4 --opt device=:<media-export> fukikae-media-ro`, then `docker run --rm --user "$(id -u):$(id -g)" -v fukikae-media-ro:/media:ro -v ~/fukikae-ws/media:/out alpine:3.20 cp "<file under /media/TV>" "/out/<series>/"`, then `docker volume rm fukikae-media-ro`. Copy only the episodes this plan names.
- GPU work runs on nvidia-smi index 0 (`gpu_device = 0` in every workspace); index 1 is vLLM.
- Disk: `/` had 26 GB free at plan time (2026-11-08). Run `df -h /` before copying each set and keep 8 GB free; run the pipeline with `--to diarize` (the harness needs only `01-separate/vocals_16k.wav` and `04-diarize`); `uv cache prune` freed 40 GB once. Default-profile copies of workspaces use `cp -al` (artifacts are written by atomic rename, so a rewrite breaks the link instead of changing the shared file).
- Tests first, with RED evidence in the task report. Markers as in CLAUDE.md (`gpu`, `media`, `ffmpeg`, `slow`).
- Conventional Commits; no AI attribution, no `Co-Authored-By`, no session links.
- Subagents never start background watchers (`watch`, `tail -f`, polling loops); a long run is one foreground command or one background command that reports when it exits.
- The bar is unchanged: recurring means 5 s of labelled speech per episode compared (R6); pass when 90% of recurring characters keep 80% of their speech on one id and no two share a modal id; `?` and overlapped speech are excluded; coverage is reported.
- Nothing is chosen by looking at a held-out episode. Each profile's held-out score runs once, after its pick is committed (R18: a failing profile ships as measured, failure stated).
- Rulings carried: R1 (pipeline code never imports `eval`), R3 (profile tables name `backend`; `[characters]` and `[resolve]` name `matcher`), R7 (holdout pools the tuning registry), R8 (embedder metric), R10 (WeSpeaker for every genre), R12/R13/R17 (joint runs through `fukikae registry resolve` and its `[resolve]` table), R15 (Demucs), R19 (an embedder change needs `registry reembed`).
- Every keep/drop decision is a ruling with its numbers in the execution ledger `docs/superpowers/ledgers/2026-11-08-ground-truth.md`.

## Review Focus

1. **Tests writing to the operator's real ledger.** Any test that runs `eval asr` or `eval identity` through the CLI would append to `~/.local/share/fukikae/results.jsonl`. Expect an autouse fixture that points `FUKIKAE_RESULTS` at a temp file for every test. Pinned in Task 1.
2. **A truncated or foreign ledger line** (a process killed mid-append, a row from a newer schema): `eval report` skips it with a warning naming the line number and still renders. Pinned in Task 1.
3. **A default-profile run on a live-action show lands in the "default" genre column**, because genre defaults to the profile name. Expect an explicit `[series] genre` to win, and Task 7 sets it in every workspace. Pinned in Task 1.
4. **A transcript with a scene the broadcast cut, or subtitles with lines the transcript lacks:** labels must not bleed across the gap; unmatched cues stay unlabelled and the cues around the gap keep their speakers. Pinned in Task 4.
5. **A capture with a cut or a jump** (a different start, a dropped segment): a single linear fit can't map it. Expect `eval subalign` to refuse (inlier share or residual out of bounds) instead of writing a mis-timed file, unless `--force`. Pinned in Task 3.

---

## Decisions this plan makes

- **Ledger location: one user-level file**, `$XDG_DATA_HOME/fukikae/results.jsonl` (default `~/.local/share/fukikae/results.jsonl`), overridden by `FUKIKAE_RESULTS` and by `--ledger` on every command that reads or writes it. Each workspace is one series in one audio language, so a per-workspace file would hold one column of the matrix; the report would have to discover workspaces to show language against language, and eval workspaces get deleted under disk pressure (Grid's variants, `t9hold`). Each row records its workspace path, so nothing is lost by pooling. Results docs commit a filtered export (`eval report --export`), so their numbers stay reproducible from the repo.
- **Row key and cells.** A matrix row is (stage, test set, split, backend, model, options hash, tag); a column is `language/genre`; a cell is the latest row's headline metric (asr `cer`, identity `fraction` with clean and merge counts, embedder `nc_overall`, tts `speaker_similarity`). `--metric` picks another metric.
- **`[series]`**: `title` (default: the workspace directory name), `genre` (default: the profile name; lowercase `[a-z0-9-]+`), and `tmdb`, `tvdb`, `anidb` accepted now into `SeriesInfo.ids` and unused, so the cards plan adds no schema; any other key is a ConfigError naming it.
- **Transcript source for BBT: bigbangtrans.wordpress.com.** All 17 S01 pages exist and fetch without an account (checked 2026-11-08), one `Name: line` paragraph per utterance, `Scene:` lines and parenthesised directions easy to drop. KnowIT VQA's speaker-labelled subtitles need a request form, and TVQA's are timed to their own video cut, so both stay fallbacks; the reader's format dispatch leaves room for them and Friends Character Mining JSON later. Note the index lists an extra `series-1-episode-08-the-lizard-spock-expansion` page, which is S02E08; S01E08 is `series-1-episode-8-the-grasshopper-experiment`.
- **Colour map is series-wide and operator-visible.** NHK keeps a character's colour across a drama's run (RTVE reassigns per episode, which is why today's reader labels `<colour>@<episode>`). `fukikae eval colour-map` pools the named lines of all given episodes and writes `labels-colours.yaml` (colour: name) in the workspace beside `labels-aliases.yaml`; the readers use it when present. Building it from every episode's named lines, held-out included, is ground-truth construction, not tuning.
- **Unmapped colours are dropped from the bar by default.** NHK commonly gives one colour to several minor characters, so scoring an unmapped colour as one identity would count a correct separation as a split. `--colour-anonymous series` keeps them as `colour:<name>`, consistent across episodes, for within-episode purity diagnostics; the ledger row records the mode.
- **Clip-set split.** When every clip's file name carries an episode number (人民的名义: `丁义珍_人民的名义 01_vocals_260.wav`), each character trains on the smallest prefix of its episodes holding at least half its clips and tests on the rest, so no episode is on both sides (R8 trains on one episode and classifies another). Otherwise (神雕侠侣: `spk01/<emotion>/...`) the first half of each character's clips in sorted path order trains, which for 神雕侠侣 partitions by emotion folder; the row records the rule. A character with no clip on one side is not scored and is listed.
- **Clip-set rows are `split = "tune"`:** they inform embedder choice, so nothing in them is held out.

## Test sets and episode splits (fixed before any run)

| Set (workspace) | Language | Ground truth | Tuning (registry order) | Held-out |
|---|---|---|---|---|
| `bbt-en`, `bbt-de`, `bbt-es` | en, de, es | bigbangtrans transcripts aligned to the release's English subtitles (`--gt ass --gt-file ~/fukikae-ws/subs/bbt/labels`) | S01E01, S01E02 | S01E03 |
| `aib` (Alice in Borderland) | ja | Netflix Japanese SDH, stream 19 on S01E01 (`--gt sdh`) | S01E01, S01E02 | S01E03 |
| `kaze` (風、薫る) | ja | ARIB colour captions retimed to the WEB-DL audio (`--gt colour --gt-file ~/fukikae-ws/subs/kaze-kaoru/aligned`) | S01E01, S01E02, S01E03 (week 1) | S01E11 (week 3) |
| `grid` (existing) | ko | Korean CC (`--gt sdh --subtitle-track 2`) | S01E02, S01E04 | S01E05 |

- Alice in Borderland S01E01 to E03 keep Arisu, Karube, Chota and Shibuki speaking in all three (Chota and Shibuki leave in E03); E04 onwards loses most of that cast. E01 to E04, E06 and E07 are WEBDL; E05 and E08 are HDTV releases that may lack the Netflix tracks. If E02 or E03 lacks the Japanese SDH track, use S01E04 in its place and record it before any run.
- Kaze, Kaoru's episodes are 15 minutes, hence three tuning episodes. The operator added weeks 1-3 (E01-E15, "Scent of the Wind"). The weekly packs aren't in the library ("Kaze, Kaoru Week NN" doesn't parse as an episode), so Task 7 copies them from their folder on the same share through the same read-only volume. The colour map may use named lines from every episode, held-out included: that builds ground truth, not tuning (controller ruling). episodes are linked into `~/fukikae-ws/media/Kaze Kaoru/` as `Kaze Kaoru - S01E<N>.<ext>`, with `<N>` the broadcast number written the way `episode_id` prints it (at least two digits: `S01E05`, `S01E105`), and the aligned caption files take the same `S01E<N>` name, because `--gt-file` directories are matched by that code and `S01E005` would not match `S01E05`.
- Ledger set names: every identity command passes `--set bbt` in all three BBT workspaces (so en, de and es land as columns of one matrix row), and `--set aib`, `--set kaze`, `--set grid` elsewhere; the clip sets are `renmin` and `shendiao`.
- Grid's tuning pair joins the live-action sweep so Korean stays in the evidence; its held-out E05 was scored once for the provisional profile and is scored again only if the new pick differs from it.

## File Structure

```
src/fukikae/
  config.py                         # modify: SeriesInfo, Config.series, [series] validation
  eval/
    ledger.py                       # new: LedgerRow, path, append, read, options hash, commit, model identity
    report.py                       # new: filters, matrix, text/markdown rendering
    clipset.py                      # new: folder-per-character clip sets, split, nearest-centroid per embedder
    cli.py                          # modify: report, subalign, transcript-align, colour-map, embedders; asr appends rows
    identity/cli.py                 # modify: --set/--tag/--no-ledger, identity_rows(), profile in settings
    identity/metrics.py             # modify: clean_count()
    groundtruth/
      base.py                       # modify: strip invisible bidi marks; Cue.style
      sdh_prefix.py                 # modify: Japanese tags (ruby, separators, descriptions, short names, pair rule)
      colour.py                     # modify: series colour map, anonymous modes, ruby events dropped
      subalign.py                   # new: TimeMap, fit to reference subtitles or to speech, retime files
      transcript.py                 # new: transcript readers, text alignment, named ASS writer
      __init__.py                   # modify: read_ground_truth(colour_names=, anonymous=)
  profiles/live-action.toml         # modify (Task 8)
tests/
  conftest.py                       # modify: autouse FUKIKAE_RESULTS
  test_config.py                    # modify
  eval/test_ledger.py, eval/test_report.py, eval/test_clipset.py              # new
  eval/test_groundtruth.py, eval/test_subalign.py, eval/test_transcript.py    # modify / new
  eval/identity/test_cli.py         # modify
docs/results/2026-11-xx-live-action-ground-truth.md (+ .jsonl export)          # new (Task 8)
```

## Media dependencies

Tasks 1 to 6 need no library media: Task 2's real-file check uses the Alice in Borderland S01E01 subtitle already extracted at `<scratch>/lib-probe/subs/Alice_in_Borderland__S01E01_t19_jpn.srt`, and Task 6 downloads its clip set from Hugging Face. Task 7 needs BBT S01E01 to E03 (the operator is adding it), Kaze, Kaoru's packs and captions (an agent is fetching them) and Alice in Borderland S01E01 to E03 (in the library). Task 8 needs Task 7. Real-file checks for Tasks 3, 4 and 5 run inside Task 7, so code review never waits on media.

---

### Task 1: Results ledger, `[series]` and `eval report`

**Files:**
- Create: `src/fukikae/eval/ledger.py`, `src/fukikae/eval/report.py`
- Modify: `src/fukikae/config.py`, `src/fukikae/eval/cli.py` (asr flags, `report` command), `src/fukikae/eval/identity/cli.py` (flags, `identity_rows`, `profile` in settings), `src/fukikae/eval/identity/metrics.py` (`clean_count`), `tests/conftest.py`, `README.md` (ledger path and `eval report`)
- Test: `tests/eval/test_ledger.py`, `tests/eval/test_report.py`, `tests/test_config.py`, `tests/eval/identity/test_cli.py`

**Interfaces:**
- Produces (`fukikae.config`): `SeriesInfo(title: str, genre: str, ids: dict[str, str | int])` (frozen); `Config.series: SeriesInfo` (default `SeriesInfo("", "default", {})`); `load_config` fills it.
- Produces (`fukikae.eval.ledger`):
  - `SCHEMA = 1`, `STAGES = ("asr", "identity", "embedder", "tts")`, `SPLITS = ("tune", "holdout")`
  - `LedgerRow` (frozen dataclass): `schema: int, run_id: str, date: str, commit: str, stage: str, backend: str, model: str | None, revision: str | None, options_hash: str, options: dict[str, Any], language: str, genre: str, series: str, test_set: str, split: str, episodes: list[str], metrics: dict[str, float | int | str], detail: dict[str, Any], workspace: str | None, tag: str | None`
  - `ledger_path(override: Path | None = None) -> Path`
  - `options_hash(options: Mapping[str, Any]) -> str`: first 12 hex of sha256 of `json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)`
  - `git_commit() -> str`: `git -C <package dir> rev-parse --short=12 HEAD`, plus `-dirty` when `git status --porcelain` is non-empty, `"unknown"` outside a checkout
  - `model_identity(kind: str, cfg: BackendConfig) -> tuple[str | None, str | None]`: model from an unloaded instance's `model` attribute (`backend_class(kind, name)(cfg.options, "cpu")`), else `cfg.options.get("model")`; revision from `cfg.options.get("revision")`, else the commit in `<HF hub cache>/models--<org>--<name>/refs/main` (honouring `HF_HUB_CACHE`, `HF_HOME`), else None; any exception gives `(cfg.options.get("model"), None)`
  - `make_row(*, stage, backend, model, revision, options, language, genre, series, test_set, split, episodes, metrics, detail=None, workspace=None, tag=None) -> LedgerRow`: fills `run_id` (uuid4 hex), `date` (UTC ISO 8601, seconds), `commit`, `options_hash`; ValueError on an unknown stage or split
  - `append_rows(rows: Sequence[LedgerRow], path: Path | None = None) -> Path`: creates parents, holds `fcntl.flock(LOCK_EX)`, writes all rows in one `write`
  - `read_rows(path: Path | None = None) -> tuple[list[LedgerRow], list[str]]`: rows plus warnings (`results.jsonl:7: not JSON, skipped`, `...: schema 2, skipped`)
- Produces (`fukikae.eval.report`): `HEADLINE = {"asr": "cer", "identity": "fraction", "embedder": "nc_overall", "tts": "speaker_similarity"}`; `filter_rows(rows, *, stages=(), backends=(), sets=(), splits=(), tags=()) -> list[LedgerRow]`; `Matrix(columns: list[str], rows: list[tuple[str, list[str]]])`; `build_matrix(rows, metric: str | None = None) -> Matrix`; `render(m: Matrix, fmt: Literal["text", "md"] = "text") -> str`.
- Produces (`fukikae.eval.identity`): `metrics.clean_count(s: IdentityScore) -> int` (passing characters in no merge); `cli.identity_rows(run: IdentityRun, cfg: Config, test_set: str, tag: str | None) -> list[LedgerRow]`: one `tune` row, plus a `holdout` row when the run has one. `backend` is the embedder backend, `model`/`revision` from `model_identity("embed", ...)`, `options` are the run's settings without `workspace`, `tune` and `holdout`, plus `profile` and `profile_hash` (`options_hash(cfg.profile_raw)`); metrics `fraction, passing, recurring, clean, merges, strict_fraction, weighted_share, coverage, verdict, nc_overall`; `detail = {"merges": [...], "nc": {...}}`.
- CLI: `eval asr` and `eval identity` gain `--set TEXT` (default: workspace directory name), `--tag TEXT`, `--no-ledger`, `--ledger PATH`; `eval asr` also gains `--split tune|holdout` (default tune). `fukikae eval report [--ledger PATH] [--stage S]... [--backend B]... [--set NAME]... [--split S]... [--tag T]... [--metric M] [--format text|md] [--export PATH]`; `--export` writes the filtered rows as JSONL.

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("FUKIKAE_RESULTS", str(tmp_path_factory.mktemp("ledger") / "results.jsonl"))

# tests/eval/test_ledger.py
def test_ledger_path_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("FUKIKAE_RESULTS", str(tmp_path / "env.jsonl"))
    assert ledger_path(tmp_path / "flag.jsonl") == tmp_path / "flag.jsonl"
    assert ledger_path() == tmp_path / "env.jsonl"
    monkeypatch.delenv("FUKIKAE_RESULTS")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert ledger_path() == tmp_path / "xdg" / "fukikae" / "results.jsonl"

def test_tests_never_touch_the_real_ledger():
    assert not str(ledger_path()).startswith(str(Path.home() / ".local"))

def test_options_hash_ignores_key_order():
    assert options_hash({"a": 1, "b": {"c": 2}}) == options_hash({"b": {"c": 2}, "a": 1})
    assert len(options_hash({})) == 12

def test_append_then_read_round_trips(tmp_path):
    row = make_row(stage="identity", backend="wespeaker", model="m", revision=None,
                   options={"threshold": 0.6}, language="ja", genre="live-action",
                   series="Alice", test_set="aib", split="tune", episodes=["S01E01", "S01E02"],
                   metrics={"fraction": 0.5})
    append_rows([row], tmp_path / "r.jsonl")
    rows, warnings = read_rows(tmp_path / "r.jsonl")
    assert rows == [row] and warnings == []

def test_make_row_rejects_unknown_stage_and_split(): ...   # ValueError for stage="diarize", split="dev"

def test_read_skips_truncated_and_future_rows(tmp_path):
    # a good row, then '{"schema": 1, "stage": "iden' (no newline), then a schema-2 row
    rows, warnings = read_rows(p)
    assert len(rows) == 1 and len(warnings) == 2 and ":2:" in warnings[0]

def test_git_commit_outside_a_checkout_is_unknown(monkeypatch): ...  # patched package dir -> "unknown"

# tests/test_config.py
def test_series_defaults_to_dir_name_and_profile(tmp_path):
    ws = tmp_path / "bbt-de"; ws.mkdir(); (ws / "fukikae.toml").write_text('profile = "live-action"\n')
    s = load_config(ws, user_config=None).series
    assert (s.title, s.genre, s.ids) == ("bbt-de", "live-action", {})

def test_explicit_genre_wins_over_profile(tmp_path):   # profile default + [series] genre = "live-action"
    ...  # series.genre == "live-action", not "default"

def test_series_accepts_external_ids_and_rejects_unknown_keys(tmp_path):
    ...  # tmdb = 1418 -> ids == {"tmdb": 1418}; imdb = "x" -> ConfigError mentioning "imdb"

def test_series_genre_must_be_lowercase_slug(tmp_path): ...  # genre = "Live Action" -> ConfigError

# tests/eval/test_report.py
def test_matrix_columns_by_language_and_genre(): ...
    # rows: identity bbt tune en/live-action 0.6, de/live-action 0.4 -> columns
    # ["de/live-action", "en/live-action"], one matrix row with both cells
def test_latest_row_wins_per_key(): ...          # same key, two dates -> the later value
def test_filters_stage_backend_set_split_tag(): ...
def test_identity_cell_shows_clean_and_merges(): ...   # "0.56 c5 m1"
def test_render_markdown_is_a_table(): ...       # header row and |---| separator
def test_report_cli_warns_on_bad_lines_and_still_renders(tmp_path): ...  # CliRunner, exit 0, warning on stderr

# tests/eval/identity/test_cli.py
def test_identity_rows_tune_and_holdout(): ...
    # a fake IdentityRun with tune and holdout scores -> 2 rows, splits ("tune","holdout"),
    # episodes ["S01E01","S01E02"] and ["S01E03"], metrics["clean"] == clean_count(score),
    # options has "profile" and no "workspace"
def test_identity_cli_appends_unless_no_ledger(tmp_path, monkeypatch): ...
    # run_identity patched to return the fake run; one CLI call appends 2 rows, --no-ledger appends 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_ledger.py tests/eval/test_report.py tests/test_config.py tests/eval/identity/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: fukikae.eval.ledger` and missing `Config.series`.

- [ ] **Step 3: Implement** `SeriesInfo` and `[series]` loading in `config.py`; `ledger.py` and `report.py` per the Interfaces block; `clean_count` in `metrics.py`; `identity_rows` and the new flags in `identity/cli.py` (append after printing, so a ledger error never hides a result; a ledger write failure prints `warning: ledger not written: ...` and exits 0); `eval asr` rows (`backend` the ASR backend, `model_identity("asr", cfg.resolve("asr", language))` when the resolved backend matches, metrics `cer`, `runtime_s` or absent when cached); the `report` command. Report row labels read `stage | set | split | backend model | opts:<hash[:8]> | tag`. README: one paragraph on the ledger path, `FUKIKAE_RESULTS`, and `eval report`.

- [ ] **Step 4: Run the tests to verify they pass, then the full suite**

Run: `uv run --directory ~/projects/fukikae pytest -q && uv run --directory ~/projects/fukikae ruff check && uv run --directory ~/projects/fukikae ruff format --check && uv run --directory ~/projects/fukikae pyright`
Expected: all pass; no file under `~/.local/share/fukikae/` created by the run.

- [ ] **Step 5: Commit**

```bash
git -C ~/projects/fukikae add -A src tests README.md
git -C ~/projects/fukikae commit -m "feat(eval): results ledger, [series] genre and eval report"
```

### Task 2: Japanese SDH speaker prefixes

Today the SDH reader finds zero speakers in Alice in Borderland S01E01's Japanese SDH track: every cue starts with U+202A (LEFT-TO-RIGHT EMBEDDING), so `_BRACKET` never matches at the line start. With those marks removed by hand it finds 195 lines, led by カルベ 54, アリス 44, チョータ 43, シブキ 18, but also 通知音 14 (a notification sound), 男性 2, ３人 1 and アリスたち 1, and it misses the full-name tags whose ruby breaks the bracket match (`（‪有栖(ありす)‪ ‪始(はじめ)‪）`). Override blocks such as `{\fscx50}` are already removed (`_ass_text` for ASS, `strip_markup` for text formats); this task pins that with a test rather than adding code for it.

**Files:**
- Modify: `src/fukikae/eval/groundtruth/base.py` (`strip_markup` also removes U+200B to U+200F, U+202A to U+202E, U+2066 to U+2069, U+FEFF), `src/fukikae/eval/groundtruth/sdh_prefix.py`
- Test: `tests/eval/test_groundtruth.py`

**Interfaces:**
- Produces (`sdh_prefix`): `strip_ruby(text: str) -> str` removes a `(かな)` or `（かな）` run of only hiragana, katakana and `ー` that directly follows a kanji; `extract_tag` applies `strip_markup` then `strip_ruby` before matching. `ja_short_names(names: Iterable[str]) -> dict[str, str]` maps a two-word CJK tag to its last word when that word is also a tag on its own in the same file; `read_sdh` applies it before the `Namer`.
- Japanese rules inside `valid_name` / `_is_description`, for tags with CJK characters:
  - several speakers: the tag contains `・`, `･`, `、` or `／` (`アリス･チョータ`);
  - description: in `DEFAULT_SOUNDS`, which gains 音楽, 拍手, 足音, 物音, 爆音, 雑音, 銃声, 歓声, 罵声, 悲鳴, 吐息, ため息, 笑い, クラクション, ノック, アナウンス, 男性, 女性, 一同, 全員; or 3+ characters ending in 音, 声, 息 or 鳴 (`通知音`, `荒い息`, `ふざけ合う声`; `琴音` stays a name); or a group (`[0-9０-９]+人$`, or ending in たち or 達);
  - `の` alone is not a description (`チョータの母` is a speaker);
  - the CJK branch of `_PAIR` needs 2+ characters on each side of と/和/与/跟 (`さとみ` is a name, `アリスとカルベ` a pair).

- [ ] **Step 1: Write the failing tests** (strings copied from the real file, U+202A included)

```python
LRE = "‪"

def test_netflix_bidi_marks_do_not_hide_tags(tmp_path):
    srt = f"1\n00:01:04,564 --> 00:01:06,941\n{LRE}（始）また行ってねえのか 面接\n"
    assert [ln.speaker for ln in read_ground_truth(write(tmp_path, srt), "sdh", None)] == ["始"]

def test_halfwidth_parens_and_tag_on_its_own_line(tmp_path):
    # "(カルベ) おい" -> カルベ; "（始）\n俺も会社に頭 下げてやったのに" -> 始 over the whole cue

def test_ruby_inside_tag_and_full_name_maps_to_given_name(tmp_path):
    # cues "（‪有栖(ありす)‪ ‪始(はじめ)‪）\n‪ホント クズ兄貴だな" and "（始）また行ってねえのか"
    assert speakers == ["始", "始"]

def test_full_name_without_short_form_is_kept(tmp_path):
    # only "（有栖‪良平(りょうへい)‪）あっ！": no space between the names, so ruby removal gives
    # one word, "有栖良平", kept as written (Task 7 aliases it to アリス)

@pytest.mark.parametrize("tag", ["（通知音）", "（合成音声）", "（荒い息）", "（ふざけ合う声）",
                                 "（女性たちの悲鳴）", "（男性）", "（３人）", "（アリスたち）",
                                 "（クラクション）", "（足音）", "（アリス･チョータ）"])
def test_japanese_descriptions_groups_and_pairs_are_not_speakers(tag): ...  # extract_tag(tag + "text") is None

@pytest.mark.parametrize("tag,name", [("（チョータの母）", "チョータの母"), ("（琴音）", "琴音"),
                                      ("（さとみ）", "さとみ"), ("（シブキ）", "シブキ")])
def test_japanese_names_still_accepted(tag, name): ...

def test_ass_override_blocks_split_parentheses(tmp_path):
    # ASS Dialogue text "{\fscx50}（{\fscx100}薫{\fscx50}）{\fscx100}どうして？" -> 薫

def test_existing_pair_rule_still_rejects_korean_and_chinese_pairs(): ...  # "(새하와 새벽)", "(张三和李四)"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_groundtruth.py -q`
Expected: FAIL on the bidi, ruby, description and short-name tests; the override-block test may already pass (state which in the report).

- [ ] **Step 3: Implement** the bidi stripping, `strip_ruby`, the Japanese rules and `ja_short_names` per the Interfaces block. Keep every existing test green (R5's Hangul clauses untouched).

- [ ] **Step 4: Run the tests, the full suite, and the real file**

Run: `uv run --directory ~/projects/fukikae pytest -q`, then count speakers in `<scratch>/lib-probe/subs/Alice_in_Borderland__S01E01_t19_jpn.srt` with `read_ground_truth(path, "sdh", None)` through `uv run --directory ~/projects/fukikae python -c ...`.
Expected: tests pass; the real file yields カルベ, アリス, チョータ, シブキ, ミナミ, 始, チョータの母, カズ, エミ and 有栖良平 (or its alias target), and no 通知音, 合成音声, 男性, ３人, アリスたち or アリス･チョータ. Record the counts and the share of cues that carry a name in the task report.

- [ ] **Step 5: Commit** `feat(eval): Japanese SDH speaker tags (bidi marks, ruby, descriptions)`

### Task 3: Subtitle-to-audio alignment (`eval subalign`)

**Files:**
- Create: `src/fukikae/eval/groundtruth/subalign.py`
- Modify: `src/fukikae/eval/cli.py`
- Test: `tests/eval/test_subalign.py`

**Interfaces:**
- Consumes: `Cue`, `read_cues`, `strip_markup`, `AlignmentError` from `groundtruth.base`; `normalize` from `eval.asr`; `speech_turns(work: Path) -> list[Turn]` from `eval.autolabel`.
- Produces:
  - `TimeMap(scale: float, offset: float)` (frozen), `apply(t: float) -> float` = `scale * t + offset`
  - `FitReport(tmap: TimeMap, mode: Literal["text", "speech"], anchors: int, inliers: int, rms_s: float, share: float | None)`
  - `fit_to_reference(labelled: list[Cue], reference: list[Cue], language: str) -> FitReport`
  - `fit_to_speech(labelled: list[Cue], speech: list[Turn]) -> FitReport`
  - `check_fit(rep: FitReport, force: bool = False) -> None` (raises `AlignmentError`)
  - `retime_file(src: Path, dst: Path, tmap: TimeMap) -> tuple[int, int]` (cues rewritten, cues clamped at 0)
- Constants: `ANCHOR_MIN_RATIO = 0.8`, `INLIER_TOL_S = 0.5`, `MAX_SCALE_DEV = 0.05`, `MIN_ANCHORS = 20`, `MIN_INLIER_SHARE = 0.6`, `MAX_RMS_S = 0.3`, `SPEECH_SCALES = (1.0, 1000/1001, 1001/1000, 25/23.976, 23.976/25, 24/23.976, 23.976/24)`, speech offset search ±300 s at 0.5 s, refined ±1 s at 0.02 s; speech mode passes at `MIN_SHARE` (0.6) from `base`.
- CLI: `fukikae eval subalign LABELLED (--reference REF | --speech WORKSPACE --episode SxxEyy) --out OUT [--language L] [--force]`; prints mode, scale, offset, anchors/inliers and rms (text) or share (speech), and the cue counts.

Text mode: normalized cue text (`normalize(strip_markup(text), language)`, at least 4 characters for CJK or 2 words otherwise) is matched exact-first, then by `difflib.get_close_matches(n=2, cutoff=ANCHOR_MIN_RATIO)`; an anchor needs a unique best match (the runner-up at least 0.1 lower). Fit by RANSAC over anchor onset pairs at least 60 s apart (`numpy.random.default_rng(0)`, 2,000 draws, scale within `1 ± MAX_SCALE_DEV`), then least squares on the inliers. Speech mode scores each (scale, offset) by the share of cue time over speech, the measure `base.alignment` uses, generalised to a scale. Retiming rewrites only timestamps: ASS/SSA `Dialogue` Start and End, SRT and WebVTT `-->` lines; every other byte is kept.

- [ ] **Step 1: Write the failing tests**

```python
def test_text_fit_recovers_offset_and_drift():
    # 120 labelled cues with distinct texts; reference = same texts at 1.001 * t + 12.5
    rep = fit_to_reference(lab, ref, "en")
    assert abs(rep.tmap.scale - 1.001) < 1e-4 and abs(rep.tmap.offset - 12.5) < 0.05
    assert rep.inliers >= 110

def test_text_fit_tolerates_rewrapped_lines_and_missing_cues(): ...
    # reference joins pairs of cues and drops 20% -> still recovered

def test_cut_capture_refuses():
    # second half of the reference shifted a further +30 s -> check_fit raises AlignmentError
    # naming the inlier share; check_fit(rep, force=True) returns

def test_too_few_anchors_refuses(): ...      # 10 matching cues -> AlignmentError

def test_speech_fit_recovers_frame_rate_drift():
    # turns at (25/23.976) * t + 3.0 over the labelled spans
    rep = fit_to_speech(lab, turns)
    assert rep.tmap.scale == pytest.approx(25 / 23.976) and abs(rep.tmap.offset - 3.0) < 0.05

def test_retime_ass_keeps_everything_but_times(tmp_path): ...
    # styles, Name field, override blocks and text byte-identical; times = scale*t+offset, centiseconds
def test_retime_srt_and_clamps_negative(tmp_path): ...   # offset -5 on a cue at 2 s -> 00:00:00,000, clamped count 1
def test_cli_writes_retimed_file_and_refuses_bad_fit(tmp_path): ...  # CliRunner: exit 0 and file; exit 1 without --force
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_subalign.py -q`
Expected: FAIL with `ModuleNotFoundError: fukikae.eval.groundtruth.subalign`.

- [ ] **Step 3: Implement** per the Interfaces block and the paragraph above.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --directory ~/projects/fukikae pytest -q`
Expected: PASS. (The real Kaze, Kaoru pair is checked in Task 7.)

- [ ] **Step 5: Commit** `feat(eval): subalign retimes subtitles to a reference track or the speech`

### Task 4: Transcript-to-subtitle aligner (`eval transcript-align`)

**Files:**
- Create: `src/fukikae/eval/groundtruth/transcript.py`
- Modify: `src/fukikae/eval/cli.py`
- Test: `tests/eval/test_transcript.py`

**Interfaces:**
- Consumes: `Cue`, `read_cues`, `strip_markup`, `split_times`, `Namer` from `groundtruth.base`; `cue_segments` from `sdh_prefix`; `normalize` from `eval.asr`; `gt_path_for` from `eval.autolabel`.
- Produces:
  - `TranscriptLine(speaker: str, text: str)`; `read_transcript(path: Path) -> list[TranscriptLine]`: `.html`/`.htm` take each `<p>`'s text (`html.parser`, entities unescaped); `.txt` takes lines. A line is `^([^:]{1,40}):\s*(.+)$`; speakers in `TRANSCRIPT_SKIP = {"scene", "credits sequence", "teleplay", "story", "written by"}` are dropped, as are lines wholly in parentheses; parentheticals are removed from both the speaker (`Sheldon (on phone)` becomes `Sheldon`) and the text. Other formats raise ValueError naming the suffix.
  - `NamedLine(start: float, end: float, speaker: str, text: str)`
  - `AlignResult(named: list[NamedLine], cue_share: float, word_share: float, speakers: dict[str, int])`
  - `align_transcript(lines: list[TranscriptLine], cues: list[Cue], namer: Namer, language: str = "en") -> AlignResult`
  - `write_named_ass(path: Path, lines: list[NamedLine]) -> None`: a minimal ASS (`[Script Info]`, one `Default` style, `[Events]` with the standard Format line), `Name` = speaker, Start and End at centiseconds
- Constants: `MIN_MATCHED = 0.5`, `MIN_MAJORITY = 0.6`; extra speaker names never used: `{"all", "both", "everyone", "together"}` (and `_MULTI` in `base` already drops `Leonard and Sheldon`).
- CLI: `fukikae eval transcript-align --transcripts DIR|FILE --subtitles DIR|FILE --out DIR --episodes S01E01,... [--alias FROM=TO]... [--language en]`; files paired by SxxEyy in the name; writes `<out>/<SxxEyy>.ass`, prints per episode cue share, word share and speaker counts. Aliases also come from `labels-aliases.yaml` in `--out`'s parent when present.

Alignment: subtitle cues split into speaker segments by `cue_segments` (dash markers) with times from `split_times`; both sides tokenized as `normalize(text, language).split()` (CJK: characters); `difflib.SequenceMatcher(None, sub_tokens, transcript_tokens, autojunk=False).get_matching_blocks()` gives each matched subtitle token the speaker of its transcript token. A segment is labelled when at least `MIN_MATCHED` of its tokens matched and one speaker holds at least `MIN_MAJORITY` of those; speakers go through the `Namer`.

- [ ] **Step 1: Write the failing tests**

```python
BIGBANG_HTML = """<div class="entrytext"><p>Scene: A corridor at a sperm bank.</p>
<p>Sheldon: So if a photon is directed through a plane with two slits in it&#8230;</p>
<p>Leonard: Agreed, what&#8217;s your point?</p><p>(They sit and begin to fill in forms).</p>
<p>Receptionist (off): Hang on.</p></div>"""

def test_read_bigbangtrans_html(tmp_path):
    assert [l.speaker for l in read_transcript(write(tmp_path, "S01E01.html", BIGBANG_HTML))] == \
        ["Sheldon", "Leonard", "Receptionist"]

def test_align_assigns_speakers_across_rewrapped_cues(): ...
    # 3 transcript lines; subtitle cues break them at other points -> each cue's speaker correct

def test_dash_cue_with_two_speakers_is_split(): ...
    # cue "- What's your point?\n- There's no point." -> two NamedLines, Leonard then Sheldon, times split

def test_scene_missing_from_subtitles_does_not_bleed():
    # transcript has an extra 6-line scene between two matched scenes; subtitles lack it
    # -> the cues before and after keep their own speakers; no NamedLine carries the cut scene's speaker

def test_subtitle_lines_missing_from_transcript_stay_unlabelled(): ...  # cue text absent -> no NamedLine for it

def test_group_speakers_are_dropped(): ...   # "All: Hello!" -> unlabelled

def test_named_ass_round_trips_through_the_harness_reader(tmp_path):
    write_named_ass(p, [NamedLine(1.0, 2.5, "Penny", "Hi.")])
    assert read_ground_truth(p, "ass", None) == [GroundTruthLine(1.0, 2.5, "Penny")]

def test_cli_pairs_files_by_episode_code(tmp_path): ...  # S01E01.html + S01E01.srt -> out/S01E01.ass
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_transcript.py -q`
Expected: FAIL with `ModuleNotFoundError: fukikae.eval.groundtruth.transcript`.

- [ ] **Step 3: Implement** per the Interfaces block.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --directory ~/projects/fukikae pytest -q`
Expected: PASS. (The 17 real pages against the release's subtitles are checked in Task 7.)

- [ ] **Step 5: Commit** `feat(eval): transcript-align names subtitle cues from untimed transcripts`

### Task 5: Broadcast colour labels with a series colour map

**Files:**
- Modify: `src/fukikae/eval/groundtruth/base.py` (`Cue.style: str = ""`, filled by `parse_ass_cues`), `src/fukikae/eval/groundtruth/colour.py`, `src/fukikae/eval/groundtruth/__init__.py`, `src/fukikae/eval/autolabel.py` and `src/fukikae/eval/identity/cli.py` (read `labels-colours.yaml`, `--colour-anonymous`), `src/fukikae/eval/cli.py` (`colour-map` command)
- Test: `tests/eval/test_groundtruth.py`

**Interfaces:**
- Produces (`colour`):
  - `RUBY_STYLE = re.compile(r"rub[iy]|ルビ", re.IGNORECASE)`; `read_colour` skips cues whose `style` matches it (ARIB converters emit furigana as their own small events)
  - `colour_counts(cues_by_episode: Mapping[str, list[Cue]], namer: Namer, sounds: frozenset[str] = DEFAULT_SOUNDS) -> dict[str, dict[str, Counter[str]]]`: episode, then colour, then tagged-name counts, from the same pieces `read_colour` builds
  - `propose_colour_map(counts, min_tagged: int = MIN_TAGGED_LINES, min_agreement: float = 0.8) -> tuple[dict[str, str], list[str]]`: a colour maps to a name holding at least `min_tagged` of its pooled tagged lines and at least `min_agreement` of them; a colour whose per-episode majority names differ is left out and reported
  - `load_colour_map(path: Path) -> dict[str, str]`, `write_colour_map(path: Path, mapping: Mapping[str, str]) -> None` (YAML `colour: name`, colour names from `COLOURS`, others a ValueError)
  - `read_colour(cues, episode, namer, sounds=DEFAULT_SOUNDS, *, colour_names: Mapping[str, str] | None = None, anonymous: Literal["episode", "series", "drop"] | None = None) -> list[GroundTruthLine]`: with no map, today's behaviour (per-episode co-occurrence, `<colour>@<episode>`); with a map, mapped colours take the map's name, and unmapped colours are dropped (`anonymous` None or "drop"), labelled `colour:<name>` ("series") or `<colour>@<episode>` ("episode")
- `read_ground_truth(..., colour_names=None, anonymous=None)` passes both through.
- Harness and auto-label: with `--gt colour` / `--source colour`, `labels-colours.yaml` in the workspace is loaded when present; new option `--colour-anonymous drop|series|episode` (default drop when a map exists); the identity ledger row's options gain `colour_map` (the mapping) and `colour_anonymous`.
- CLI: `fukikae eval colour-map WORKSPACE --episodes ... --gt-file DIR [--write]`: per colour, per-episode tagged-name counts, the proposed map and conflicts; `--write` writes `WORKSPACE/labels-colours.yaml`, refusing to overwrite an existing file without `--force`.

- [ ] **Step 1: Write the failing tests** (ARIB-shaped ASS fixtures; `\c&H00FFFF&` is yellow, `&HFFFF00&` cyan)

```python
def test_series_map_pools_named_lines_across_episodes(tmp_path):
    # E01: 2 yellow lines tagged （りん）; E02: 2 more -> pooled 4 >= 3 -> {"yellow": "りん"}
    # per episode alone neither reaches 3 (today's reader would say yellow@S01E01)

def test_conflicting_colour_is_left_out_and_reported(): ...
    # cyan tagged 直美 in E01 and 虎太郎 in E02 -> not in the map, one conflict line naming both

def test_unmapped_colour_dropped_by_default_and_kept_for_diagnostics(): ...
    # green never tagged: drop -> no green lines; series -> "colour:green" in both episodes, same label

def test_map_overrides_per_episode_cooccurrence(): ...
    # map yellow -> りん while E03 has 3 yellow lines tagged someone else -> lines say りん; the tag lines keep their tag

def test_fscx_blocks_and_colour_in_one_override(): ...
    # "{\c&H00FFFF&\fscx50}（{\fscx100}りん{\fscx50}）{\fscx100}行きます" -> yellow piece tagged りん

def test_ruby_style_events_are_ignored(): ...   # Style "Rubi" event over a yellow line adds no piece and no time split

def test_colour_map_cli_writes_and_refuses_overwrite(tmp_path): ...

def test_existing_colour_behaviour_unchanged_without_map(): ...  # current VTT/EBU-TT/ASS tests stay green
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_groundtruth.py -q -k colour`
Expected: FAIL on the map, anonymous and ruby tests.

- [ ] **Step 3: Implement** per the Interfaces block, inside `colour.py` (the piece-building loop is shared by `read_colour` and `colour_counts`, not copied).

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --directory ~/projects/fukikae pytest -q`
Expected: PASS. (The real ARIB captions are checked in Task 7; if their furigana events carry no ruby style, Task 7 extends the rule with the evidence.)

- [ ] **Step 5: Commit** `feat(eval): series colour maps for broadcast colour captions`

### Task 6: Clip-set embedder check (`eval embedders`)

**Files:**
- Create: `src/fukikae/eval/clipset.py`
- Modify: `src/fukikae/eval/cli.py`
- Test: `tests/eval/test_clipset.py`

**Interfaces:**
- Consumes: `Segment` from `eval.identity.segments`; `nearest_centroid_accuracy` and `OVERALL` from `eval.identity.metrics`; `ResidentEmbedder` from `eval.identity.cache`; `run_ffmpeg` from `media`; Task 1's `make_row`, `append_rows`, `model_identity`.
- Produces:
  - `AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}`; `DEFAULT_EPISODE_PATTERN = r"(?<=\s)(\d{1,3})(?=_)"`
  - `Clip(character: str, path: Path, episode: int | None)`
  - `scan_clipset(root: Path, episode_pattern: str = DEFAULT_EPISODE_PATTERN, exclude: Collection[str] = ()) -> list[Clip]`: each top-level directory is a character; audio found recursively; hidden directories (`.fukikae-cache`) skipped
  - `split_clips(clips: list[Clip]) -> tuple[list[Clip], list[Clip], str, list[str]]`: train, test, rule (`"episode"` or `"half"`), characters not scorable
  - `prepare_audio(clips: list[Clip], cache: Path) -> dict[Path, tuple[Path, float]]`: 16 kHz mono wav per clip under `cache/16k/`, keyed by sha256 of (relative path, size, mtime_ns), with its duration
  - `embed_clips(clips, audio: dict[Path, tuple[Path, float]], emb_cfg: BackendConfig, embed: Callable[[BackendConfig, Path, list[tuple[float, float]]], np.ndarray], cache: Path) -> np.ndarray`: one row per clip, span `(0, duration)`, cached as `cache/emb-<sha>.npy` keyed by `[emb_cfg.backend, emb_cfg.options]` and the clip keys
  - `clipset_accuracy(train, test, train_embs, test_embs) -> dict[str, float]`: builds `Segment(episode="train"|"test", start=0.0, end=duration, label=character)` and calls `nearest_centroid_accuracy`
- CLI: `fukikae eval embedders CLIPSET --embedder NAME... --language L [--genre G (default live-action)] [--set NAME (default: directory name)] [--exclude CHARACTER]... [--tag T] [--no-ledger] [--ledger PATH] [--no-wait]`; `--embedder` repeats; each runs with its backend's defaults (no per-embedder options in this plan, so `options` is `{}` apart from `split_rule`). Prints, per embedder, overall accuracy, characters scored, train and test clips and the split rule; one ledger row per embedder (`stage="embedder"`, `split="tune"`, `episodes=[]`, metrics `nc_overall, characters, train_clips, test_clips`, `detail` per-character accuracy, options include `split_rule`).

- [ ] **Step 1: Write the failing tests** (fixture: generated wav files in a tmp tree; a fake embed function returning a fixed vector per character plus noise)

```python
def test_scan_takes_top_level_dirs_as_characters_recursively(tmp_path): ...
    # renmin/丁义珍/丁义珍_人民的名义 01_vocals_260.wav, shendiao/spk01/happy/x.wav -> characters, episodes 1 / None

def test_episode_split_keeps_episodes_on_one_side(): ...
    # character with clips in eps 1,1,2,3,3,3 -> train eps {1,2} (3 of 6 clips), test ep {3}
def test_half_split_when_any_clip_lacks_an_episode(): ...   # rule "half", sorted-path halves
def test_single_episode_character_is_not_scored(): ...      # listed in the 4th return value

def test_accuracy_is_one_for_separable_fake_embeddings(): ...   # clipset_accuracy -> OVERALL == 1.0
def test_embedding_cache_hit_skips_the_embedder(tmp_path): ...  # second call: embed not called
def test_cli_writes_one_row_per_embedder(tmp_path, monkeypatch): ...
    # ResidentEmbedder patched; two --embedder flags -> 2 rows, stage "embedder", split "tune"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --directory ~/projects/fukikae pytest tests/eval/test_clipset.py -q`
Expected: FAIL with `ModuleNotFoundError: fukikae.eval.clipset`.

- [ ] **Step 3: Implement** per the Interfaces block (`ResidentEmbedder` keeps one embedder loaded and switches between them; ffmpeg via `run_ffmpeg`).

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --directory ~/projects/fukikae pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit** `feat(eval): eval embedders scores nearest-centroid accuracy on clip sets`

- [ ] **Step 6: Download the clip set** (pinned revision; two sets only)

Run: `uv run --directory ~/projects/fukikae huggingface-cli download noncegeek/cantonese-drama-voice-fine-grained --repo-type dataset --revision 88fa3c11a57a4bb72703d49912ce266bdb17b63b --include "人民的名义/*" --include "神雕侠侣_1995版/*" --local-dir ~/fukikae-ws/clips/cantonese-drama-voice`
Expected: 23 character folders under `人民的名义/` (2,410 clips) and `spk01` to `spk05` under `神雕侠侣_1995版/`. Record counts.

- [ ] **Step 7: Run the embedders** (GPU index 0; `ecapa`, `wespeaker`, `eres2netv2`, `titanet`, `campplus`, the names `fukikae backends list` reports)

Run: `mise -C ~/projects/fukikae exec -- uv run --directory ~/projects/fukikae fukikae eval embedders "~/fukikae-ws/clips/cantonese-drama-voice/人民的名义" --embedder ecapa --embedder wespeaker --embedder eres2netv2 --embedder titanet --embedder campplus --language zh --set renmin --tag t6`, then the same on `神雕侠侣_1995版` with `--language yue --set shendiao`.
Expected: 10 ledger rows; `fukikae eval report --stage embedder` shows both sets. Ruling in the execution ledger: if another embedder beats WeSpeaker by at least 5 points of `nc_overall` on both sets, record it as a proposal for the operator (R10 keeps WeSpeaker in this plan; switching would need `registry reembed`, R19). Commit nothing but the ledger note.

### Task 7: Live-action test sets and baselines [needs media: BBT, Kaze, Kaoru; Alice in Borderland is in the library]

**Files:**
- Modify: none in `src/` unless a real-file check fails (then: a failing test first, a fix, and the commit named for it)
- Workspaces (outside the repo): `~/fukikae-ws/{bbt-en,bbt-de,bbt-es,aib,kaze}`; labels under `~/fukikae-ws/subs/{bbt,kaze-kaoru}/`

**Interfaces:**
- Consumes: Tasks 1 to 5 (CLI commands and ledger), `fukikae run --to`, `fukikae registry resolve`, `eval identity --from-assignments`.
- Produces for Task 8: five workspaces diarized on their tuning and held-out episodes, ground-truth paths that pass the alignment check, `labels-aliases.yaml` and (Kaze) `labels-colours.yaml` per workspace, and baseline ledger rows tagged `t7-baseline` for both profiles on tuning episodes only.

- [ ] **Step 1: Copy media** with the read-only NFS volume (Global Constraints): BBT S01E01 to E03, Alice in Borderland S01E01 to E03, and Kaze, Kaoru's tuning and held-out episodes (named per the splits table, linked as `Kaze Kaoru - S01E<N>`). Confirm with `mise -C ~/projects/fukikae exec -- ffprobe -v error -show_entries stream=index,codec_type,codec_name:stream_tags=language,title -of compact <file>` that BBT has eng, ger and spa audio plus an English text subtitle, and that each Alice episode has the Japanese SDH SubRip track. Record stream indices, Kaze broadcast numbers and `df -h /` in the execution ledger.

- [ ] **Step 2: Workspaces.** Each `fukikae.toml` sets `source`, `source_language`, `gpu_device = 0`, `profile = "live-action"` and `[series] title` and `genre = "live-action"`; `aib` and `kaze` copy the `[asr.lang.ja]` table from `~/fukikae-ws/fairytail/fukikae.toml`. The three BBT workspaces share one media folder. Run `mise -C ~/projects/fukikae exec -- uv run --directory ~/projects/fukikae fukikae run <ws> <episode> --to diarize` for every named episode, one at a time, checking `df -h /` between sets.

- [ ] **Step 3: BBT labels.** Fetch the 17 S01 pages to `~/fukikae-ws/subs/bbt/transcripts/S01E<NN>.html` (`curl -sL <url> -o <file>`; URLs from the index, minus the S02 page noted in Decisions). Extract the release's English text track per episode to `~/fukikae-ws/subs/bbt/release-en/S01E<NN>.srt` (`mise ... ffmpeg -i <mkv> -map 0:<idx> <out>`); if it is image-based, take an external English .srt and retime it with `eval subalign --speech ~/fukikae-ws/bbt-en --episode S01E<NN>` first. Run `fukikae eval transcript-align --transcripts .../transcripts --subtitles .../release-en --out ~/fukikae-ws/subs/bbt/labels --episodes S01E01,...,S01E17`. Expected: cue share at least 0.7 per episode; Sheldon, Leonard, Penny, Howard and Raj on every episode. A lower share is reported, not tuned away.

- [ ] **Step 4: Kaze labels.** For each episode, `fukikae eval subalign <ARIB .ass> --reference <web .srt> --language ja --out ~/fukikae-ws/subs/kaze-kaoru/aligned/S01E<N>.ass`; record scale, offset, anchors, inliers and rms. If text matching fails (different caption text), use `--speech ~/fukikae-ws/kaze --episode S01E<N>`. Then `fukikae eval colour-map ~/fukikae-ws/kaze --episodes <all named> --gt-file .../aligned --write`; record the map, conflicts and which colours stay unmapped, and check the furigana events (Task 5's ruby rule).

- [ ] **Step 5: Coverage check per set.** `fukikae eval auto-label <ws> --episodes <tuning> --out <ws>/labels --report` with each set's ground-truth flags; write `labels-aliases.yaml` where one character has two spellings (`有栖良平: アリス` in `aib`). Record per set: labelled share of cues, coverage of speech, recurring characters at 5 s, and the alignment report. A set whose tuning episodes have fewer than 3 recurring characters is flagged thin in the execution ledger.

- [ ] **Step 6: Baselines on tuning episodes only.**
  - live-action profile: `fukikae registry resolve <ws> --episodes <tuning>`, then `fukikae eval identity <ws> --from-assignments --tune <tuning> <gt flags> --set <set> --tag t7-baseline`.
  - default profile: `cp -al <ws> <ws>-default`, remove its `profile` line (keep `[series]`), delete `.fukikae/registry/` and each episode's `05-characters/`, run `fukikae run <ws>-default <ep> --to characters` on the tuning episodes in order, then the same `eval identity --from-assignments` with `--set <set>` and `--tag t7-baseline`.
  Expected: 10 tune rows (5 sets by 2 profiles); `fukikae eval report --tag t7-baseline --format md` renders them by `en/de/es/ja` and `live-action`. No held-out episode is scored here.

- [ ] **Step 7: Record** the counts, maps, alignment fits and the baseline matrix in the execution ledger; commit any code fix from the real-file checks separately (`fix(eval): ...`).

### Task 8: Re-pick the live-action profile, hold out once, write the results [needs Task 7]

**Files:**
- Modify: `src/fukikae/profiles/live-action.toml`, `README.md` (live-action bullet), `CLAUDE.md` only if profile rules change
- Create: `docs/results/2026-11-xx-live-action-ground-truth.md`, `docs/results/2026-11-xx-live-action-ground-truth.jsonl`
- Test: `tests/test_profiles.py` (the profile's tables and `[resolve]` values)

**Interfaces:**
- Consumes: Task 7's workspaces and labels; `eval identity` (harness mode for sweeps, `--from-assignments` for held-out); `eval report --export`.

- [ ] **Step 1: Sweep on tuning episodes only** (harness, diarization segmentation, `--embedder wespeaker`, `--tag t8-sweep`): matcher `centroid`, `joint` with `--matcher-option episode_cannot_link=false`, and `joint` with `episode_cannot_link=true`; `--threshold` 0.50, 0.55, 0.60, 0.65, 0.70, 0.75; sets `bbt-en`, `bbt-de`, `bbt-es`, `aib`, `kaze`, `grid` with their tuning episodes and ground-truth flags. 108 runs, embeddings cached after the first per set.

- [ ] **Step 2: Pick by Task 9's rule** from `fukikae eval report --tag t8-sweep --metric clean` (and `merges`, `fraction`): clean characters summed over the six sets, then fewest merges summed, then mean fraction. BBT counts three times, as Fairy Tail's two tracks did in Task 9; state that. A language variant (`[resolve.lang.<L>]` or `[characters.lang.<L>]`) ships only if the best setting on that language's sets beats the genre pick there by at least 5 points of fraction with no fewer clean characters. Record the pick and every variant decision as rulings with their numbers.

- [ ] **Step 3: Write the profile test, then the profile.** The test asserts the picked `[characters]` matcher, `[resolve]` matcher, threshold and `episode_cannot_link`, and any language table. Run it RED, write `live-action.toml` (R3: every table names `backend`, `[characters]`/`[resolve]` name `matcher`; header comment states sets, numbers and the date, and drops PROVISIONAL), run GREEN with the full suite, and commit `feat(profiles): live-action profile from four-language ground truth` before any held-out run. Record that commit hash.

- [ ] **Step 4: Held-out, once per set** (`--tag t8-holdout`): for a joint pick, `fukikae registry resolve <ws> --episodes <tuning>`, then `fukikae registry resolve <ws> --episodes <held-out>` (tuning actors are anchors, R13), then `fukikae eval identity <ws> --from-assignments --tune <tuning> --holdout <held-out> <gt flags> --set <set>`; for a centroid pick, `fukikae run <ws> <ep> --to characters` in order instead of resolve. The default profile is held out once on the same episodes in the `-default` copies for comparison. Grid E05 is re-scored only if the pick differs from the provisional settings. Each verdict is recorded; nothing is re-tuned (R18).

- [ ] **Step 5: Write the results doc** from the ledger: `fukikae eval report --set bbt --set aib --set kaze --set grid --set renmin --set shendiao --format md` for the matrix and `--export docs/results/2026-11-xx-live-action-ground-truth.jsonl` for the rows. Sections: test sets and how each label set was made (with counts, alignment fits, colour map); baselines; sweep and pick; held-out verdicts per language; clip-set embedder results; limits. Limits name the biases: SDH names skew towards off-screen and voice-over speech (Netflix tags the speaker when the face doesn't show it), so labelled speech under-represents on-screen dialogue; colour covers only the main cast and unmapped colours are excluded; BBT's labels are English-timed and score the dubs on the English timeline; BBT is one show counted in three languages; any set flagged thin in Task 7. README's live-action bullet and the profile header match the doc.

- [ ] **Step 6: Commit** `docs(results): live-action ground truth, profile pick and held-out verdicts`
