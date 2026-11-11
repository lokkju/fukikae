# Character cards: final review fix waves

Copied from .superpowers/sdd/2026-11-08-character-cards/final-fix-wave-report.md at merge, 2026-11-10.

# Final-review fix wave (2026-11-10)

## Implementer report: reviews A, B and C

Branch `feat/character-cards`, worktree `.claude/worktrees/character-cards`, 30 commits on top of `7f59c91` (`942a50b`..`69debb1`). I was the only committer. Each finding got a failing test first, then the fix.

Verification at `69debb1`: the default suite gives 1418 passed and 26 deselected (1363 before the wave). `-m media` gives 2 passed. The real-model embedder tests (`-m "slow or gpu or media"`, GPU 0) give 5 passed, CAM++ included. ruff check and format are clean, and pyright reports 0 errors. GPU 0 was free throughout.

### Review A (cards, registry, packs)

| finding | commit | what changed |
|---|---|---|
| A1 | `942a50b` | `CardStore.put` keeps the stored fingerprint and `[meta]` when the id already exists. A fingerprint on the incoming card replaces the stored one (the recording path). A corrupted stored file is rewritten. |
| A2, A9 | `3bfa770` | Registry schema 3: `card_history.chosen_by` is written at insert, and a lock or unlock row records the chosen_by of the assignment it locks. The 2-to-3 migration fills it from the last earlier non-lock row with the same card. History rows are numbered in id order; `card_history(by_time=True)` sorts the same rows by time for `voices history`. |
| A3 | `1f4816c` | A `TarInfo` subclass refuses pax (x, g, X), GNU L/K and GNU sparse headers as soon as their 512-byte header is parsed, before tarfile buffers the payload. |
| A4 | `7b2c5c4` | Export runs import's checks. A row whose name, external id or language import would refuse, or whose card is for another language, is left out and listed. A series title, author, notes or series id over import's caps refuses the export. Config validates `target_language` (and suggests `pt-BR` for `pt_BR`), and `set_character_external_id` validates the id. |
| A5 | `648e840` | `[voice.post]` keys at their defaults and empty `[style]` values are dropped. Language tags (card target, references, registry keys, config) are case-normalized to BCP 47's conventional form before hashing. The spec example card's id changes once (`31a12ee8...` to `6578f2f6...`), and the test says why. |
| A6 | `072ae5c` | Rollback no longer deletes the cards and sidecars it created (they are content-addressed, so a concurrent import may be using them). Merged fingerprints are still restored. |
| A7, A8 | `8375f77` | `derive` drops the fingerprint. `assign` and `edit` run the stage's card check (`check_loaded_card`, split out of `check_card`) and refuse missing sidecars before assigning. `tests/cards/conftest.py` maps the test backend `fake` to the test Synthesizer. |
| A10 | `6d618d0` | `_session` turns RuntimeError (verify render, worker crash), sqlite `DatabaseError`, a newer schema and OSError into one-line errors; a command's own `typer.Exit` passes through. `--lang` is validated and normalized. |
| A11 | `632ccb2` | Import and `assign <pack>:<character>` warn when a card's fingerprint threshold is below 0.5. |
| A12 | `d91350e` | The required voice keys are NFC. `[style]` key names must be lowercase snake case of at most 32 characters. The spec's catchphrase key is `target`. |

### Review C (stages, media, docs)

| finding | commit | what changed |
|---|---|---|
| C1 | `07a8a19` | The tts stage key's `identity` holds, per backend in use, `{options, identity}`, so flipping tf32, fp16 or a mode changes the key. |
| C2 | `11b1c61` | `plan_fit` with a zero-length or inverted slot places the render at the slot, unstretched, then borrows and overflows as usual. Subtitle events with end <= start are skipped as `empty`. |
| C3, C4 | `3e1b36c` | The translate key holds the speaking characters' names. Each cue's budget uses the rate of its character's card backend, else `clone_backend`'s, and extras use the `[tts]` backend's. The key holds `cps: {extras, unassigned, characters}`, listing only rates that differ from `unassigned`. The results doc's budget line is corrected. |
| C5 | `d6a526c` | In subtitle mode, a `TranslateBackendError` after `MAX_BACKEND_FAILURES` falls through to `_keep_subtitle_lines`, and the episode continues. |
| C6 | `3c814ad` | GpuBusy, TranslateBackendError and BackendError are user errors. A multi-episode run prints every episode's result, then raises the first unexpected error's traceback. |
| C7 (= B6), B7 | `fa73f84` | `gpu_lock` is reentrant per thread; another thread opens its own descriptor and waits on flock like another process would. The lock directory is fixed at `/tmp/fukikae-<uid>` (it ignores XDG_RUNTIME_DIR and TMPDIR). After each flock the acquirer checks that its descriptor is still the file at the path, and takes the lock again if it isn't. `held_fds` returns the calling thread's descriptors. |
| C8 | `90f57c9` | `merge_layers`: subtitle events with the same spoken text that overlap or touch (within 50 ms) become one cue. |
| C9 | `a7b950c` | SenseVoice and emotion2vec declare `option_keys`. `check_tag_config` (run by `tag_enabled` and the stage) refuses unknown `[tag]` and `[tag.emotion]` keys, with a "did you mean" hint, and a non-boolean `enabled`. For a backend without `option_keys`, only near-misses of the stage keys are refused. |
| C-docs | `8427274`, plus `07a8a19` for indextts2.py | The anime and live-action profile comments say tag runs on the GPU. The README says only Kokoro is rate-fitted and documents `fukikae run <ws> E1 E2 ...` with the overlapped timefit and mix. The in-repo ledger's D26 line now notes that the stage key gained the options in C1; its per-render-key claim was already correct. I did not edit the controller's `progress.md`. |

### Review B (TTS backends, workers, eval)

| finding | commit | what changed |
|---|---|---|
| B1 | `cb76a92` | `reference_origins(card)`. Held-out exemplars exclude turns that overlap the rendered card's reference origins. A card with a reference that has no origin gets a note. |
| B2 | `705ef0d` | WER-floor clips are named by the sha256 of the vocals' bytes and the exact span. |
| B3 | `8e1c9b5` | `WorkerSynthesizer.lock_revision()`. `model_identity` appends `+lock:<12 hex>` to a worker backend's revision. I left `model_revision` (the card pin) unchanged on purpose, so existing cards still pass `check_card` after a lock bump. Only cache keys and ledger rows change. |
| B4 | `1e1040a` | `quantize_factor`: rate factors round up to 0.05 steps, capped at max_speedup, for both the first estimate and the refit. eval tts-ab reuses the stage's render when `07-tts/plan.json` has the same card and a key derived from the job's render key (plain, fitted or retried) and the wav is still cached. |
| B5 | `13b8456` | A card seed above 2^32-1 is refused at load. `retry_seed()` wraps modulo 2^32. |
| B8, B9, B10 | `612ae79` | `death_signal_hook()` resolves prctl through ctypes in the parent, once. The worker starts with `start_new_session=True`, and every terminate or kill signals its process group. At EOF the wait is bounded by `EOF_WAIT_S` (5 s), then the kill path runs. |
| B11, B12 | `613f9d9` | Pooled WER skips cues with zero reference words. The held-out headline is a mean over characters, like the ceiling, and the report says both are. |
| B13 | `89062c7` | `_set_tf32` sets only `matmul.allow_tf32` (cuDNN's TF32 is already torch's default, so behaviour is unchanged). Docstring and README corrected. |
| B14 | `c57509a` | The blinding seed is `secrets.randbits(64)`, stored as `blind_seed` in key.toml only. The spec seed still picks the cues. |
| B15 | `6dde2c1` | A second `load` is refused with "already loaded; unload it first", and the first model stays usable. |
| B16 | `69debb1` | CAM++ really gives 192 dimensions (I confirmed this on FT E12), so the class's declared 512 was wrong as well as the test. The class now declares 192, and the real-model test checks against the embedder's declared dim. `docs/results/2026-11-05-embedders.md` still lists 512 for an older voxceleb CAM++ file; I left that historical row alone. |

### Calls I made

- **Hooks.** With `git commit -- <paths>`, pre-commit's Python hooks report "no files to check" (A1's commit `942a50b`). I ran ruff by hand on it, and pyright passed later on the whole tree. From A2 on, I staged the explicit paths with `git add` and committed with no pathspec, so the hooks really ran. The ledger's earlier "false alarm" note about this was wrong: the skip happens whenever a commit uses a pathspec, docs or code.
- **A5** normalizes language case (`PT-br` to `pt-BR`) but doesn't validate card target languages beyond non-empty. Config, `--lang` and the pack paths validate.
- **A9.** Rows are numbered in id order everywhere (`revert --to n` follows that numbering). Only `voices history` shows them by time, keeping each row's `n`.
- **C4.** A per-character rate is listed in the key only when it differs from the unassigned rate, so a default clone assignment doesn't re-translate. A stock fallback (Kokoro, 15.5) for a character that defaulted to the IndexTTS2 rate does re-translate on the next run, because its budget really changed.
- **C6.** An unexpected error still exits with its traceback, but only after every episode has been reported.

### Not fixed or deferred

Nothing. I re-ran the full suite only with default markers plus `-m media` and the embedder real-model tests; the other slow and gpu tests (real IndexTTS2 renders and the like) were not re-run in this wave.

## Pre-merge round (2026-11-10, after the re-review)

Ten commits, `b1ad26d..27aca49`, on top of `69debb1`. Verification at `27aca49`: the default suite gives 1443 passed and 26 deselected (1418 before). `-m media` on GPU 0 gives 2 passed. ruff check is clean, pyright reports 0 errors, and `ruff format --check` is clean on `src` and `tests`; on the whole tree it flags Python code blocks in three plan documents under `docs/superpowers/plans/`, which this round didn't touch.

| Item | Commit | What changed |
|---|---|---|
| 1 C8 | `b1ad26d` | Identical-text events merge only when `cue.start < previous end`. The layered-events test now has an overlapping event (merged) and a touching one (kept apart), and a new test keeps "Wait!" 10-11 and "Wait!" 11-12 as two cues. `LAYER_TOUCH_S` is gone. |
| 2 leftover scan | `9c6f0c3` | `leftover_jobs` reads `/proc/<pid>/cmdline` as an argv array (`_argv`, `_is_job`): a worker is `-m` followed by exactly `fukikae.backends.tts.worker`; a llama-server is `basename(argv[0]) == "llama-server"` plus an `--api-key-file` (either spelling) whose basename starts `fukikae-llama-key-`. A `bash -c "... fukikae.backends.tts.worker ..."` and an editor that names llama-server as an argument don't count (real-process test plus an 11-case table). The pid listing is now `_proc_pids()`, and an autouse fixture in `tests/conftest.py`, `started_pids`, limits it to the pids a test added, so every test that takes a GPU lock (gpulock, stages, workers) ignores real fukikae jobs on the machine. The llama-server leftover test now starts its sleeper with `argv[0] = "llama-server"`. |
| 3 killpg | `3db2de3` | `_reap` sends SIGTERM, waits up to `KILL_WAIT_S` for the leader to exit, sends SIGKILL, and only then `wait()`s, every time, including when the worker had already exited. Nothing before it reaps any more: unload, `_stop`, `_fail`, the EOF path, `_request` and the stderr tail check the exit with `waitid(WEXITED, WNOHANG, WNOWAIT)` (`_exit_status`, `_wait_exit`) instead of `poll()`/`wait(timeout)`. `_signal_group` does nothing once Popen has reaped the leader, since the pgid could belong to someone else by then. Tests: a worker that exits before its hello has its `sleep 60` grandchild killed (it survived before), and a spy on `os.killpg` shows both signals go out while `/proc/<leader>` still exists, for a running and an exited leader. |
| 4 PDEATHSIG | `d9348e2` | The hook runs in an `os.fork()` child that exits with whether `PR_GET_PDEATHSIG` reads SIGTERM; the test then asserts the pytest process's own death signal is still 0. |
| 5 migration | `274a2cd` | `_migrate_2_to_3` first rewrites `card_history` and `card_assignments` keys with `normalize_language`. History rows that meet under one key merge in id order. Two current cards that meet keep the newer one (higher history id), the older stays in history (reachable with `voices revert`), and each collision is logged as a warning naming the character, both cards and both spellings. I chose keep-newest over refusing because nothing is lost and a refusal would block opening the registry. |
| 6 stale id | `a9945cd` | `CardStore.get` raises `StaleCardError` when the file's id doesn't match and its text holds something the A5 rules drop or rewrite (language case, a `[voice.post]` default, an empty `[style]` field): the message says the card id rules changed, gives today's id and says to re-assign with `fukikae voices assign <workspace> <character> <old id>`. `voices assign` takes `CARD` as an id, not a file, so that is the command; given the stale id, it stores the card under its new id, says so and assigns that. Any other mismatch still says the file was edited in place. |
| 7 holder pid | `8964e8e` | The pid and job are written (ftruncate+pwrite) straight after a flock that passes the inode check, before the leftover scan. When the scan refuses, the line is cleared before the descriptor closes; a waiter that times out never touches the holder's line (now asserted). |
| 8 B7 docstring | `8964e8e` | The module docstring says the inode check covers a file unlinked or replaced before or while the lock is taken, and that a file removed while the lock is held isn't noticed (a later job would lock the new file beside it). |
| 9 `_session` | `63a9cef` | click's `Exit` and `Abort` pass through, so Ctrl-C at a prompt prints click's "Aborted!" instead of "error: Abort". One-line errors are now `user_errors()` (BackendError, GpuBusy, TranslateBackendError, CardError, ValueError and the rest), a newer registry schema (now `RegistrySchemaError`, a RuntimeError subclass), sqlite and OS errors. Any other RuntimeError keeps its traceback; fukikae has no debug flag to point at. |
| 10 empty export | `e77cecd` | When `export_selection` returns no rows, `voices export` prints the left-out list, exits 1 with "no character has a card that can be exported; nothing was written", and says to name characters with `fukikae registry rename`. No pack is written. |
| 11 tag re-run | `27aca49` (comment only) | See below. |

### Item 11: why the tag stage re-ran

I copied S01E11's and S01E12's tag inputs (`vocals_16k.wav`, `turns.json`) and `fukikae.toml` out of `~/fukikae-scratch/t12/fairytail` into a temp workspace and recomputed the tag cache key with the source tree at `c407845`, `afcb513`, `6caa15d`, `a7b950c`, `69debb1` and HEAD. `c407845` reproduces E12's stored key (`a675e74b7f6f770c`, written 2026-11-09 20:16). `afcb513` (D26: SenseVoice and emotion2vec on the GPU by default) and every later tree reproduce the key E11 re-ran under at 2026-11-10 02:12 (`794b27232b9508a4`); E11's other inputs hadn't changed since 2026-11-09 19:40. The config key differs only in `"device": "cpu"` becoming `"cuda"` and the new `"emotion_device": "cuda"`. Backend, options and both model identities are identical, and a7b950c's validation changed nothing in the key.

I kept the device in the key, because the outputs differ. On E12, HEAD on CPU reproduces the stored 2026-11-09 tags.json exactly, so CPU is deterministic and the code hasn't changed it. HEAD on CUDA (GPU 0) gives the same 12 SenseVoice events and the same 537 emotion spans and labels, but 271 of the 537 emotion2vec confidences differ, by up to 0.0043 (none crosses `SCORED_MIN`, 0.5). A confidence is the tts plan's `emotion_vector` weight (`stages/tts.py`), so a CPU-made tags.json and a GPU-made one give different tts keys and renders. `27aca49` records this in a comment on `TagStage.config_key`. Every workspace tagged before D26 re-runs tag once (about a minute per episode on GPU 0).
