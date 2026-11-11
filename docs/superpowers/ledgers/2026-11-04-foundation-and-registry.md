<!-- SDD progress ledger for the foundation plan, preserved at merge (2026-11-05): rulings R1-R15 and every deferred minor. -->
# SDD ledger — plan: docs/superpowers/plans/2026-11-04-foundation-and-registry.md

Repo: ~/projects/doppio, branch feat/foundation (from main 2d705b6). Spec: docs/superpowers/specs/2026-11-04-doppio-design.md (reachable).
Test media: ~/doppio-ws/media/{Chimera (2021),Death Note}/Season 1/ (local copies; ums share unmounted on ml-01 since 2026-11-04).

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T2 -> T5, T9, T12, T14, T16 | models (Transcript, Diarization, Separation, Assignments) | consistent |
| T3 -> T5, T6 | Config.resolve, cfg.gpu_device | consistent |
| T4 -> T8-T12 | Backend protocols, loader extras map | consistent; each backend task adds its extra to the map |
| T5 -> T7 | Stage protocol takes `Episode`, defined in T7 | CONFLICT: T5 precedes T7. Ruling R1 |
| T5 cache_key | "inside the workspace" needs the workspace root | GAP: Episode lacks it. Ruling R1 |
| T6 -> T8 | wait_for_vram, device_string | consistent |
| T7 -> T8 | 00-extract/audio.wav | consistent |
| T9 -> T11 | AsrStage writes 03-asr; T11 needs it to write eval/asr-<backend> and force a backend | GAP. Ruling R2 |
| T12 -> T14, T15, T16 | clusters.npz, turns.json | consistent |
| T13 -> T14, T15 | Actor/Character, store methods | consistent; T15 adds split_cluster (plan says so) |
| T14 -> T15 | ClusterEvidence | consistent (T15 after T14) |
| T5 -> T15 | Manifest record deletion to mark stale | consistent |
| T1 self | addopts excludes gpu, media | consistent |
| T7 self | ffmpeg-only test marked `media`, but media tests skip without DOPPIO_TEST_MEDIA | CONFLICT. Ruling R3 |
| T2-T4, T6, T8-T16 self | tests vs code vs files | consistent |
| T10 self | qwen-asr API unknown to the plan | plan already tells implementer to check the README; no ruling |
| T17 | needs operator labels | stops at a queued action by design |

Ruling R1: T5 creates `src/doppio/workspace.py` with the `Episode` dataclass `(source: Path, id: str, work: Path, workspace_root: Path)`; T7 adds `Workspace` and `find_episode` to the same file. cache_key treats paths under `ep.workspace_root` as workspace artifacts. — The plan's T5 can't reference a type that doesn't exist yet; workspace_root is the minimal field the fingerprint rule needs. — If wrong: one field rename in T7.
Ruling R2: `AsrStage.__init__(self, backend: str | None = None, out_dir: str = "03-asr")`; `backend` overrides the resolved ASR backend name (options still from the resolved per-language table when the names match, else the backend's defaults); T11 uses `AsrStage(backend=b, out_dir=f"eval/asr-{b}")`. — T11's contract requires it and the plan didn't define the hook. — If wrong: a constructor tweak in T9/T11.
Ruling R3: add pytest marker `ffmpeg` (tests needing only the ffmpeg binary; run by default when ffmpeg is on PATH, skip otherwise). T7's read-only/awkward-path test uses `ffmpeg`, not `media`. — Review Focus 1, 2 and 5 must run without Plex media to be useful. — If wrong: a marker rename.

## Tasks
Task 1: dispatched (BASE 2d705b6, implementer a6d2b2a48bd0582e4, sonnet)
Task 1: implementer DONE_WITH_CONCERNS (d8e8ce8). Reviewer dispatched (ae203dd0ebece29c9).
Task 1: minor (deferred): pre-commit hooks not installed in .git/hooks (commitizen commit-msg inactive); install before merge.
Note: pre-commit's formatter reformatted the plan's code blocks; controller committed that as c8de2cf (style, formatting only; auto mode refused discarding it).
Task 1: complete (commits 2d705b6..d8e8ce8, review clean)
Task 1: minor (deferred): pyright pre-commit hook additional_dependencies lists only typer; hook pin v1.1.414 vs dev pyright>=1.1.0. Fix before installing hooks.
Task 1: minor (deferred): ruff S101 ignored globally and per-file (redundant).
Task 1: minor (carried to Task 7): ffmpeg marker needs conftest skip logic when ffmpeg isn't on PATH.
Task 2: dispatched (BASE c8de2cf, implementer a149e0c340c40ae06, sonnet)
Task 2: implementer DONE (b7751eb). Reviewer dispatched (a7521159447e3e60f).
Task 2: review: spec OK, 1 Important (atomic_path temp loses suffix; ffmpeg/soundfile need it). Fix round 1 sent to implementer (FIX_BASE b7751eb).
Task 2: minor (deferred): read_json validates before checking schema_version, so a newer schema with new required fields raises ValidationError, not ArtifactVersionError; parse dict, check version, then validate (also removes getattr noqa).
Task 2: minor (deferred): atomic_path has no fsync (power-loss safety only).
Task 2: minor (deferred): no test that a lower schema_version loads.
Task 2: fix round 1/5 (1 addressed, 0 open; commits b7751eb..4dce235)
Task 2: complete (commits c8de2cf..4dce235, review clean)
Task 3: dispatched (BASE 4dce235, implementer a6c48ecdfa0fa201a, sonnet)
Task 3: implementer DONE (07ef1b7).
Task 3: complete (commits 4dce235..07ef1b7, review clean)
Task 3: minor (deferred): load_config's user_config typed Path|None|object (collapses to object; a str is silently ignored).
Task 3: minor (deferred): non-table `lang`/`pair`/`registry` values raise AttributeError, not ConfigError.
Ruling R4: Config.resolve inherits base-table options only when the override's backend equals the base backend (or the override omits backend); an override naming a different backend gets only its own options. — The spec's per-language example swaps faster_whisper for qwen3_asr, and backend-specific options (compute_type, vad_filter) must not leak across backends. — If wrong: one condition in resolve. Implemented with Task 4 (small, adjacent), with a test.
Task 4: dispatched (BASE 07ef1b7, implementer a9a85da1ef9cd3af5, sonnet; includes R4 config fix)
Task 4: implementer DONE (e1dba87, 0e67bc7).
Task 4: complete (commits 07ef1b7..0e67bc7, review clean)
Task 4: minor (deferred): BackendUnavailable hint says `pip install`; project is uv-only, use neutral wording.
Task 4: minor (deferred): loader _load catches only ImportError; a missing attribute raises AttributeError and crashes available().
Task 4: minor (deferred): no test for create() on a kind with no entry points; base-without-backend + override-with-backend drop unpinned.
Task 5: dispatched (BASE 0e67bc7, implementer a3b981b83986c26b3, sonnet)
Task 5: implementer DONE (57ac929); no RED evidence (implemented before first test run).
Task 5: review: spec OK, 1 Important (interrupted-rerun test doesn't pin RF3). Fix round 1 sent (FIX_BASE 57ac929).
Task 5: minor (deferred): `i >= start and from_stage is not None or ...` precedence reads like a bug; parenthesize.
Task 5: minor (deferred): hash cache (size, mtime_ns) racy-clean window; skip caching when mtime within ~2 s of now.
Task 5: minor (deferred): wrong-shape hashcache entry raises TypeError instead of being ignored.
Task 5: minor (deferred): directory fingerprint includes stray `.*.tmp*` files from a SIGKILL'd atomic_path; filter them.
Task 5: minor (deferred): from_stage after to_stage accepted silently; local import of atomic_path; corrupt manifest discarded without a warning.
Task 5: fix round 1/5 (1 partially addressed: manifest-first caught, never-records-failures not; commits 57ac929..2b8b02d). Fix round 2 sent: assert b's record is failed after the interrupt (FIX_BASE 2b8b02d).
Task 5: fix round 2/5 (1 addressed, 0 open; commits 2b8b02d..1d49d18)
Task 5: complete (commits 0e67bc7..1d49d18, review clean)
Note: a round-1 re-reviewer left a scratch copy at /tmp/rr-mut-eTYE (157M); its rm was denied by auto mode; left for the user.
Task 6: dispatched (BASE 1d49d18, implementer a82e7e46682324652, sonnet)
Task 6: implementer DONE (5ad0c3c); no RED evidence. implementer-common.md now requires RED output.
Task 6: complete (commits 1d49d18..5ad0c3c, review clean)
Task 6: minor (deferred): test_module_importable_without_pynvml is vacuous (doesn't block pynvml import); "1" in msg assertion weak, use "GPU 1".
Task 7: dispatched (BASE 5ad0c3c, implementer a9d6bca6c97cdc5d9, sonnet)
Task 7: implementer DONE_WITH_CONCERNS (eb5ef8d): added huggingface-hub core dep, runctx module for --no-wait, ruff config tweaks.
Task 7: review: spec OK, 1 Important (RF2 track choice not pinned by audio content; fixture lacks video so stream indexes coincide). Fix round 1 sent (FIX_BASE eb5ef8d).
Task 7: minor (deferred): readonly fixture passes vacuously as root; assert not os.access(W_OK).
Task 7: minor (deferred): "und"/commentary tracks can win choose_audio_track's first-non-English fallback (matches spec wording).
Task 7: minor (deferred): doctor's `except ImportError` for huggingface_hub is dead code (core dep).
Task 7: minor (carried to Task 8): runctx.vram_timeout_s has no consumer or test yet; Task 8's GPU stage must read it and test --no-wait.
Task 7: minor (deferred): test_doctor_runs asserts only labels, not exit code or ok/MISS.
Task 7: fix round 1/5 (1 addressed, 0 open; commits eb5ef8d..cd79da6)
Task 7: complete (commits 5ad0c3c..cd79da6, review clean)
Task 7: minor (deferred): the -map mutation errored rather than hitting the tone assertion; a `0:1` wrong-track mutation wasn't run.
Task 8: dispatched (BASE cd79da6, implementer a043e515e88ff78d8, sonnet)
Task 8: implementer DONE_WITH_CONCERNS (00ba356). demucs 4.1.0 + torch 2.15.0+cu132; peak ~543 MiB on the 90 s clip.
Ruling R5: SeparateStage's default background is `mix - vocals` (option background="residual"); "stems" (sum of non-vocal stems) stays available. — Residual reconstructs the input exactly (vocals + background == mix), which the remix needs; htdemucs_ft stem sum leaves a -19 dB residual. Deviates from the plan's "background = sum of non-vocal stems". — If wrong: flip one option default.
Ruling R6: cfg.gpu_device is the nvidia-smi (NVML, PCI order) index. doppio.cli sets os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID") before anything imports torch, and gpu.device_string(cfg) maps the NVML index to its position in CUDA_VISIBLE_DEVICES when that's set (returning "cpu" with a logged warning if the card isn't visible), else uses it directly. wait_for_vram keeps the NVML index. — Without it the VRAM gate can check one card and torch load onto another. — If wrong: a helper change in gpu.py. Sent to Task 8's implementer before review (concerns about correctness).
Task 8: R5/R6 applied (395b930, fbca83e). Review dispatched over cd79da6..fbca83e.
Task 8: review: spec OK, 2 Important (ConfigError fallback drops options without backend key; load() outside try/finally). Fix round 1 sent (FIX_BASE fbca83e).
Task 8: minor (deferred): test_waits_then_runs_when_vram_frees runs ffmpeg without @pytest.mark.ffmpeg; dead PATH skip.
Task 8: minor (deferred): stems/residual arithmetic in demucs.py has no non-GPU test.
Task 8: minor (deferred): backend's Separation.vocals_16k placeholder holds the 44.1 kHz path.
Task 8: minor (deferred): CUDA_DEVICE_ORDER only set on doppio.cli import; setdefault keeps a user's FASTEST_FIRST; document in device_string.
Task 8: fix round 1/5 (2 addressed, 0 open; commits fbca83e..88edc4a)
Task 8: complete (commits cd79da6..88edc4a, review clean)
Task 8: minor (deferred): separate fallback doesn't strip a falsy `backend` key from options (resolve does).
Task 9: dispatched (BASE 88edc4a, implementer a52f89595459598f2, sonnet)
Task 9: implementer DONE_WITH_CONCERNS (cc5a28d): torchaudio 2.11.1 forced_align; PyAV 19 workaround (wave+numpy); nvidia-cublas-cu12 in whisper extra for CTranslate2; whisper peak ~4.5 GiB.
Task 9: review: spec OK, 1 Important (test_wav2vec2 imports torch at module level; breaks collection without whisper extra). Fix round 1 sent (FIX_BASE cc5a28d).
Ruling R7: no hand-written CTC Viterbi fallback is owed — my clarification asked for one only if torchaudio had removed forced_align, and 2.11.1 has it. — Avoids writing unused code. — If wrong: a torchaudio bump breaks alignment loudly at import (see deferred minor).
Task 9: minor (deferred): ctc_align imports torchaudio.functional unconditionally; catch ImportError/AttributeError with a clear message naming the version, cap torchaudio <next-major.
Task 9: minor (deferred): _read_wav16k duplicated in faster_whisper.py and wav2vec2.py; move to a model-free doppio/audio.py.
Task 9: minor (deferred): nvidia-cublas-cu12 needs `; sys_platform == 'linux'` marker in the whisper extra.
Task 9: minor (deferred): AsrStage proceeds when no aligner supports the language and fails later in load_for; raise a clear error.
Task 9: minor (deferred): FasterWhisperTranscriber ignores the `languages` option; detect_language hardcodes vad_filter.
Task 9: fix round 1/5 (1 addressed, 0 open; commits cc5a28d..29dd412)
Task 9: complete (commits 88edc4a..29dd412, review clean)
Task 10: dispatched (BASE 29dd412, implementer ad16b27a5ca32f031, sonnet)
Task 10: implementer DONE_WITH_CONCERNS (3c329e8): transformers capped <5 (qwen-asr pins 4.57.6), huggingface-hub <1.0; ASR vram 5800, aligner 3200.
Ruling R8: Qwen3-ASR's ~28 s segments stand for this plan; cue segmentation from word timing and diarization turns belongs to the translate plan's segmentation step. — ASR segments feed only alignment and CER here; neither needs short segments. — If wrong: the chunker gains a pause-split threshold.
Task 10: reviewer dispatched (a711eed6283d4e472). Process note: the dispatch told the reviewer not to flag segment length (citing R8), which pre-judges a finding; the controller should have let it surface and ruled. Final review should look at segment length fresh.
Task 10: complete (commits 29dd412..3c329e8, review clean)
Task 10: minor (deferred): qwen3 aligner count-mismatch fallback and empty-words branches untested; chunk-count assertion loose (2 or 3).
Task 10: minor (deferred): max_chunk_s unvalidated (tiny values loop forever; >30 exceeds model limit).
Task 10: minor (deferred): Transcript.language "" when no chunk language maps; batch_size/max_chunk_s undocumented; comment the transformers<5 / hub<1.0 pins in pyproject.
Task 11: dispatched (BASE 3c329e8, implementer a704b003418c8997e, sonnet; includes the full-episode Chimera ASR comparison)
Task 11: implementer DONE (975a02f, c3957d7, 975ce57). Chimera S01E01 CER: qwen3_asr 0.1251 (121 s), faster_whisper 0.1287 (140 s); [asr.lang.ko]=qwen3_asr in ~/doppio-ws/chimera.
Task 11: review: code approved; 2 Important (results doc overstates winner; Latin-in-reference penalty unstated) + confirmed gap (cached rerun reports ~0 s). Fix round 1 sent (FIX_BASE 975ce57).
Task 11: minor (deferred): no test for extract_subtitle_text/evaluate (ffmpeg-marked srt mux test).
Task 11: minor (deferred): eval cli falls back to "en" normalization when source_language unset (keeps spaces for Korean); unknown --backend error path; empty reference gives opaque jiwer error; _SDH strips parenthetical dialogue (document).
Task 11: fix round 1/5 (3 addressed, 0 open; commits 975ce57..5cd72e3). With Latin stripped: qwen3_asr 0.1177, faster_whisper 0.1173 (tie).
Task 11: complete (commits 3c329e8..5cd72e3, review clean)
Task 11: minor (deferred): results doc :15 still cites "small runtime and CER edge" as reasons (contradicts :13 and the Latin table; segment count is the only stated basis left); :19 "guaranteed error on both sides" should read "a Hangul transcription cannot match".
Task 12: dispatched (BASE 5cd72e3, implementer a80a584fe3a02f100, sonnet)
Task 12: implementer DONE (f379a89). pyannote vram 3000, ecapa 1900; pyannote-audio>=4.0.7, speechbrain>=1.1.2 co-install without new pins.
Ruling R9: clusters.npz clean_seconds is the total of ALL clean spans for the cluster, before the 50-span embedding cap. — The 20 s new-character bar measures how much a speaker talks; summing only embedded spans would understate it. — If wrong: one sum moves.
Task 12: review: spec OK, 1 Important (span-cap test can't fail; R9 past the cap untested). Fix round 1 sent (FIX_BASE f379a89).
Task 12: minor (deferred): _read_wav16k now copied six times (faster_whisper, wav2vec2, qwen3, demucs variant, pyannote, ecapa); _resolve private import across stages; _gpu_wait duplicated in asr.py and diarize.py. One shared stage/audio helper module would remove all.
Task 12: minor (deferred): ecapa savedir = HF snapshot dir (SpeechBrain may write there); full-episode pyannote VRAM unmeasured (3000 MB from the 90 s clip).
Task 12: fix round 1/5 (1 addressed, 0 open; commits f379a89..3be6771)
Task 12: complete (commits 5cd72e3..3be6771, review clean)
Task 13: dispatched (BASE 3be6771, implementer a2b8dcf5331e1007c, sonnet)
Task 13: implementer DONE (a33cd0d); added exemplar_count().
Task 13: complete (commits 3be6771..a33cd0d, review clean)
Task 13: minor (deferred): test_centroid_none_without_exemplars checks set_actor only (misnamed); set_actor can't clear a name.
Task 13: minor (carried to Task 14): add_exemplars on an unknown actor raises sqlite3.IntegrityError (not KeyError); an empty embeddings array inserts a zero-length blob that later breaks recompute (callers must skip zero-row clusters).
Task 14: dispatched (BASE a33cd0d, implementer aa474c4428a56a8ce, sonnet)
Task 14: implementer DONE_WITH_CONCERNS (3a1a99e).
Ruling R10: pin() inherits only previous assignments that carry a character_id (matched, new, pinned); clusters whose previous status was extra or review are re-matched each run. — Pinning exists to keep a character's voice stable; freezing extras/review as "pinned, no actor" loses their status and blocks an extra from becoming a character. — If wrong: one filter in pin().
Ruling R11: CharactersStage calls registry.delete_episode(episode) before adding exemplars whenever no previous assignments.json exists (first run or hand-deleted), so a plain rerun can't duplicate exemplars. — Duplicate exemplars skew centroids toward re-run episodes. — If wrong: one call removed.
Task 14: R10/R11 applied (dc42caf). Ruling R11 amended: delete_episode runs whenever assignments.json OR turns.prev.json is missing (pinning impossible either way). — Implementer's broader condition is the correct one.
Task 14: review (opus): 2 Important (two-file write can mis-pin after interrupt; unpinned re-run duplicates exemplars / duplicate actor). Fix round 1 sent (FIX_BASE dc42caf): turns stored inside Assignments (optional field), turns.prev.json dropped; Registry.delete_exemplars(episode, cluster) for uninherited prior clusters.
Task 14: minor (deferred): an actor with no character yields matched with character_id None (never pinned; re-adds exemplars) — unreachable via this stage today.
Task 14: minor (deferred): registry/cli.py imports private job._load_manifest; interrupted rematch loses old assignments (prints None -> x).
Task 14: minor (deferred): per-call transactions, so an interrupted run can leave part of an episode's exemplars written.
Task 14: fix round 1/5 (2 addressed, 0 open; commits dc42caf..c461910). Added pin_sources + swap-safe relabel_exemplars.
Task 14: complete (commits a33cd0d..c461910, review clean)
Task 14: minor (deferred): relabel_exemplars commits before assignments.json is written; Ctrl-C between them can make exemplar labels drift (swap case swaps back); same class as the per-call transaction minor.
Task 14: minor (deferred): two current clusters inheriting one previous cluster: the reverse dict keeps one; exemplars follow one cluster only (untested edge).
Task 14: minor (deferred): stale turns.prev.json files from earlier runs remain on disk (harmless).
Task 15: dispatched (BASE c461910, implementer afaa370506169a378, sonnet)
Task 15: implementer DONE (a6c9832): job manifest helpers public; load_clusters moved to matching; merge/split rewrite ids in assignments.json; scipy core dep; cluster --threshold is cosine distance (default 1 - match_threshold).
Task 15: complete (commits c461910..a6c9832, review clean)
Task 15: minor (deferred): test_cli_merge_and_split doesn't assert the assignments.json id remap or stale marking after merge/split; --threshold untested.
Task 15: minor (deferred): merge marks only DROP's episodes stale (KEEP's centroid changes too); _open creates the registry on list/show; cluster's registry writes commit before assignments.json (interrupt leaves a seeded registry that refuses rerun).
Task 15: for final review: verify the first `doppio run` after `registry cluster` pins the seeded ids (assignments carry turns, so R11's delete path shouldn't fire).
Task 16: dispatched (BASE a6c9832, implementer ad4dc70aac70c2bc5, sonnet)
Task 16: implementer DONE (5fb03a5).
Task 16: review: 2 Important (bare `?` in labels.yaml is invalid YAML and YAMLError uncaught; labels not normalized/coerced). Fix round 1 sent (FIX_BASE 5fb03a5): quote labels on write, wrap YAMLError, reject non-string labels, casefold identity.
Task 16: minor (deferred): --force resample leaves stale wavs; zero-recurring prints only "rework" (say "nothing recurring"); CLI test covers only the no-recurring path; span past pcm end gives an empty wav; merged_ids parallel list (use triples).
Task 16: fix round 1/5 (2 addressed, 0 open; commits 5fb03a5..da7c742)
Task 16: complete (commits a6c9832..da7c742, review clean)
Task 16: minor (deferred): _Quoted representer registered on global yaml.SafeDumper (scope to a local Dumper); bare-bool test's final sample assertion passes for an unintended reason.
Task 17: steps 1-3 dispatched (BASE da7c742, runner ae1e7ef38ba43baae, sonnet); stops for operator labeling.
Task 17: steps 1-3 done. S01E01 14 clusters (5 new, 9 extra); S01E02 16 clusters (4 matched 0.971/0.951/0.866/0.699, 4 new, 2 review 0.650/0.678, 6 extra); 9 actors; 102 clips; peak GPU 7366 MiB. Waiting on operator labels (queued action). Steps 4-5 after labels.
Final whole-branch review dispatched in parallel (tasks 1-16 code complete).
Final review (opus): ready to merge with fixes. C1 rematch/R11 matches an episode against its own earlier new actors (centroid kept after delete_episode) -> threshold sweep meaningless; I1 eval asr overwrites the pipeline's "asr" manifest record; I2 pinned + newly matched clusters can share an actor unflagged; I3 merge leaves two characters (two voices) for one person. Must-fix minors: pyright hook deps + install hooks; uv wording in BackendUnavailable; results doc lines 14/18; registry cluster in one transaction; merge/split test coverage; rebuild-episode-exemplars-in-one-transaction for Task 14 windows; Segment doc + max_chunk_s default 28.
Ruling R12: v1 `registry merge KEEP DROP` folds DROP's characters into KEEP's lowest-id character (deleting DROP's characters and remapping character_id as well as actor_id in assignments.json); a later flag can keep separate characters for an actor playing two roles. — Merge exists to fix one person split in two, who should get one English voice; the plan's "moves characters" left two. — If wrong: the old behavior returns behind a flag.
Ruling R13: C1 fix: when an episode's exemplars are removed (rematch or the R11 path), also delete actors whose exemplars all came from that episode, unless locked, named, or voiced; episode exemplars are rebuilt in one transaction per run. — Prevents self-matching at cosine 1.0. — If wrong: an unnamed actor seen only in one episode is recreated on rematch (same result, new id).
Final fix wave dispatched (one implementer, opus).
Final fix wave: DONE_WITH_CONCERNS (261609e..ad6fbd1, 13 commits; 195 passed). Accepted: named-character actors spared (within R13 'named'); id_high_water table prevents id reuse; pyright hook runs locally via uv; pydoclint skips short docstrings.
Final fix wave re-review (opus): all findings addressed, no new Critical/Important. New minors: final prune can orphan a review row's actor (near-unreachable); rematch churns ids for an episode's own new actors (read the sweep by joins, not change counts). T14 two-clusters-one-source case still deferred.
Branch state: code complete and reviewed; merge held until Task 17 steps 4-5 (operator labels, A3).
Note: subtitle survey (~250 shows) found no consecutive named-speaker pair; single-episode named ASS in Fairy Tail 100YQ S01E11 and QA in Another World S01E01. A3 stays on hand labels. Candidate for a later plan: eval speakers --from-subtitles (named ASS lines -> labels).
D4 ruling: rename doppio -> fukikae. Rename dispatched after this. Web subtitle search for Chimera speaker-labeled subs dispatched.
Rename dispatched (afbc733a3712f923f, sonnet; BASE ad6fbd1). Logs ideas in docs/ideas.md: face/active-speaker detection, video labeling clips, subtitle-derived labels. Controller moves ~/doppio-ws -> ~/fukikae-ws (.doppio -> .fukikae, doppio.toml -> fukikae.toml) after the web search finishes, then the repo dir.
Web subtitle search: no speaker-labeled Chimera subs; embedded tracks are Viki plain; subtitlecat AppleTor English plain (+7 s); subscene/subdl/opensubtitles blocked (Cloudflare/PoW). Queued: browser MCP key fix (action), video labeling clips (decision).
Rename done (a005542 refactor!, 7e8819e docs/ideas.md). Controller moved ~/doppio-ws -> ~/fukikae-ws (.doppio -> .fukikae, doppio.toml -> fukikae.toml, source path updated) and the repo to ~/projects/fukikae (uv sync re-run). Verified: S01E01/S01E02 `fukikae run --to characters` re-ran only extract (identical audio stopped propagation); registry intact, 9 actors.
Minor (deferred): a test or eval step may write E0*.srt into the cwd (rename agent found six in the repo root); check eval asr's temp srt path.
Task V1 (video labeling clips, D5) dispatched (BASE 7e8819e, implementer a76a5bf9467842106, sonnet); writes ~/fukikae-ws/chimera/labels-video. A4: browser MCP still down; entry is ~/.claude.json user-level mcpServers.lab-browser-stealth.
Task V1: implementer DONE (4557d17); 102 mp4 clips (14M) in ~/fukikae-ws/chimera/labels-video. Controller reinstalled git hooks at the new path (old INSTALL_PYTHON pointed at ~/projects/doppio/.venv); pyright 0 errors with all extras.
Task V1: complete (commits 7e8819e..4557d17, review clean). Alignment check: both Chimera sources have video and audio start_time 0.000, so vocals line up with picture.
Task V1: minor (deferred): --force removes old clips before sampling succeeds; video mode still reads vocals_16k unused; missing vocals.wav gives a raw MediaError; non-zero start_time sources untested (would drift).
Pivot: milestone test material moves to Crunchyroll-subtitled anime (CR ASS fills Name field on every line; library E11 of Fairy Tail 100YQ proves it). Stealth-browser agent dispatch for Subscene/OpenSubtitles (Chimera) refused by auto mode; not retried. Plain-fetch AnimeTosho agent dispatched; Sonarr CR WEB-DL grab queued as fallback action. Chimera labeling on hold.
AnimeTosho plain fetch: no subtitle attachments served for 10 E12 releases; CR ASS only inside full releases. A5 (Sonarr grab) is now the path.
Operator added Fairy Tail episodes in CR formats; survey agent dispatched to find consecutive runs with named ASS.
Ruling R14: milestone 4 test material moves to Fairy Tail: 100 Years Quest S01E12+S01E13 (same WEBDL-1080p release, jpn audio, CR ASS with 100% named lines; E14 and E11 as extra episodes); ground truth from ASS Name fields via `eval auto-label` (Task V2), replacing hand labels. Chimera stays as the ASR result. — Operator direction; automatic labels remove the labeling bottleneck. — If wrong: Chimera hand labels remain possible (video clips exist).
Copied E11-E14 to ~/fukikae-ws/media/Fairy Tail - 100 Years Quest/Season 1 (4.3G, read-only volume, removed). Workspace ~/fukikae-ws/fairytail (ja, qwen3_asr for ja, gpu 0). Pipeline E12->E13->E14 through characters running in background (log ~/fukikae-ws/fairytail/run.log). Task V2 dispatched (aaf8fd64973503186, sonnet; BASE 4557d17).
Task V2: implementer DONE (7e94e47). Note: .venv entry scripts for pytest/ruff carry stale doppio shebangs; uv sync --reinstall after the background run.
Task V2: review: 2 Important (speaker-name variants not merged; /I inner-thought names dropped). Fix round 1 sent (FIX_BASE 7e94e47).
Ruling R15: auto-label canonicalizes names (whitespace, parenthetical spacing), strips /I, /OS, /OC, /VO and (V.O.)/(O.S.) suffixes, applies operator aliases (--alias and <workspace>/labels-aliases.yaml), and only suggests near-duplicates; MAKAROV (WRAITH) stays distinct from MAKAROV. Controller sets NABARU: NEBARU in ~/fukikae-ws/fairytail/labels-aliases.yaml after checking E12-E14 spellings. — Ground truth must not split one character over spelling. — If wrong: an alias line added or removed.
Task V2: minor (deferred): Line.text unused; generic roles (WIZARD A/B, YOUNG *) labeled as distinct characters; IGNORED_NAMES 'ed' could swallow a character; min_overlap boundary untested; choose_subtitle_track leaves temp subtitle files on mid-loop exception. Labels cover the 50-span cap per cluster (diarize artifact limit).
Task V2: fix round 1/5 (2 addressed, 0 open; commits 7e94e47..291171a). S01E12 now keeps 354 named lines.
Task V2: complete (commits 4557d17..291171a, review clean)
Task V2: minor (deferred): empty alias values in labels-aliases.yaml become "None"/"" targets (CLI --alias rejects them); malformed YAML gives a traceback, not "error:".
