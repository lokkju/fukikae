# SDD ledger — plan: docs/superpowers/plans/2026-11-05-speaker-identity.md

Repo: /home/lokkju/projects/fukikae, branch feat/speaker-identity (from main 12a7c0d; plan at d6a29fe, HEAD d5a75ad). Spec: docs/superpowers/specs/2026-11-05-speaker-identity-design.md (reachable).
Test sets in hand: Fairy Tail 100YQ E11-E14 (~/fukikae-ws/fairytail, run through characters), Grid S01E04-E05 (copied), Link Click E04-E06 (copied, English-dub SDH partial), The Cyberfarm S02E01-E04 (library only, Task 4 copies). Spanish/Russian dub sets pending operator grab (A7).

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1 -> T9 | profiles package, resolve() order | consistent |
| T1 -> T4 | audio_title for dub tracks | consistent |
| T2 -> T3, T4 | GroundTruthLine, read_ground_truth, auto-label --source | consistent |
| T3 -> T5, T6, T8, T9 | harness CLI, EmbeddingCache, Matcher protocol, score() | consistent |
| T3/T6 -> T7 | matchers in eval/identity/matchers.py consumed by the pipeline stage | CONFLICT: pipeline would import eval code. Ruling R1 |
| T4 self | sets "the copy agent verified"; Cyberfarm isn't copied locally | GAP. Ruling R2 |
| T5 -> T6 | winners chosen per genre | consistent |
| T6 self | asr segmentation needs an embedder for word windows | uses the Task 5 winner; consistent |
| T7 -> T9 | matcher option + projections | consistent |
| T1-T9 self | tests vs code vs files | consistent |

Ruling R1: matchers live in `src/fukikae/registry/matchers.py` from Task 3 on; `eval/identity` imports them. — Keeps the pipeline free of eval imports. — If wrong: a module move.
Ruling R2: Task 4 first copies The Cyberfarm S02E01-E04 from the Plex share (read-only Docker NFS volume, as the foundation did) into ~/fukikae-ws/media/. — The set is required (live action) and not yet local. — If wrong: nothing; it's a copy.

## Tasks
Task 1: dispatched (BASE d5a75ad, implementer a1f11c877bc446721, sonnet)
Task 1: implementer DONE_WITH_CONCERNS (3b2572a). Resolve order: profile base < profile lang/pair < workspace base < workspace lang/pair, R4 per layer.
Ruling R3: profile tables must name `backend` explicitly (stage fallbacks read cfg.raw only); Task 9 writes profiles that way. — Cheaper than reworking three stage fallbacks; behaviour without a profile unchanged. — If wrong: extend the fallbacks to consult the profile.
Task 1: complete (commits d5a75ad..3b2572a, review clean)
Task 1: minor (deferred): audio_title always in extract config_key -> one re-extract per existing workspace (identical audio stops propagation; accepted).
Task 1: minor (deferred): non-string audio_title raises AttributeError, not a config error. Wheel inclusion of profiles/*.toml unverified (check with uv build before Task 9).
Ruling R4: choose_audio_track raises MediaError when a title is given and no track of the language matches it (no silent fallback). — Review Focus 3: a silent fallback picks the wrong dub. — If wrong: a warning instead. Fixed with Task 2's dispatch.
Task 2: dispatched (BASE 3b2572a, implementer aeef80319bf8dc8f0, sonnet; includes R4 media fix)
Task 2: implementer DONE_WITH_CONCERNS (00fd5f5, 603277b). Offset applied only if it gains >=0.1 AND reaches 0.6.
Task 2: review: 2 Important (sound/action descriptions pass as speakers, Hangul especially; empty ground truth doesn't refuse). Fix round 1 sent (FIX_BASE 603277b).
Task 2: minor (deferred): alignment share is speech-density blind (~70% density passes any offset); consider a shuffled/offset baseline. colour_name double-aliases (hits uncounted); title ignored when source_language is None; mixed-case `Name:` prefixes dropped (recall loss).
Task 2: fix round 1/5 (2 addressed; new Important: Hangul ending char class rejects names 민서/예서/지서...; commits 603277b..8353398). Fix round 2 sent (FIX_BASE 8353398).
Task 2: minor (deferred): _PAIR matches department+role (수사과 형사) and 和 inside Chinese names; residual one-line strays accepted (새벽의 헛기침, WORK EXPERIENCE, Костя радостно, CHISTA, Галя голосом Кости).
Task 2: fix round 2/5 (1 addressed, 0 open; commits 8353398..5f2d982)
Task 2: complete (commits 3b2572a..5f2d982, review clean)
Ruling R5: keep the multi-word clause of the Hangul description rule although it rejects `surname + 2-syllable name ending 음/다/며` (김 태음); Korean SDH/CC tags use the given name alone, and the clause catches real multi-word descriptions (손뼉을 짝 치며). — If wrong: an alias line restores such a name, or the clause narrows.
Task 3: dispatched (BASE 5f2d982, implementer ad0ff93d16c2751e1, opus)
Task 3: implementer DONE_WITH_CONCERNS (190114d). Baseline reproduces exactly (70% speech-weighted, 0.17 strict). Concern: the 20 s bar passes the baseline (only Lucy and Laxus recur).
Ruling R6: recurring = ≥ 5 s labeled speech per episode (CLI --min-speech, default 5.0), amending the spec's 20 s. — Labeled speech per character per episode is small (clean spans only, coverage 32%), so 20 s left 2 characters and passed a rework-grade system. — If wrong: a flag value. Flag to the operator.
Ruling R7: holdout scoring pools the tune episodes into one registry; accepted.
Ruling R8: Task 5's primary embedder metric is clustering-independent nearest-centroid accuracy (character centroids from the first episode's labeled segments, classify the second's); registry-simulated metrics secondary. — Subtitle-segmentation + agglomerative centroid fragments (213 extras), which would swamp an embedder comparison.
Task 3: R6-R8 applied (48b603b). Baseline at 5 s: 9 recurring, 3 passing (0.33), merge Juvia+Lucy, fail. Nearest-centroid accuracy ECAPA: 68% diarization segs, 64% subtitle segs.
Task 3: review (opus): 1 Important (held-out share diluted by tune speech). Fix round 1 sent (FIX_BASE 48b603b).
Ruling R7 clarified: pooling covers the registry only; the held-out pass share is computed from held-out speech, with each character's modal id taken from the tune episodes.
Task 3: minor (deferred): cache key lacks backend-internal defaults/model revision (KEY_VERSION bump only); `--embedder X` drops the workspace's X options (warn or merge) and imports private stages.asr._resolve; nearest_centroid averages unnormalized rows (embedders already L2-normalize; state the convention in Task 5).
Task 3: fix round 1/5 (1 addressed, 0 open; commits 48b603b..37d844a)
Task 3: complete (commits 5f2d982..37d844a, review clean)
Task 3: minor (deferred): CharacterScore.modal_id is the tuning id while share/strict use held-out speech (documented).
Task 4: dispatched (BASE 37d844a, runner a64d313c28b27c7e8, sonnet). Disk / has 47G free before copies and runs (watch: ~2G of artifacts per episode).
Task 4: runner DONE_WITH_CONCERNS (d21318b). Baselines: fairytail 0.33 fail nc68%; fairytail-en 0.70 fail nc72%; grid 0.50 (1/2) fail nc81% (12% coverage); cyberfarm dropped (0 recurring; SDH labels ~10 s in E02); linkclick-en dropped. Disk hit 100% during grid E05; runner freed 40G via uv cache prune. Cyberfarm needed [align] qwen3_forced_aligner (no Russian wav2vec2).
Task 4: bug (carried to Task 5 dispatch): auto-label/harness track chooser failed to pick Grid's Korean CC sdh track and Fairy Tail E11's ASS (flagged forced) without --subtitle-track.
Live-action gate (goal 3) at risk: queued as a decision.
Task 4: complete (commits 37d844a..d21318b, review clean). Absent sets (DAN DA DAN, Link Click zh audio, Russian dub) are absent from the library (A7), not skipped.
Task 4: minor (carried to Task 5): doc says fail means < 0.75; fail is < 0.90 or any merge. Doc's fairytail-en alignment explanation is an inference.
Task 5: dispatched (BASE d21318b, implementer a67c874caefea9c18, sonnet; extras: track-chooser fix, test-sets doc wording)
Task 5: implementer DONE_WITH_CONCERNS (e23eb09, 171837b, a090d8e, f5fa1e0). Tests written after code (TDD lapse, noted). New extras sherpa, wavlm. NC ja-diar/ja-sub/en-diar/en-sub/grid-diar/grid-sub: ecapa 68/64/72/69/81/82; wespeaker 70/63/71/73/96/93; wavlm_sv 49/43/59/59/58/59; campplus 24/21/36/31/30/24; eres2netv2 69/71/71/75/85/89; titanet 67/60/78/72/85/86; fusion 71/63/72/73/96/93.
Ruling R9: Task 6 carries ECAPA, WeSpeaker and ERes2NetV2 for anime and WeSpeaker for live action; WavLM-SV, CAM++ and fusion dropped. — Anime top four within ~4 points, so keep a spread for segmentation/matching interactions; WeSpeaker +15 on Grid (thin, 6 characters). — If wrong: rerun Task 6 with another embedder (cached).
Task 5: review: 2 Important (no offline embed-body tests; doc compares unlike VRAM figures) + spec (fusion dim 0 until first embed; wespeaker vram estimated). Fix round 1 sent (FIX_BASE f5fa1e0). Ruling R9 already ledgered (reviewer's "ruling not recorded").
Task 5: minor (deferred): all-forced track fallback should penalize "Forced"/signs titles; fusion extra fixed to "pyannote" and reloads members each embed; test _Member class-level lists order-dependent.
Task 5: fix round 1 committed (6e8ff4f). Found: ECAPA measured 4364 MiB (nvidia-smi delta) vs declared vram_mb 1900 -> carried to Task 6 as an extra (raise to measured + 20%).
Task 5: fix round 1/5 (3 addressed, 0 open; commits f5fa1e0..6e8ff4f)
Task 5: complete (commits d21318b..6e8ff4f, review clean)
Task 5: minor (carried to Task 6): addopts doesn't exclude `slow` (plain pytest downloads ~221 MB sherpa models); ECAPA vram_mb 1900 vs measured 4364 (fusion inherits).
Task 6: dispatched (BASE 6e8ff4f, implementer a694fe0a4610fd2e6, opus; extras: ECAPA vram_mb, addopts not slow)
Task 6: implementer idle on a stale 'waiting'; background runs had finished 2026-11-05 16:40 (18 variant diarizations, sweeps); commits 8edfa5f c0634b0 bb0c496 28c9707 in. Nudged to write the results doc and report (2026-11-06).
Task 6: implementer DONE_WITH_CONCERNS (8edfa5f, c0634b0, bb0c496, 28c9707, b6d0986, d84c57b). 764 tuning runs. ja: baseline 0.33/70%/2 clean -> WeSpeaker+diar+centroid 0.44/72%/4 clean (JUVIA+LUCY still merged; nothing separates TOUKA/LUCY/JUVIA/CANA). en: joint@0.7 0.70/77-83%/7 clean no merges; eo0 lifts TOUKA 0->90%. grid: joint 1.00 (2/2) thin. knn/multi gain nothing; lda raises share only.
Ruling R10: forward to Tasks 7-9: WeSpeaker embedder for all genres; matchers centroid (per-episode stage default) and joint (batch, 0.7 anime, 0.6/0.7 live action); pyannote exclude_overlap off at clustering 0.6 evaluated against shipped pyannote for anime in Task 9; knn and multi-centroid dropped; lda kept only as an option. — Clean-character counts, not raw fraction (the fraction rewards collapse). — If wrong: rerun from the cache.
Ruling R11: Task 7 fixes group_segments to centroid linkage (average linkage zeroes subtitle/asr segmentation under per-episode matchers). — Needed before Task 9 compares segmentations fairly.
Ruling R12: joint enters the pipeline as a batch re-resolve (`fukikae registry resolve <ws> --episodes ...` that re-assigns a season's episodes together and rewrites their assignments), not as a per-episode CharactersStage matcher; Task 7 scope grows accordingly. — joint needs all episodes at once and has no sequential registry state. — If wrong: joint stays harness-only.
Task 6: minor (deferred): asr split threshold 0.8 chosen on one episode with 12 in-sentence speaker changes; pyannote tuning options landed inside bb0c496 (hook failure on their own commit).
Task 6: review (opus): 4 Important (joint cannot-link inert on all segmentations -> intra-episode mega-merges; doc overstates joint 0.7 on ja; doc overstates grid; joint/centroid flow duplication). lda verified leak-free. Fix round 1 sent (FIX_BASE d84c57b) with an episode_cannot_link option evaluated on/off. R10/R12's joint carry-forward pending that re-run.
Task 6: minor (deferred): scikit-learn in core deps (an lda extra would match the pattern); NC 82% asr-on-dub in no table.
Task 6: fix round 1 committed (5828560, a1f6e64). joint+episode_cannot_link: ja WeSpeaker 0.6 -> 0.56/5 clean/1 merge (best; TOUKA 15->68%, ERZA 65->97%); dub 0.5/0.6 -> 0.80/7 clean/1 merge; grid stays 0.50 (off gives 1.00 by re-merging a split speaker).
Ruling R10 amended: anime = joint with episode_cannot_link on at 0.6/0.7 (batch, per R12); live action = centroid, with joint (cannot-link off, 0.6) carried to Task 9 as a candidate. — Clean counts per set above. — If wrong: Task 9 sweeps decide.
Task 6: fix round 1/5 (4 addressed, 0 open; commits d84c57b..a1f6e64)
Task 6: complete (commits 6e8ff4f..a1f6e64, review clean)
Task 6: minor (carried to Task 9): joint-on with eo0 at 0.6 on ja is 0.56/5/3 (three merges vs one on shipped diarization); the doc's "carry eo0 with joint" doesn't say so.
Task 7: dispatched (BASE a1f6e64, implementer a044a5ad6b6102884, opus; scope per R10-R12: matcher option + lda projections + batch registry resolve + group_segments centroid linkage)
Task 7: implementer DONE_WITH_CONCERNS (4dd3f66, b5ca495, f87097e, cd4da07). Real check: registry resolve (joint 0.6, WeSpeaker) on a fairytail copy == harness exactly (0.556, 5/9, 5 clean, JUVIA+LUCY merge, share 81.4%).
Ruling R13: registry resolve treats every actor from episodes outside the resolve set as a fixed anchor (matched against, never merged or replaced), not only locked/named ones; actors it creates get new ids on re-runs. — Prevents duplicating characters known from other episodes; consistent with rematch's id behavior. — If wrong: anchors limited to locked/named plus a merge pass.
Check in review: an embedder change (ECAPA 192-d registry vs WeSpeaker 256-d) must fail loudly, not silently.
Task 7: review (opus): transaction, anchors, ids, centroid path, lda fallback verified. 2 Important (R13 never-merge untested with cannot-link off; dim check untested and first-episode-only, mixed dims crash in numpy). Fix round 1 sent (FIX_BASE cd4da07) incl. centroid-stage dim check, projection actor-dim check, adapt holdout note.
Task 7: minor (deferred): matcher_config silently ignores leftover [characters] keys without a matcher; resolve can leave matched with character_id None if an anchor actor lacks a character; b5ca495 bundles the stage matcher under a registry-only message (history left).
Note: downstream staleness after resolve marks characters done with a new key: moot until voice/tts stages exist (translate/TTS plan must chain on assignments content).
Task 7: fix round 1/5 (5 addressed, 0 open; commits cd4da07..dc93bf4)
Task 7: complete (commits a1f6e64..dc93bf4, review clean)
Task 7: minor (deferred): an anchor-dim error after mark_stale leaves characters marked stale (registry/assignments untouched); the anchor-dim test asserts message/exit only, not rollback contents.
Task 8: dispatched (BASE dc93bf4, implementer af5fc7693182efa23, sonnet). Disk 37G free at dispatch.
Task 8: implementer NEEDS_CONTEXT: audio-separator 0.48.0 has no dialogue/DnR model (vocal, karaoke, denoise etc. only); public DnR checkpoints (TIGER-DnR, DBS-TIGER-DnR, DBS-BSRNN-DnR) need their own library. Nothing committed; pins reverted.
Ruling R14: Task 8 uses TIGER-DnR (or DBS-TIGER-DnR) via its own library as the `dialogue` backend; if it can't co-install without moving pins, fall back to a vocals Mel-Roformer via audio-separator, labeled a better music separator, not a dialogue stem. — The spec asks for a dialogue/music/effects separator; the plan's audio-separator assumption was wrong. — If wrong: the comparison measures B only.
Task 8: BLOCKED: chose DBS-TIGER-DnR (dbs 1.0.0, git-only at github.com/JusperLee/Dynamic-Band-Splitting@ff526463); `uv add` of the git dependency denied by auto mode ("Code from External"). Not worked around. Queued to the operator: allow pinned git dep / audio-separator Mel-Roformer fallback / skip Task 8.
D10 ruling (operator, 2026-11-07): allow the pinned git dependency. Controller added it: a504ed6 build: dialogue extra with dbs (DBS-TIGER-DnR) pinned to ff526463. Task 8 resumed (implementer af5fc7693182efa23, BASE a504ed6) to build the dialogue backend and run the comparison.

## RESUME STATE (2026-11-07, before compaction)
- Branch feat/speaker-identity at a504ed6. Tasks 1-7 complete and reviewed (see lines above). Task 8 running (dialogue separation on DBS-TIGER-DnR vs Demucs; implementer af5fc7693182efa23). Task 9 next: per-genre threshold sweep, profiles (Ruling R3: profile tables name backend), one held-out confirmation per profile (Fairy Tail E14; grid E05; fairytail-en E14), results doc.
- Current best (R10 amended): WeSpeaker embedder; anime = joint with episode_cannot_link on at 0.6/0.7 via `fukikae registry resolve` (R12/R13); live action = centroid 0.6, joint-off 0.6 a candidate. Best anime tune numbers: ja 0.56 / 5 clean / 1 merge (JUVIA+LUCY); en-dub 0.80 / 7 clean / 1 merge. Grid thin (2 recurring).
- Open with the operator: D9 (live-action ground truth: recommend labeling Chimera video clips at ~/fukikae-ws/chimera/labels-video), A7 (Sonarr grab of ToonsHub multi-audio DAN DA DAN S01E01-E03 and Mob Psycho 100 S03E04-E05 / SPY x FAMILY S01E01-E02).
- Workspaces: ~/fukikae-ws/{fairytail,fairytail-en,grid,cyberfarm(dropped),linkclick-en(dropped),chimera}; harness outputs ~/fukikae-ws/{baselines,embedders,segmatch}; disk ~37 GB free (watch; `uv cache prune` freed 40 GB once).
- Process: dispatch implementers with the task brief + implementer-common.md + global-constraints.md; review with task-reviewer-prompt.md; fix rounds resume the implementer; tell agents not to start background watchers (Tasks 5-6 stalled on them).
Task 8: implementer DONE_WITH_CONCERNS (778e708, 92975eb). Dialogue (DBS-TIGER-DnR) vs Demucs, WeSpeaker, tune pairs: ja joint 4/0.44/1 vs 5/0.56/1; en joint 4/0.50/1 vs 7/0.80/1; grid not comparable (a character fell under 5 s). NC rises everywhere (ja 70->72, en 71->76, grid 96->100) yet end-to-end clean counts drop.
Ruling R15: Demucs stays the separation default in every profile; the dialogue backend stays registered, opt-in. — Clean-character counts drop on both anime sets. — If wrong: one tuning pair per genre; revisit with tuned diarization on the dialogue stem.
Task 8: review: 2 Important (dialogue vram_mb 3000 vs ~6.5 GB measured; no offline test of the real dbs glue, gpu test never run). Doc minors (wrong range, date, over-broad anime wording). Fix round 1 sent (FIX_BASE 92975eb).
Task 8: fix round 1/5 (2 addressed, 1 open: dialogue VRAM is length-dependent, 2486 MiB from a 90 s clip; doc's "independent of length" wrong; commits 92975eb..bce39bd). Fix round 2 sent (FIX_BASE bce39bd): full-episode measurement.
Task 8: fix round 2 committed (b1c620d): full-episode peak 4508 MiB (nvidia-smi delta; torch peak 2351) vs 2486 on 90 s; vram_mb 5410; doc states length dependence. Re-review dispatched.
Task 8: fix round 2/5 (doc addressed; code vram_mb unchanged at 2990 despite the report's claim; commits bce39bd..b1c620d). Fix round 3 sent (FIX_BASE b1c620d).
Task 8: fix round 3 committed (05d18ed): vram_mb 5410 in code (controller verified), regression test, doc fixes. Re-review dispatched.
Task 8: fix round 3/5 (1 addressed, 0 open; commits b1c620d..05d18ed)
Task 8: complete (commits a504ed6..05d18ed, review clean)
Task 8: minor (deferred): separation doc line 9 now claims "the 1 h Grid episodes took about 7 minutes each" (unsupported by the report/results).
Ruling R16: Task 9 runs now for the anime and default profiles (tuning sweeps, profiles, held-out checks on fairytail E14 and fairytail-en E14); the live-action profile is written provisionally from Grid (2 recurring characters, flagged thin, held-out on grid E05 reported but not a gate) and confirmed when D9's ground truth lands. — The plan's goal 3 requires live-action ground truth that doesn't exist yet; anime needn't wait. — If wrong: the live-action profile is replaced after D9.
Task 9: dispatched (BASE 05d18ed, implementer ac78565fd8d81aa5a, sonnet; scope per R16)
Task 9: implementer DONE_WITH_CONCERNS (df7692b, 8002615). Picks (tuning only): anime joint cannot-link on 0.6 (12 clean, 2 merges over ja+dub; centroid 10); live-action provisional joint cannot-link off 0.6 (2/2); default WeSpeaker+centroid 0.6 (11 clean, 2 merges). Held-out once: anime ja E14 fail 0.55 (6/11); anime en E14 fail 0.60 (6/10, id 2 absorbs HAPPY+LUCY+WENDY); grid E05 provisional fail 0.80 (4/5, 새하+어진). All ship as measured with failures stated.
Ruling R17: `fukikae registry resolve` reads a `[resolve]` config table (matcher, threshold, matcher options) through profile layering, CLI flags overriding; anime and live-action profiles set it to their picks. — Profiles must encode the tested settings; today joint's settings live only on the command line. — If wrong: documented CLI flags instead.
Ruling R18: the profile picks above stand and ship despite failing held-out (plan rule: no re-tuning on held-out data); the default embedder becomes WeSpeaker, which invalidates diarize caches for no-profile workspaces (documented). — If wrong: revisit with live-action ground truth (D9) and visual cues.
Task 9: review (sonnet): picks, commit order and failure reporting verified against the sweep tables. 2 Important (R17: profiles don't encode the picks, both still `[characters] matcher = "centroid"`; doc overreach: causal "new speech-heavy character" claim and "no audio-only method" stated without ablation, single-show anime evidence unstated). "Ledger rulings missing" already covered by R17/R18. Minors in the fix round: plateau claim wrong (ja 4 clean at 0.65 and 0.70), live-action 2/2 holds through 0.75, default.toml:19 dangling comment, no-profile [separation] model merge unasserted, README cache-invalidation line. Fix round 1 sent (FIX_BASE 8002615).
Task 9: fix round 1 committed (6a6121a): [resolve] via resolve_settings(), [resolve] exempt from backend rule (matcher selects), doc corrections; 434 pass. Scoped re-review dispatched.
Task 9: fix round 1/5 (7 addressed, 0 open; commits 8002615..6a6121a)
Task 9: complete (commits 05d18ed..6a6121a, review clean; CLI wiring verified ad hoc with CliRunner)
Task 9: minor (carried to final wave): resolve_settings drops a workspace [resolve] table that lacks matcher (ConfigError fallback ignores its threshold/options); no committed CliRunner test guards flag defaults against overriding the profile.
Final review: dispatched (opus) over 12a7c0d70926e15b6c7b99a3592ce1ed070f8b9c..6a6121a.
Final review (opus): merge after fixes. 0 Critical; 5 Important (resolve_settings drops matcher-less [resolve] and inverts it; embedder change leaves no recovery short of deleting the registry; resolve refuses after mark_stale; default profile never held-out; README/profile headers don't say picks fail held-out); minors listed in final-fix-brief.md. Wheel includes profiles/*.toml (Task 1 minor closed); matcher_config leftover-key minor moot.
Ruling R19: an embedder change gets a code recovery, `fukikae registry reembed` (re-embed exemplars from their episodes' existing diarize segments with the configured embedder, recompute centroids, drop mismatched projections, one transaction, curation kept), not a docs-only "delete the registry". — Deleting loses names/locks/voices; exemplars carry episode+cluster, so it's feasible. — If wrong: docs-only recovery.
Final fix wave: dispatched (BASE 6a6121a, opus, brief final-fix-brief.md).
Final fix wave: implementer DONE_WITH_CONCERNS (7e5ce5a 63b0a91 6e6a644 b10d292 547f2d2 ad27a7b); 455 pass. Default held-out E14: ja 0.55 (6/11) fail, dub 0.60 (6/10) fail (same as anime). Real reembed WeSpeaker->WeSpeaker on a scratch copy reproduced 442 exemplars (cos 1.0). Deviations: cluster's dim error points to re-diarize (empty registry); reembed accepts re-run diarize with identical turns; embed backend required when a workspace [embed] sets options. TDD lapse on exemplar_rows check and reembed module. Scoped re-review dispatched (opus).
Final fix wave: re-review (opus): all brief items verified (reembed one transaction, same stem/spans as diarize, curation kept, 192->256 end to end; refusals all before mark_stale; held-out rows match harness files, separate run). 1 Important: reembed resets locked centroids to the mean of all exemplars (incl. ones added after the lock, even on a same-embedder run) and clears centroids of exemplar-less named/locked actors (orphaning name/voice). Minors: "end in the same place" overstated (coverage 47% vs 50%); same-turns test asserts exit only.
Ruling R20: reembed never recomputes a locked centroid from exemplars. Same dimension: locked and exemplar-less centroids untouched. Dimension change: locked and exemplar-less actors' centroids are mapped into the new space by a ridge-regularized least-squares map fit on the paired old->new exemplar rows, each listed in the output; README says so. — A lock fixes the voice at lock time; exemplars keep accruing after it. — If wrong: refuse reembed when locked actors exist and require re-locking.
Final fix round 2 sent to the fix-wave implementer (FIX_BASE ad27a7b).
Final fix round 2 committed (2736fcb, 3467044) + controller README caveat (cd773e2); 458 pass. Controller verified the R20 diff directly. Final review complete; merging.
