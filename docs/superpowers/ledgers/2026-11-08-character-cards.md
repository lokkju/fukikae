# Character-cards plan: execution ledger

Plan: `docs/superpowers/plans/2026-11-08-character-cards.md`. Facts and numbers from each task's real runs.

## Task 4: translate, ASR path, managed llama-server

Run 2026-11-08 on a copy of the Fairy Tail workspace (E11, ja to en) at `~/.claude/jobs/d640b8ae/tmp/ft-ws`, never the original.

### llama-server and the model

- `github:ggml-org/llama.cpp` build `b12201`, asset `llama-b12201-bin-ubuntu-cuda-12.8-x64.tar.gz`. That asset alone lists no CUDA device: it loads `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12` from its own directory (RUNPATH `$ORIGIN`), and the release ships them as the separate asset `cudart-llama-b12201-bin-ubuntu-cuda-12.8-x64.tar.gz`. mise's `postinstall` runs `.mise/scripts/llama-cudart`, which downloads that asset, checks its sha256 (`db17380427c3431e290ed1654d9eafbf4f73f523f431cf1a869929a5161fa82a`) and puts the three libraries next to `llama-server`. With them, `--list-devices` under `CUDA_VISIBLE_DEVICES=0` shows exactly one device, the 3090 at nvidia-smi index 0. The Vulkan asset runs too, but `CUDA_VISIBLE_DEVICES` doesn't restrict it (both cards are listed), so I didn't use it.
- Model `Qwen/Qwen3-8B-GGUF`, file `Qwen3-8B-Q6_K.gguf` (6.7 GB), revision `7c41481f57cb95916b40956ab2f0b139b296d974`. llama-server held 7.5 GB on GPU 0 while it ran (`vram_mb` default 10000).

### Runs

The workspace copy's separate, asr and diarize keys were stale against today's code (GT6 identity keys), so `fukikae run` re-ran them on GPU 0 (250 s). Its registry was ECAPA (192-d) against today's WeSpeaker clusters (256-d), so the copy runs on a fresh registry; characters and tag (SenseVoice, CPU) took 325 s. Character 1 (195 s of speech, the most) got a stock test card with a `[style]` table and a catchphrase; no character has a name, so speakers are `char<id>` and `extra:<cluster>`.

| Run | Cues | Translated | Errors | Over budget after retry | Hit rate | Wall time (translate stage) |
|---|---|---|---|---|---|---|
| 1 (b1238c5) | 536 | 536 | 0 | 161 | 70.0% | 482 s |
| 2 (818bfc1) | 393 | 393 | 0 | 23 | 94.1% | 315 s |
| 3 (967d815, fix round 1) | 393 | 393 | 0 | 22 | 94.4% | 310 s |

Run 1 showed 105 cues of at most two characters: Qwen3's aligner stamps each character with about 20 ms, and a diarization turn boundary falling inside a word cut off one character as its own cue ("ロ" | "ックと..."). The same 20 ms stamps made every slot end too early, so budgets were short. 818bfc1 folds such fragments into the neighbour of the same ASR segment and lets a point-stamped word sound until the next word, at most 0.3 s. Run 2's 23 overruns are a median of 3 characters over (13 of them 3 or fewer), the largest 12.

- 34 cues carried a delivery direction, which is all 34 cues with a non-neutral emotion tag.
- No cue was skipped as song: SenseVoice found no `singing` span on this episode's vocals (events: bgm, crying, laughter only), so the opening theme's lyrics were translated as dialogue.
- `nvidia-smi --id=0` showed no process after each run. An accidental `timeout 170` kill of the CLI mid-run also left no llama-server behind.
- Run 3 (after fix round 1): no window was marked. There were no format or backend retries, and no prompt had to shrink. The 22 overruns are a median of 4 characters and at most 16, each flagged per cue (`over_budget`, `overrun`). The 34 directions all pass the four-word plain-tag rule. Lone interjections stay as their own cues ("え？" c0068, c0296). llama-server was ready 3.3 s after starting (GGUF cached), and no key file or process was left after the run.

## Task 5: subtitle-first path with condense mode

Run 2026-11-08 on a copy of the Fairy Tail workspace (E12, ja audio, Crunchyroll's English ASS) at `~/.claude/jobs/d640b8ae/tmp/t5-ws`, never the original. `[translate] source = "auto"` took the subtitle path: no `[subs]` and no sidecar, so it picked embedded stream 4 (eng, ASS, default), passing over stream 3 (eng, "Forced", forced disposition, three sign lines). The copy's extract, separate, asr and diarize keys were stale against today's code and re-ran on GPU 0 (212 s); the original workspace's registry is ECAPA (192-d), so the copy uses Task 4's re-embedded E11 registry, and characters re-ran against it. No character has a name, so speakers are `char<id>` and `extra:<cluster>`.

| Cues | Spoken verbatim | Condensed | Over budget as written | Over budget after condense | Skipped | Unknown speaker | Errors | Translate stage |
|---|---|---|---|---|---|---|---|---|
| 357 | 343 | 11 | 12 | 1 | 3 (sign) | 3 | 0 | 11.7 s incl. llama-server start |

- Skipped: three signs, each with Name `SIGN`: the title card "A Card in the Hand" (`White-Standard`, `\pos`), "Next Time" and "Scarlet Showdown" (styles `Next Time`, `Next Ep Title`). This release subtitles neither the opening nor the ending and has no SDH notes, so no cue was dropped as song or SDH; those rules are covered by tests only. The five `Flashback - Top` lines (top alignment from the style, not an override) stayed dialogue.
- Condensed, for example: "I won't ever do that to you again." to "I won't do that again." (budget 32); "Do you have any idea how many times my potions have saved your ass?!" to "My potions saved your ass plenty times!" (64); "What's the matter?" to "What's up?" (15). One cue came back no shorter than the subtitle and speaks the subtitle line, flagged over budget for time fitting.
- Speaker against the ASS Name field (354 named spoken cues, 26 names): mapping each name to the label it most often gets, 284 agree (80.2%); each assigned label's most common name covers 221 (62.4%), because the characters stage merges voices (char1 is NEBARU 18, YOUNG MAKAROV 13 and five smaller roles; char2 is WENDY 11, HAPPY 8, NATSU 8, CHARLE 5). At the diarize-cluster level the same measure gives 281/354 (79.4%), so the overlap rule loses nothing against the clustering it reads from. NATSU's 40 lines go mostly to `extra:SPEAKER_14` (26) and ERZA's 20 to `extra:SPEAKER_11` (16): the characters stage left SPEAKER_14 (50 s of clean speech) at `review` and SPEAKER_11 (16 s) at `extra`, so those lines get the extras pool's voice, not a character's. ASR words broke three overlap ties; agreement is 281 with or without them.
- `nvidia-smi --id=0` showed no process after the run.

## Task 6: tts stage, Kokoro and stock assignment

Run 2026-11-09 on a fresh copy of Task 4's E11 workspace (`~/fukikae-scratch/t6-ft-ws`), never the original, with `[tts] backend = "kokoro"` and `device = "cpu"`. GPU 0 was left to the ground-truth plan; nothing in this run touched a GPU. Kokoro-82M pinned at `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987`.

- The copy's translate stage was stale only because a missing `glossary.toml` is fingerprinted with its absolute path (`job._fingerprint`), so a copied workspace re-runs translate. Its manifest key was rewritten by hand rather than re-running llama-server on GPU 0. The same was done once more after the card edit below, which kept the `[style]` unchanged.
- Character 1's card was Task 4's test card with `model_revision = "test"`. The stage refused it before loading anything, naming the character, the card and both revisions. Acting as operator on the copy, I re-pinned the revision with `voices edit`, which kept the style and made card `b072743c85843496` (locked). The refused run had already given characters 2 to 8 their automatic cards.

| Character | Cues | Median F0 (s read) | Estimate | Voice | By |
|---|---|---|---|---|---|
| 1 | 83 | 274 Hz (120 s) | f | am_adam | edit (operator) |
| 2 | 50 | 148 Hz (116 s) | m | am_michael | auto |
| 3 | 28 | 168 Hz (73 s) | u | af_nicole | auto |
| 4 | 20 | 255 Hz (44 s) | f | af_sarah | auto |
| 5 | 56 | 279 Hz (120 s) | f | af_heart | auto |
| 6 | 21 | 269 Hz (40 s) | f | af_sky | auto |
| 7 | 52 | 230 Hz (98 s) | f | af_bella | auto |
| 8 | 53 | 151 Hz (90 s) | m | am_fenrir | auto |
| extras SPEAKER_00, _02, _09, unknown | 9, 9, 9, 3 | | | af_alloy, am_echo, bf_alice, bm_daniel | extras pool |

| Run | Rendered | From cache | Renders | Wall time |
|---|---|---|---|---|
| 1 | 393 | 8 (repeated lines) | 671 (393 + 278 rate fits) | 368 s, CPU (~14.7 cores), 2.5 GB RSS |
| 2 | 0 | | 0 | "up to date" |
| 3 (character 2 to `stock:kokoro:bm_george`) | 50, all character 2 | 343 | 86 | 69 s |

- Rate fit: 294 of 393 renders were longer than their cue, and 279 were still longer after the 1.25 cap. Kokoro's English runs well over the Japanese slots: 800.6 s of speech for the episode. That is Task 7's to fit, flag or place whole.
- Character 1, the cast's loudest voice at a median of 274 Hz, has an F0 that would read as female. Characters 4 to 7 sit between 230 and 279 Hz. Without names or ground truth, nothing here confirms the anime estimates; the thresholds are adult-speech norms.

## Task 7: timefit, mix and mux

Run 2026-11-09 on fresh copies, never the originals: `~/fukikae-scratch/t7/e11-ws` (Task 6's E11 workspace, ASR path) and `~/fukikae-scratch/t7/e12-ws` (Task 5's E12 workspace, subtitle path, with `[tag]` sensevoice + emotion2vec and `[tts]` kokoro on cpu added). E11 re-ran translate (CD6/CD7 key change; llama-server on GPU 0), tts (163 renders), timefit and mix in 12 m 21 s. E12 ran tag, translate and tts first; its tts refused character 1's Task 4 test card (`model_revision = "test"`), re-pinned on the copy with `voices edit` as in Task 6 (card `b072743c85843496`), then tts (372 renders), timefit and mix in 9 m. A second `fukikae run` on each: "up to date". GPU 0 empty afterwards.

**Ruling candidate (song fallback):** a theme-song detector on the separation stems (`fukikae.songs`), independent of the tag. A second is song-like when the vocals are on in at least 60% of its 100 ms frames (above -40 dBFS) and the bed is within 3 dB of the vocals or louder. The best 80 s window in the first 360 s and in the last 240 s is a theme when at least 50% of its seconds are song-like; it grows while the next 5 s are 40% song-like, up to 100 s, padded 1 s. A fitted cue whose slot midpoint is in a song span is not placed, and the original vocals stay untouched there. Subtitle song cues (`skip = "song"`) and tag `singing` spans are song spans too; tag `bgm` spans only with `[mix] protect_bgm = true` (default false: on E11, 10 dialogue cues sit in SenseVoice bgm spans outside any song). Chosen over cross-episode OP/ED matching because it needs no other processed episode and found every theme measured.

| Episode | Detector spans (score) | Truth (lyrics or subtitle gap) |
|---|---|---|
| FT E11 | 65-147 (0.74), 1338-1421 (0.90) | 66-147, 1338-1420 |
| FT E12 | 83-165 (0.73), 1338-1421 (0.93) | gap 85-178, gap 1332-1424 |
| FT E13 (prototype) | 207-287 (0.72), 1339-1420 (0.92) | 210-~300, 1338-1422 |
| FT E14 (prototype) | 64-149 (0.94), 1334-1414 (0.70) | 64-152, 1335-1421 |
| Kaze E01-E03 (prototype) | head 0.69-0.80 | opening song 147-231 (E01), 44-119 (E02) |
| BBT en/de E01-E03, Grid E02/E04/E05 (prototype) | none (best 0.22) | no theme song of that shape |

E13 also has a 0.86 window at 611 s, mid-episode; the position limit keeps it out (unverified what it is).

| | E11 (ASR) | E12 (subs) |
|---|---|---|
| Rendered cues | 393 | 354 (3 signs skipped) |
| Over slot at the card's rate / rate-fitted / still over after the rate fit | 111 / 111 / 94 | 18 / 18 / 15 |
| Stretched (median, max combined with rate) | 41 (1.0225, 1.25) | 13 (1.0296, 1.25) |
| Borrowed (seconds) | 63 (9.3 s; max 0.62 s, c0197) | 5 (0.57 s) |
| Left long | 0 | 1 (c0329, 0.01 s: rate 1.25 used the budget, 0.35 s before, next cue adjacent) |
| Song spans / cues dropped | detector 65-147, 1338-1421 / 41 (all 16 opening and 25 ending lyric lines, c0015-c0030, c0368-c0392; the lines either side kept) | detector 83-165, 1338-1421 / 0 (the release subtitles neither theme) |
| Laughter spans kept (tag) | 17.5-22.5, 327.5-332.5, 1167.5-1172.5 | 65-72.5, 652.5-660 |
| Dub gain to the original dialogue level | -3.30 dB | -3.83 dB |
| Loudness original / premix / final (LUFS) | -21.25 / -21.96 / -21.27 | -21.28 / -22.71 / -21.25 |
| LRA original / final | 9.9 / 9.1 | 9.6 / 9.6 |
| loudnorm mode | dynamic (linear would pass -1 dBTP) | dynamic |

Outputs: `~/fukikae-scratch/t7/e11-ws/.fukikae/episodes/S01E11/out/S01E11.fukikae.mkv` (h264 copied; aac eng dub default; aac jpn; ass eng dub subtitles; original ass eng, forced kept) and `~/fukikae-scratch/t7/e12-ws/.fukikae/episodes/S01E12/out/S01E12.fukikae.mkv` (h264; eng dub default; jpn; the original eng dub track; dub ASS; five original ASS tracks; eight font attachments).

### Task 7 fix round 1 (rulings CD10, CD11)

- CD10, loudness: one linear gain to the original's integrated loudness, then `alimiter` at 4x the rate with a -1 dBTP ceiling (no loudnorm). Subtitles: the dub's ASS is default; every original subtitle track is kept with default and forced cleared; the dub audio is default.

| ebur128 (I LUFS / LRA LU / TP dBTP) | original | premix | final | gain |
|---|---|---|---|---|
| E11 | -21.2 / 9.9 / -1.1 | -21.9 / 10.0 / -1.1 | -21.2 / 10.0 / -1.0 | +0.7 dB |
| E12 | -21.2 / 9.4 / -1.1 | -22.7 / 10.2 / -1.4 | -21.2 / 10.2 / -1.0 | +1.5 dB |

The premix's own range (10.0, 10.2) passes to the final unchanged; it differs from the original's because the dub replaces the voices.
- CD11: `protect_bgm` defaults to false (SenseVoice BGM under dialogue: 10 E11 lines) and a dubbed cue clears the original vocals over its whole slot; both are recorded in the README as departures from the foundation spec. A slot overlapping an undubbed cue (errored, untranslated, skipped) clears only the dub's played span there.
- Song detector tightened: an anime negative crossed the energy bar (E13 battle at 611-691 s, score 0.86; E14 384-464 s, 0.54), so a candidate must also be sung: at least 70% of active 40 ms frames of `vocals_16k.wav` periodic (autocorrelation peak >= 0.7 at 80-1000 Hz). Committed settings, `scripts/song_scores.py` on ~/fukikae-ws: FT openings/endings voiced 0.78-0.94 (all accepted), Kaze openings 0.76-0.77 (accepted); negatives E13 middle 0.51 and E14 middle 0.33 (refused by voicing), FT cold opens score 0.15-0.28, BBT en/de and Grid best score 0.23. Settings are `[mix.songs]`; candidates go to `09-mix/report.json` and the run log.
- Re-mux on the copies: E11 song candidates head 65-147 (0.738, voiced 0.864), tail 1338-1421 (0.900, 0.911), 41 cues dropped (detector 41); E12 head 83-165 (0.725, 0.865), tail 1338-1421 (0.925, 0.883), 0 dropped.

## Task 8: `eval tts` Kokoro baseline (2026-11-09)

Commits 6009538 and eba6683. Kokoro stock cards, 8 cues per character, seed 0, renders at the card's own rate (no rate fit, no stretch), trimmed as the tts stage trims. Similarity is WeSpeaker cosine against the character's three longest clean original turns (stock cards have no references), so these are floors on the scale a clone will be judged on. WER is faster-whisper large-v3 (no VAD) on the render against the cue text, pooled over words.

| set | pair | cues | chars | sim | WER | fit | fit within 1.25 | stretch p50 / p90 | cps median | emotion agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| Fairy Tail E11 (t7 copy) | ja-en | 64 | 8 | 0.039 | 0.039 | 0.77 | 0.91 | 1.00 / 1.25 | 16.5 | 0.13 (8 labelled, SenseVoice; it labels 1 of 8 renders) |
| Fairy Tail E12 (t7 copy) | ja-en | 72 | 9 | 0.045 | 0.046 | 0.93 | 1.00 | 1.00 / 1.00 | 16.7 | 0.13 (67 labelled, emotion2vec; renders 66% neutral) |
| BBT S01E01 de (t8 copy) | de-en | 24 | 3 | 0.097 | 0.018 | 0.29 | 0.46 | 1.31 / 2.98 | 17.4 | 0.43 (23 labelled) |

BBT against the original English track (bbt-en, same timeline): render / original line duration median 1.13, emotion agreement with the original line 0.36, render-to-original-actor similarity 0.09 (24 of 24 cues matched English words).
Per-card speech rate (CD8), cps median: Kokoro voices 12.2 (af_nicole) to 20.4 (am_puck) across the three sets; the CD8 global 15.5 holds as a median, the spread is real.
BBT de: only 3 characters on the copy (the leads merge on the German dub, as the live-action profile notes); diarize and characters re-ran on the copy under the current profile before tag, translate and tts; translate left 146 of 334 cues over budget, which is why fit is 0.29.

### Task 8 fix round 1 (e29a7bb): rerun with coverage, WER floor and minimum slot

Ruling CD12 is implemented: `prepare_cue` and `render_cue` in `stages/tts.py` are the one render path; TtsStage and `eval tts` both call them, and a test pins identical audio under the same key.
Emotion agreement now counts only cues labelled on both sides and becomes a headline only at 50% or more coverage on both sides. Similarity is reported per reference source. Cues with a slot under 1.0 s are left out and counted. BBT matching is within 0.1 s.
WER floor: the same ASR (faster-whisper large-v3, no VAD) on natural English speech against the named-ASS ground truth `~/fukikae-ws/subs/bbt/labels` (OCR'd release subtitles): BBT on the English track at the sampled cues; Fairy Tail on a fixed 40-clip set from bbt-en S01E01. The floor includes subtitle condensation and OCR errors ("|" for "I").

| set | cues (short left out) | sim (original) | WER / floor | fit | fit within 1.25 | stretch p50 / p90 | cps | emotion: agreement (source / render coverage) |
|---|---|---|---|---|---|---|---|---|
| Fairy Tail E11 | 64 (114) | 0.038 | 0.046 / 0.151 | 0.88 | 0.98 | 1.00 / 1.06 | 17.3 | low coverage, SenseVoice (0.17 / 0.03), 0 both-labelled |
| Fairy Tail E12 | 72 (8) | 0.051 | 0.043 / 0.151 | 0.94 | 1.00 | 1.00 / 1.00 | 16.8 | 0.10 (0.94 / 0.86), 59 cues, emotion2vec |
| BBT S01E01 de | 24 (85) | 0.099 | 0.010 / 0.102 | 0.21 | 0.67 | 1.18 / 1.52 | 18.2 | 0.47 (0.96 / 0.83), 19 cues |

BBT against the English original: duration ratio 1.28, emotion agreement 0.56 (18 cues; coverage 0.92 / 0.83), similarity to the original actor 0.089, 0 unmatched cues.
Kokoro takes no delivery; its renders are neutral (E12 65%, BBT 60% of labelled renders), so agreement tracks the neutral share of the source labels.
The rows from 6009538 and eba6683 stay in the append-only ledger under their older options hashes; the e29a7bb rows supersede them.

## Task 9a: worker plumbing, reference selection (CD5) and IndexTTS2 (2026-11-09)

Commits d068ea7, 4e335a8, b2df55f, 53ff885, 0cf20e5, 924bfda, 32d39ea.

Calls for review:
- Worker environments are separate uv projects (`workers/<extra>/`, own Python and lock, venv in `.venvs/<extra>`), not extras of fukikae's lock: `indextts` declares `requires-python = ">=3.10,<3.12"` and pins torch 2.8, transformers 4.52.1 and numpy 2.2.6, which fukikae (>= 3.12) cannot resolve. The worker runs the venv's interpreter directly with `PYTHONPATH=<checkout>/src`; `worker.py` and `impl_*.py` import only the standard library and the model's packages.
- IndexTTS2 declares `capabilities = {emotion_audio}` only. Its emotion vector and emotion text replace the emotion audio prompt rather than add to it, and every cue has a source span; the released inference code has no duration control (`use_speed` fixed at 0, no target length). QwenEmotion is not loaded.
- Reference selection excludes theme songs and singing (`reference_exclusions`): on E11 the opening's separated lyrics were the cleanest turns and became two of character 1's three references (WER 0.654 for that character before the fix, 0.192 after).
- `[tts] device` is a stage-wide preference: the E11/E12 copies said `device = "cpu"` (for Kokoro), which put IndexTTS2 on the CPU; removed on the 9a copies. A per-backend device may be wanted.

Pins: IndexTeam/IndexTTS-2 740dcaff396282ffb241903d150ac011cd4b1ede; indextts git d9e41aac89fd00b3d71497fddb287b7f24613712; w2v-bert-2.0 da985ba0, MaskGCT semantic codec 265c6cef, CAMPPlus e4b6ede7, BigVGAN 22k 633ff708. Downloads 7.2 GB (HF cache on /nvme), worker venv 8.3 GB. Load about 70 s cold (WeTextProcessing builds its FST once, 43 s), 9.0 GB on GPU 0, about 3 to 4 s per line (E11: 40 renders plus scoring in 199 s; E12: 72 in about 410 s). Speech rate after trimming 14.3 cps (E11 13.8, E12 14.6; per card 11.1 to 18.3), now `IndexTts2Synthesizer.speech_cps`.

eval tts on copies at ~/fukikae-scratch/t9a (8 cues per character, seed 0, minimum slot 1 s). Similarity for both backends is against the same `select_references` spans (`references`); characters without 6 s of clean, non-song speech have no clone and are skipped for IndexTTS2 (E11: 4, 7, 8), so Kokoro's E11 references figure covers the same five characters.

| set | backend | cues | sim (references) | WER / floor | fit | fit within 1.25 | stretch p90 | cps | emotion agreement | BBT: dur ratio / emo vs original / sim to original |
|---|---|---|---|---|---|---|---|---|---|---|
| FT E11 | indextts2 | 40 (5 chars) | 0.521 | 0.080 / 0.151 | 0.58 | 0.90 | 1.26 | 13.8 | low coverage (SenseVoice) | |
| FT E11 | kokoro | 64 (8) | 0.001 (same 5) | 0.046 / 0.151 | 0.88 | 0.98 | 1.06 | 17.3 | low coverage | |
| FT E12 | indextts2 | 72 (9) | 0.483 | 0.080 or 0.556 / 0.151 | 0.89 | 0.94 | 1.07 | 14.6 | 0.30 (63 cues) | |
| FT E12 | kokoro | 72 (9) | 0.042 | 0.043 / 0.151 | 0.94 | 1.00 | 1.00 | 16.8 | 0.10 (59) | |
| BBT de E01 | indextts2 | 24 (3) | 0.609 | 0.024 / 0.102 | 0.13 | | 1.63 | 17.0 | 0.32 (22) | 1.37 / 0.43 / 0.125 |
| BBT de E01 | kokoro | 24 (3) | 0.094 | 0.010 / 0.102 | 0.21 | | 1.52 | 18.2 | 0.47 (19) | 1.28 / 0.56 / 0.089 |

E12 IndexTTS2 has two rows over the same renders (a stray second run): WER 0.080 and 0.556. The difference is faster-whisper (no VAD) looping on one render, character 5's c0232, where "That's not it." came out as a laugh ("Mwahahahaha"), the source line's laughter carried by the emotion reference. Every E12 character's references carry a scored emotion (CD5 notes on all nine): emotion2vec labels most E12 turns non-neutral at >= 0.5, so the demotion rarely leaves enough neutral speech. E11 character 1 still hallucinated one line (c0025, "You didn't know?" three times) before the song fix; that cue was a translated lyric.
Similarity against the clone's own reference spans is generous by construction for IndexTTS2; the listening checkpoint decides.
The first E11 IndexTTS2 row (sim 0.488, WER 0.171) predates 0cf20e5 and is superseded.
