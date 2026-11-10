# Character-cards plan: execution ledger

Plan: `docs/superpowers/plans/2026-11-08-character-cards.md`. Facts and numbers from each task's real runs.

## Task 4: translate, ASR path, managed llama-server

Run 2026-11-08 on a copy of the Fairy Tail workspace (E11, ja to en) at `<scratch>/ft-ws`, never the original.

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

Run 2026-11-08 on a copy of the Fairy Tail workspace (E12, ja audio, Crunchyroll's English ASS) at `<scratch>/t5-ws`, never the original. `[translate] source = "auto"` took the subtitle path: no `[subs]` and no sidecar, so it picked embedded stream 4 (eng, ASS, default), passing over stream 3 (eng, "Forced", forced disposition, three sign lines). The copy's extract, separate, asr and diarize keys were stale against today's code and re-ran on GPU 0 (212 s); the original workspace's registry is ECAPA (192-d), so the copy uses Task 4's re-embedded E11 registry, and characters re-ran against it. No character has a name, so speakers are `char<id>` and `extra:<cluster>`.

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

### Task 9a fix round 1 (rulings CD14 to CD18)

Commits 65edcc5 to cd1f5b6. eval tts re-run on the t7 copies (~/fukikae-scratch/t7, which still say `[tts] device = "cpu"`: IndexTTS2 ran on GPU 0 through its own default, CD14), with the deterministic ASR (CD16) and held-out similarity (CD17). Held-out exemplars exist for 3 characters per episode (2, 3, 5 on E11; 1, 2, 4 on E12): single-episode runs leave few eligible turns once the picks and the sampled cues' delivery spans are out. The ceiling is those clips against the reference clips.

| set | backend | cues | sim held-out (headline for clones) | ceiling | sim references | WER / median / floor | fit | stretch p90 | cps | retried / suspect | WER > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FT E11 | indextts2 | 40 (5 chars) | 0.424 (7 clips, 29.4 s) | 0.682 | 0.517 | 0.080 / 0.000 / 0.151 | 0.63 | 1.17 | 13.8 | 3 / 1 (c0058) | c0267 |
| FT E11 | kokoro | 64 (8) | -0.027 | 0.682 | 0.001 | 0.046 / 0.000 / 0.151 | 0.88 | 1.06 | 17.3 | 0 / 0 | c0058, c0267, c0337 |
| FT E12 | indextts2 | 72 (9) | 0.304 (7 clips, 28.8 s) | 0.536 | 0.489 | 0.074 / 0.000 / 0.151 | 0.89 | 1.07 | 14.6 | 1 / 1 (c0232) | c0160, c0235, c0236, c0284 |
| FT E12 | kokoro | 72 (9) | 0.054 | 0.536 | 0.042 | 0.043 / 0.000 / 0.151 | 0.94 | 1.00 | 16.8 | 0 / 0 | c0196, c0235, c0236, c0284 |

Ledger run_ids: 873e0913 (E11 indextts2), 636a717d (E11 kokoro), 0ae8795e (E12 indextts2), 0aabe773 (E12 kokoro); tags candidates-fix1 and baseline-fix1. A second IndexTTS2 E12 pass over the same renders gave identical metrics and the same WER > 0.5 list, and the two-pass gpu test on c0232's render gave the same hypothesis twice. The duplicate Task 9a E12 row 25a875dc (WER 0.556) is retracted (record 07209874).

c0232 ("That's not it.", the laugh render) is caught by the length guard: 3.44 s against 0.98 s expected, seed 1235 gave 2.51 s, still over the 2.45 s bound, so it is marked suspect. The laughter drop could not fire there: the tag stage has no laughter event over that span (E12 has three laughter or crying spans in all). c0235, c0236 and c0284 are over 0.5 WER for both backends, so the cause is the text, not the voice.

### Task 9b: CosyVoice 3 beside Kokoro and IndexTTS2

CosyVoice 3 (Fun-CosyVoice3-0.5B-2512 at 29e01c4e, CosyVoice code 074ca6dc, Matcha-TTS dd9105b3) in `[tts.cosyvoice3] mode = "auto"`, which was cross-lingual for every card here (Japanese and German clips, English text). eval tts on the t7 copies (FT) and ~/fukikae-scratch/t8/bbt-de (`--reference-workspace` bbt-en), 8 cues per character, seed 0, deterministic ASR (CD16), held-out similarity with the ceiling (CD17). The clone cards' reference spans are identical to IndexTTS2's on all three sets (same `select_references`, checked card by card). Kokoro and IndexTTS2 on FT were re-run over their cached renders without ledger rows and reproduced the fix-round rows exactly; BBT got fresh rows for all three (no held-out rows existed for it).

| set | backend | cues | sim held-out | ceiling | sim refs | WER / median / floor | fit | stretch p90 | cps | emotion agr. | retried / suspect | WER > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FT E11 | kokoro | 64 (8) | -0.027 | 0.682 | 0.001 | 0.046 / 0 / 0.151 | 0.88 | 1.06 | 17.3 | low cov. | 0 / 0 | c0058, c0267, c0337 |
| FT E11 | indextts2 | 40 (5) | 0.424 | 0.682 | 0.517 | 0.080 / 0 / 0.151 | 0.63 | 1.17 | 13.8 | low cov. | 3 / 1 | c0267 |
| FT E11 | cosyvoice3 | 40 (5) | 0.428 | 0.682 | 0.478 | 0.080 / 0 / 0.151 | 0.80 | 1.12 | 15.0 | low cov. | 0 / 0 | c0054, c0058, c0267 |
| FT E12 | kokoro | 72 (9) | 0.054 | 0.536 | 0.042 | 0.043 / 0 / 0.151 | 0.94 | 1.00 | 16.8 | 0.10 | 0 / 0 | c0196, c0235, c0236, c0284 |
| FT E12 | indextts2 | 72 (9) | 0.304 | 0.536 | 0.489 | 0.074 / 0 / 0.151 | 0.89 | 1.07 | 14.6 | 0.31 | 1 / 1 | c0160, c0235, c0236, c0284 |
| FT E12 | cosyvoice3 | 72 (9) | 0.249 | 0.536 | 0.455 | 0.061 / 0 / 0.151 | 0.90 | 1.00 | 16.4 | 0.17 | 0 / 0 | c0160, c0184, c0196, c0235, c0284 |
| BBT de | kokoro | 24 (3) | 0.086 | 0.605 | 0.094 | 0.010 / 0 / 0.102 | 0.21 | 1.52 | 18.2 | 0.47 | 0 / 0 | - |
| BBT de | indextts2 | 24 (3) | 0.540 | 0.605 | 0.608 | 0.024 / 0 / 0.102 | 0.08 | 1.73 | 17.0 | 0.35 | 0 / 0 | - |
| BBT de | cosyvoice3 | 24 (3) | 0.444 | 0.605 | 0.544 | 0.021 / 0 / 0.102 | 0.08 | 2.48 | 12.8 | 0.39 | 0 / 0 | - |

Held-out covers 3 of 5 (E11), 3 of 9 (E12) and 3 of 3 (BBT) characters; the table and rows now say so (`similarity_headline_note`). Against the English original (BBT): duration ratio 1.28 / 1.38 / 1.61, emotion agreement 0.56 / 0.41 / 0.65, similarity to the original actor 0.089 / 0.117 / 0.154 (Kokoro / IndexTTS2 / CosyVoice 3).
Ledger run_ids: 4333da21 (E12 cosyvoice3), a219b732 (E11 cosyvoice3), 0879aa04, 217d180d, 26d1032f (BBT kokoro, indextts2, cosyvoice3); tag candidates-9b.
Resources: worker venv 5.4 GB, model 5.1 GB (HF cache on /nvme), about 3.8 GB of VRAM over idle on GPU 0, load 28 s, about 0.6 to 0.8 s of compute per second of speech; a second render with the same seed is byte-identical.

### Task 9c: Qwen3-TTS beside Kokoro, IndexTTS2 and CosyVoice 3

Qwen3-TTS 1.7B Base (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` at fd4b2543, `qwen-tts` 0.1.1 by version and lock hash), bf16 on GPU 0, `[tts.qwen3_tts] mode = "auto"`, which was `icl` (the clips' transcripts and speech codes) for every card here. eval tts on the same copies and settings as Task 9b (8 cues per character, seed 0, deterministic ASR, held-out similarity with the ceiling); the clone cards' reference spans are identical to IndexTTS2's and CosyVoice 3's on all three sets, checked card by card. The rows were rendered at e68e1f3, when the render guard still used a provisional 15.0 cps; the declared 13.2 came after (d847f9b). A second render with the same seed is byte-identical (gpu test).

| set | backend | cues | sim held-out | ceiling | sim refs | WER / median / floor | fit | stretch p90 | cps | emotion agr. | retried / suspect | WER > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FT E11 | kokoro | 64 (8) | -0.027 | 0.682 | 0.001 | 0.046 / 0 / 0.151 | 0.88 | 1.06 | 17.3 | low cov. | 0 / 0 | c0058, c0267, c0337 |
| FT E11 | indextts2 | 40 (5) | 0.424 | 0.682 | 0.517 | 0.080 / 0 / 0.151 | 0.63 | 1.17 | 13.8 | low cov. | 3 / 1 | c0267 |
| FT E11 | cosyvoice3 | 40 (5) | 0.428 | 0.682 | 0.478 | 0.080 / 0 / 0.151 | 0.80 | 1.12 | 15.0 | low cov. | 0 / 0 | c0054, c0058, c0267 |
| FT E11 | qwen3_tts (icl) | 40 (5) | 0.462 | 0.682 | 0.500 | 0.054 / 0 / 0.151 | 0.53 | 1.54 | 12.3 | low cov. | 2 / 2 (c0267, c0304) | c0267 |
| FT E11 | qwen3_tts (xvector) | 40 (5) | 0.370 | 0.682 | 0.415 | 0.054 / 0 / 0.151 | 0.75 | 1.40 | 14.8 | low cov. | 2 / 0 | c0058, c0126, c0267 |
| FT E12 | kokoro | 72 (9) | 0.054 | 0.536 | 0.042 | 0.043 / 0 / 0.151 | 0.94 | 1.00 | 16.8 | 0.10 | 0 / 0 | c0196, c0235, c0236, c0284 |
| FT E12 | indextts2 | 72 (9) | 0.304 | 0.536 | 0.489 | 0.074 / 0 / 0.151 | 0.89 | 1.07 | 14.6 | 0.31 | 1 / 1 | c0160, c0235, c0236, c0284 |
| FT E12 | cosyvoice3 | 72 (9) | 0.249 | 0.536 | 0.455 | 0.061 / 0 / 0.151 | 0.90 | 1.00 | 16.4 | 0.17 | 0 / 0 | c0160, c0184, c0196, c0235, c0284 |
| FT E12 | qwen3_tts (icl) | 72 (9) | 0.335 | 0.536 | 0.457 | 0.037 / 0 / 0.151 | 0.89 | 1.01 | 13.7 | 0.21 | 0 / 0 | c0184, c0196, c0235, c0284 |
| FT E12 | qwen3_tts (xvector) | 72 (9) | 0.309 | 0.536 | 0.431 | 0.035 / 0 / 0.151 | 0.94 | 1.00 | 15.3 | 0.22 | 0 / 0 | c0235, c0284 |
| BBT de | kokoro | 24 (3) | 0.086 | 0.605 | 0.094 | 0.010 / 0 / 0.102 | 0.21 | 1.52 | 18.2 | 0.47 | 0 / 0 | - |
| BBT de | indextts2 | 24 (3) | 0.540 | 0.605 | 0.608 | 0.024 / 0 / 0.102 | 0.08 | 1.73 | 17.0 | 0.35 | 0 / 0 | - |
| BBT de | cosyvoice3 | 24 (3) | 0.444 | 0.605 | 0.544 | 0.021 / 0 / 0.102 | 0.08 | 2.48 | 12.8 | 0.39 | 0 / 0 | - |
| BBT de | qwen3_tts (icl) | 24 (3) | 0.242 | 0.605 | 0.273 | 0.010 / 0 / 0.102 | 0.17 | 1.74 | 18.1 | 0.40 | 0 / 0 | - |
| BBT de | qwen3_tts (xvector) | 24 (3) | 0.372 | 0.605 | 0.440 | 0.024 / 0 / 0.102 | 0.17 | 1.80 | 16.6 | 0.24 | 0 / 0 | - |

Held-out covers 3 of 5 (E11), 3 of 9 (E12) and 3 of 3 (BBT) characters, the same characters for every backend. Noise caveat for every comparison in this table (and the earlier Task 9a and 9b tables it extends): each held-out similarity is the mean over n = 3 characters per set, eight cues each, so a gap such as CosyVoice 3's 0.428 against IndexTTS2's 0.424 and Qwen3-TTS's 0.462 on E11 is within noise. Only the large gaps (BBT icl at 0.242 against IndexTTS2's 0.540, or Kokoro's near-zero rows against every clone) are outside it; the ranking among the three clones on Fairy Tail is not established. On BBT, icl collapses for two of the three characters (held-out 0.151 and 0.094, against 0.393 and 0.306 in xvector); on Fairy Tail it is the better mode on both episodes. Against the English original (BBT, icl): duration ratio 1.375, emotion agreement 0.58, similarity to the original actor 0.051.
Ledger run_ids: a3ce5e60 (E11), d89f2158 (E12), 8e1f33eb (BBT), tag candidates-9c; the xvector A/B rows 1c8e91ea, 918cf58f, 21785803, tag candidates-9c-xvector.

Instruction A/B (ruling CD20). The Base model's API takes no instruction; its `generate` accepts `instruct_ids` (the CustomVoice and VoiceDesign models' input). Four English lines, one E11 reference clip (character 3), seed 1234, rendered with no direction and with "angry, shouting", "whispering", "sad, crying" and "happy, excited" (20 renders, `~/fukikae-scratch/t9c/probe`), then scored with emotion2vec, loudness and pitch. Directions changed the render (other samples) but not as intended: angry scored 0.00 to 0.04 on all four lines, "sad, crying" never came out sad (one came out happy at 1.00), "happy, excited" came out happy once, on the same line that was happy with no direction, and "whispering" was 2 to 4 dB quieter on two lines and as loud as the undirected render on the other two. No delivery capability is declared.

Resources: worker venv 6.9 GB on /, model 4.3 GB (HF cache on /nvme), 4.7 GB of VRAM over idle on GPU 0 (nvidia-smi peak 5130 MiB against 470 MiB), load plus two renders 22 s.

### Task 9b fix round 1 (ruling CD20)

CosyVoice 3's `instruction` capability is withdrawn (its prompt wording was unverified, and instruct mode replaces the LLM's speaker prompt); its own control tags are stripped by the worker. Every CosyVoice 3 row above was re-run without instructions (only the directed cues re-rendered), tag candidates-9b-fix1; Kokoro on FT E11 was re-run on the clones' five characters (1, 2, 3, 5, 6; `--characters`, same cues per character), tag baseline-9b-fix1.

| set | backend | cues | sim held-out | ceiling | WER / median / floor | fit | stretch p90 | cps | emotion agr. | suspect | WER > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FT E11 | kokoro (clone chars) | 40 (5) | -0.027 | 0.682 | 0.054 / 0 / 0.151 | 0.85 | 1.08 | 16.3 | low cov. | 0 | c0058, c0267, c0337 |
| FT E11 | indextts2 | 40 (5) | 0.424 | 0.682 | 0.080 / 0 / 0.151 | 0.63 | 1.17 | 13.8 | low cov. | 1 | c0267 |
| FT E11 | cosyvoice3 | 40 (5) | 0.433 | 0.682 | 0.080 / 0 / 0.151 | 0.88 | 1.04 | 16.1 | low cov. | 0 | c0054, c0058, c0267 |
| FT E12 | cosyvoice3 | 72 (9) | 0.251 | 0.536 | 0.061 / 0 / 0.151 | 0.90 | 1.00 | 16.4 | 0.17 | 0 | c0160, c0184, c0196, c0235, c0284 |
| BBT de | cosyvoice3 | 24 (3) | 0.420 | 0.605 | 0.017 / 0 / 0.102 | 0.13 | 2.15 | 15.7 | 0.32 | 0 | - |
| BBT de | cosyvoice3 zero_shot | 24 (3) | 0.525 | 0.605 | 0.044 / 0 / 0.102 | 0.04 | 2.09 | 13.8 | 0.37 | 0 | c0150, c0252 |

Held-out similarity rests on 3 characters per FT episode (7 clips, about 29 s) and 3 on BBT; E12's 0.251 (CosyVoice 3) against 0.304 (IndexTTS2), or E11's 0.433 against 0.424, is within what three characters can resolve.

Instruction A/B (the same cues with and without the direction; the earlier candidates-9b rows had it): E11 held-out 0.428 with, 0.433 without; E12 0.249 with, 0.251 without, emotion agreement 0.173 both; BBT 0.444 with, 0.420 without, emotion agreement 0.389 with, 0.316 without, but against the English original's emotions 0.647 with, 0.667 without, and the directed renders were much slower (BBT cps 12.8 with, 15.7 without; duration ratio to the English 1.61 with, 1.40 without). No clear help, so it stays withdrawn.

Mode A/B on BBT de (German transcripts): zero_shot raises held-out similarity (0.525 against 0.420, near IndexTTS2's 0.540) but slows the speech (13.8 against 15.7 cps, duration ratio to the English 1.75 against 1.40) and costs intelligibility (WER 0.044 against 0.017, two renders over 0.5). The prompt tokens set the pace: the LLM continues the German speaker's tempo. In the default cross-lingual mode a German-referenced card runs at 15.7 cps, close to the declared rate, so it needs no per-card rate; zero_shot cards would (CD8).

The declared speech rate is now 16.3 cps (FT E11 16.1, E12 16.4, without instructions), measured on Fairy Tail only; the earlier 15.9 included the slower directed renders.
Ledger run_ids: cc9734e1 (E11 kokoro, 5 characters), b500f419 (E11 cosyvoice3), eded422d (E12 cosyvoice3), 0242d0d4 (BBT cosyvoice3), 8c9be93d (BBT cosyvoice3 zero_shot, tag ab-9b-zero-shot).

## Task 10: packs and verification (2026-11-09)

Pack format: a directory or a `.tar.zst` (ustar in zstd, `zstandard` 0.25). The archive is read as a stream through a capped decompressor and never extracted by `tarfile`; every member is checked against the layout (pack.toml, cast.toml, `cards/<id>.toml`, `clips/<sha256>.wav`, `latents/` and `vectors/<sha256>.npy`), copied into a private staging directory, and only then are cards parsed (strict loader, id recomputed), sidecars hashed and loaded (wav header, `.npy` mapped without pickle). Default limits: 64 MiB per file, 1 MiB per TOML file, 2 GiB in total, 10,000 files; clips at most 30 s, as on export. Exports are byte-reproducible (sorted members, zeroed owners and times).

Real round trip on scratch copies (`~/fukikae-scratch/t10`, `cp -al` with private registries) of the Fairy Tail E11 and E12 workspaces. Character names and external ids are placeholders set for the test, not identifications. E11 copy: seven named characters, six Kokoro stock cards and the IndexTTS2 clone card of character 6 (0134522483aa1a54, two clips, 7.6 s), assigned by hand; one unnamed character. `voices export` wrote `~/fukikae-ws/packs/fairytail-en.tar.zst` (310 KiB, 7 cast entries, unnamed character left out); `--no-clips` also left out the IndexTTS2 card. Import into the E12 copy (names changed: full-width, lower case, longer names with matching external ids; one character locked): 3 assigned (two by external id, one by name), 1 unchanged (same card already there), 1 skipped (locked), 2 unmatched (a name and an external id the E12 copy doesn't have), series matched by title only, with the warning.

Fingerprints (WeSpeaker ResNet34 on the CPU, threshold 0.85): Kokoro af_nicole card f1f8511e98b95055 recorded, then verified at cosine 0.998 (CPU, 17 s per run); IndexTTS2 card 0134522483aa1a54 recorded, then verified at 1.000 on GPU 0 under the GPU lock (61 s per run, mostly the worker's load), and again on import into the E12 copy (1.000). The Kokoro card was not re-verified on import: the E12 store already held the same card id without a fingerprint, and import never overwrites a store file.

### Task 10 fix round 1 (rulings CD22, CD23)

Fingerprint embedders are an allowlist (`FINGERPRINT_EMBEDDERS`: WeSpeaker ResNet34, `pyannote/wespeaker-voxceleb-resnet34-LM` at 837717dd, loaded by path from the local cache with the hub offline); any other embedder string verifies as `unavailable` with nothing looked up or loaded. Rerun of the round trip on fresh copies (`~/fukikae-scratch/t10b`, pack `~/fukikae-ws/packs/fairytail-en-fix1.tar.zst`): the same mapping as before (3 assigned, 1 unchanged, 1 skipped, 2 unmatched); the E12 store's copy of the Kokoro card f1f8511e98b95055 had no fingerprint, so the pack's was merged in (CD22) and verified at cosine 0.998; the IndexTTS2 card verified at 1.000 on GPU 0. The import took 66 s, both verifications included.

## Task 11: listening checkpoint, the blind A/B pack (2026-11-09)

`fukikae eval tts-ab <spec.toml> -o <folder>` builds the pack (`src/fukikae/eval/ab.py`). The spec used is `~/fukikae-review/ab-2026-11.spec.toml` (seed 20261109); the pack is `~/fukikae-review/ab-2026-11/` (owner-only, 194 MB). No default changed: stock stays the interim default until the operator's verdict (D15).

- **Sets.** Fairy Tail E12, ja to en, subtitle path (`~/fukikae-scratch/t7/e12-ws`), candidates kokoro, indextts2, cosyvoice3 (cross_lingual) and qwen3_tts (icl), with the official English dub track (`~/fukikae-ws/fairytail-en`) as a second reference. BBT S01E01, de to en (`~/fukikae-scratch/t8/bbt-de`), the same four plus cosyvoice3:zero_shot and qwen3_tts:xvector (CD21's per-card modes, the two the Task 9 A/Bs left open), with the English original (`~/fukikae-ws/bbt-en`).
- **Cues.** 12 (6 per set) over 4 FT characters (1, 2, 3, 5) and 3 BBT characters, drawn from `eval tts`'s own sample (8 per character, seed 0), so the renders are the ledger rows' renders, through the stage's render path with its rate fit (`render_for_slot`, now shared by the stage, CD12) and timefit's `plan_fit`/`apply_fit`. Suspect lines kept, named only in `key.toml`: FT c0232 (IndexTTS2's render-guard flag, the laugh) and c0284 (over 0.5 WER for all four in their eval rows); BBT c0252 (zero_shot over 0.5 WER). The other BBT rows (9b/9c) list no high-WER cues and no render there failed the guard.
- **Renders made for the pack:** CosyVoice 3's rate-fit renders (21 BBT, 7 FT, the eval scored the card's own rate), zero_shot's 47 on the bbt-de copy (its eval ran on the bbt-de-zs copy), and the scene's 47 clone renders; all on GPU 0 under `gpu_lock`. Everything else came from the copies' tts caches.
- **Blinding.** Per cue, `random.Random(f"ab-blind:{seed}:{slot}")` over the sorted candidate names; files are only `cueNN_A.wav`..., `cueNN_ref.wav`, `cueNN_ref_en.wav`; every clip of a cue covers one window (slot and every placed render plus 0.3 s), mono 48 kHz, at -23 LUFS by linear gain plus the -1 dBTP limiter (CD10). Checked: equal frames per cue, -23.0 LUFS (one clip -23.1), no backend name in `index.html` or `README.txt`, the key unlinked.
- **Scene.** FT E12 922.6 to 997.6 s (75 s, picked as the window with the most cloned characters speaking, clear of songs), 23 cues through tts, timefit and mix per candidate, no character needing a stock fallback; `audio/scene01_*.wav` and `video/scene01_*.mkv` (720p h264, FLAC).
- **Verdict.** `verdict.toml` takes per-cue and per-scene rankings by letter (filled before opening the key), then `[overall] pick` (a candidate name, or `stock`), `passed`, and `[[override]]` per character (set, registry id, backend or `stock`, optional mode). Task 12 reads it with `fukikae.eval.ab.read_verdict(pack)`, which resolves every letter through `key.toml` and rejects names that aren't candidates; picking kokoro counts as `stock`.

## Task 12: the verdict applied, full dubs and the held-out row (2026-11-09)

**Ruling D24 (the operator's verdict, D15 checkpoint).**
- Default clone backend: IndexTTS2. `[tts] clone_backend = "indextts2"` is in the default, anime and live-action profiles; `[tts] backend` stays Kokoro for stock and extras.
- Passed: IndexTTS2, Qwen3-TTS and CosyVoice 3.
- No per-character overrides. D12 resumes; characters under 6 s of clean speech fall back to stock, reported.
- Source: the blind rankings in `~/fukikae-review/ab-2026-11/verdict.toml`, read with `read_verdict`, with `[overall]` left empty by the operator. Decoded first places and points (3/2/1): indextts2 4 (scene included) and 20, qwen3_tts 3 and 17, cosyvoice3:zero_shot 2 and 10, cosyvoice3 2 and 9, qwen3_tts:xvector 2 and 9; kokoro never ranked.

**Ruling CD25.** The subtitle path reads a dialogue track only (`media.dialogue_track`):
- never a track whose title says signs, songs or forced;
- never one with under 3 cues a minute, or under 40% of the densest track of the language (from `NUMBER_OF_FRAMES` and the duration);
- a forced flag yields to a dialogue-sized count, and a flagged track of unknown size is skipped;
- plain tracks win over CC/SDH.

FT E11's only English track (flagged forced, "English", 344 cues, 14.4 a minute) now counts as dialogue. E12 to E14 skip the 3-cue "Forced" track.

**The clone path.** `cards.auto.workspace_cloning` builds the clone source. It applies CD5 selection over the character's turns in every processed episode of the workspace and takes the episode with the most picked speech. `ensure_assignments(..., cloning=...)` makes the clone card or the reported stock fallback, `by = "auto"` and unlocked, under the CD2, CD3 and CD9 refusals. Two additions to selection:
- a sparse-text filter, which drops a turn whose text rate is under 0.3 of the speaker's median and under 1 character a second (0.4 words a second in spaced languages);
- a stock gender estimate that skips theme songs and singing.

Fingerprints are not recorded at assignment (`voices verify --record` does it). `voices clone --clip --transcript [--backend]` assigns a hand-cut clone card, locked.

**The female-sounding FT characters (operator note).** These are merged clusters, not wrong turns:
- In the A/B copy, character 3 (SPEAKER_13) is the opening's singer plus single lines from seven speakers. Its picks were Young Makarov's "くそ" followed by 3.4 s with no speech, and a Natsu line.
- Character 2 (SPEAKER_12) is Charle, Wendy and Happy plus two Natsu spans. Its picks were Charle's line and a turn holding both Happy and Natsu.
- In the full run, by subtitle names across E11 to E14, character 2 is Wendy 67, Happy 65, Charle 25 and Natsu 15. Its card cloned Wendy and Charle in E14; character 3's cloned Touka in E13.
- The sparse filter drops the "くそ" turn, which sends A/B character 3 to stock (`af_heart`, unknown band). Character 2's picks are unchanged.
- An embedding-centroid check would not separate these voices: Charle's spans sit at 0.72 to 0.75 to the centroid.
- The fix belongs in identity: subtitle speaker names, or a registry split followed by `voices clone`.
- Clips are in `~/fukikae-review/t12-refs/`.

**IndexTTS2's comma pause.** The text was the subtitle line verbatim, and IndexTTS2's normaliser keeps the comma. The model renders the vocative comma as a 170 ms pause. Documented, not changed.

**Full dubs (`~/fukikae-scratch/t12/out/`).**
- FT E11 to E14 on the subtitle path, 251 to 354 cues each. 15 characters were cloned and none fell back to stock. 2 to 7 lines an episode stay over their slot after fitting.
- BBT de E01 to E03 on the ASR path, 4 characters cloned. 48 to 55 lines an episode stay over after fitting, because of the German budget. E03 has 168 extras cues.
- The FT copy's registry was reembedded from ECAPA first.

**Held-out row.** FT E14, IndexTTS2, run `f912c020`, run once with committed defaults:
- held-out similarity 0.449 over 2 characters (ceiling 0.698), and 0.518 against the references over 6 characters;
- WER 0.104 pooled (median 0);
- fit 0.75;
- emotion agreement 0.357;
- 0 suspect renders.

Results doc: `docs/results/2026-11-09-character-cards.md`.

## Performance task (D26, 2026-11-10)

Measured on copies of the Fairy Tail workspace (`~/fukikae-scratch/perf/ft`), GPU 0 (RTX 3090; it runs at a 200 W limit, and the clock drops to 800-900 MHz in IndexTTS2's diffusion decoder).

**Where IndexTTS2's time goes.** 12 E14 lines in fp32: the GPT's sampling 1.85 s a line (bound by kernel launches: 35% SM busy, and beams 3 or 1 cost the same), the diffusion decoder 1.55 s (25 steps over the 15 s prompt plus the line, compute-bound under the power limit), BigVGAN 0.15 s. Every option tried, 12 lines each, seconds a line: fp32 3.9; fp16 4.0; beams 1 3.9; tf32 3.4; torch.compile 8.4 (it recompiles per line length); fp16+tf32 with 4 threads 4.2; `accel` (IndexTTS2's flash-attention and CUDA-graph GPT engine, flash-attn 2.8.3 from its prebuilt wheel) 3.3, and with fp16 and tf32 2.2 at steady state; two workers sharing the GPU 10% more lines an hour. IndexTTS2 takes one line per call (its decoder caches are sized for batch 1), so there is no batching. The stage already keeps one worker per backend for the whole episode. DeepSpeed is not in the locked environment.

**Quality gate (eval tts, deterministic ASR, held-out similarity; same cues, tags and cards).**

| run | episode | held-out sim | WER | median WER | suspect | fit | cps |
|---|---|---|---|---|---|---|---|
| fp32 (before), `91a2d375` | E13 | 0.389 | 0.104 | 0 | 0 | 0.81 | 13.2 |
| fp32 (before), `63604187` | E14 | 0.449 | 0.104 | 0 | 0 | 0.75 | 13.2 |
| tf32, `b39eb263` | E13 | 0.400 | 0.100 | 0 | 0 | 0.81 | 12.7 |
| tf32, `f40b6b7e` | E14 | 0.451 | 0.081 | 0 | 0 | 0.75 | 12.8 |
| fp16+accel+tf32, `c37c5a10` | E13 | 0.366 | 0.095 | 0 | 0 | 0.63 | 11.8 |
| fp16+accel+tf32, `a461efed` | E14 | 0.428 | 0.130 | 0.091 | 0 | 0.65 | 13.1 |
| accel, `28c1a37e` | E13 | 0.383 | 0.095 | 0 | 0 | 0.63 | 11.8 |
| fp16+tf32, `bd02a3ab` | E13 | 0.410 | 0.119 | 0 | 0 | 0.78 | 12.9 |

tf32 passes and is on in the profiles (`[tts.indextts2] tf32 = true`, in every render's key). `accel` fails: held-out similarity -0.024 (E13) and -0.021 (E14), median WER +0.09 on E14, and slower speech (fit 0.81 to 0.63). Its engine samples at the temperature only, without top-p, top-k, beams or the repetition penalty. It was reverted, and the worker lock is unchanged. fp16 passes on E13 but is slower here, so it stays off.

**Rate estimate.** A backend that takes a rate renders each cue once at the rate its estimate needs: the text over the card's measured characters a second (the median of its earlier renders once three are in, else the backend's `speech_cps`). One more render follows only on a miss of over 10%. IndexTTS2 takes no rate, so none of its renders were ever rate fits: its extra renders are the CD18 guard retries (5 to 6 an episode). This applies only to Kokoro's stock and extras lines. E13 had 8 estimated cues, 6 of them re-rendered; E14 had 2, with none re-rendered. No saving on this series.

**Tag on the GPU.** SenseVoice and emotion2vec default to cuda under the GPU lock, with the CPU as the fallback. On the same turns, GPU against CPU: identical events, and identical emotion labels on 467/467 (E13) and 441/441 (E14) turns. The largest confidence difference is 0.006, with no turn crossing 0.5. SenseVoice takes 45 s against 200 s, emotion2vec 10 s against 108 s, using 1.0 and 1.4 GB of VRAM (peak reserved).

**Several episodes.** `fukikae run <ws> S01E13 S01E14` runs E13's timefit and mix (01:16:56-01:18:34) beside E14's extract and separate.

**Per-stage minutes, before (Task 12, sequential) and after (`--from extract`, both episodes in one run, empty tts cache).**

| stage | E13 before | E13 after | E14 before | E14 after |
|---|---|---|---|---|
| separate | 2.44 | 2.15 | 2.38 | 2.20 |
| asr | 1.42 | 2.07 | 1.33 | 0.83 |
| diarize | 0.56 | 0.56 | 0.58 | 0.56 |
| tag | 5.35 | 0.88 | 5.12 | 0.88 |
| translate | 0.14 | 0.13 | 0.25 | 0.25 |
| tts | 13.77 | 13.53 | 22.13 | 19.30 |
| timefit | 0.04 | 0.06 | 0.07 | 0.06 |
| mix | 1.39 | 1.57 | 1.48 | 1.43 |
| sum | 25.2 | 21.0 | 33.4 | 25.6 |

The two episodes took 45.0 min of wall time, against 58.5 min of stage time before. E13's rerun spoke 199 lines with IndexTTS2, against 168 before, because the characters stage now matches against a registry that holds E11 to E14. That is why its tts time barely moved. E14 rendered the same 329 renders 13% faster. The target of 5 to 10 min an episode is not reached: IndexTTS2 alone is 19 min for E14's 320 lines. What remains is model settings outside the options (diffusion steps, the 15 s prompt), `accel` with its sampling restored (a change in the pinned indextts code), or the GPU's power limit.
