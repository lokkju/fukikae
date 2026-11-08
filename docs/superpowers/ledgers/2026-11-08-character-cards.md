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
