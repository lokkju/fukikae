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

Run 1 showed 105 cues of at most two characters: Qwen3's aligner stamps each character with about 20 ms, and a diarization turn boundary falling inside a word cut off one character as its own cue ("ロ" | "ックと..."). The same 20 ms stamps made every slot end too early, so budgets were short. 818bfc1 folds such fragments into the neighbour of the same ASR segment and lets a point-stamped word sound until the next word, at most 0.3 s. Run 2's 23 overruns are a median of 3 characters over (13 of them 3 or fewer), the largest 12.

- 34 cues carried a delivery direction, which is all 34 cues with a non-neutral emotion tag.
- No cue was skipped as song: SenseVoice found no `singing` span on this episode's vocals (events: bgm, crying, laughter only), so the opening theme's lyrics were translated as dialogue.
- `nvidia-smi --id=0` showed no process after each run. An accidental `timeout 170` kill of the CLI mid-run also left no llama-server behind.
