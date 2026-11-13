# ZONOS3-Mini as a clone engine: spike against IndexTTS2 (#43)

_Run 2026-11-13 on GPU 0 (RTX 3090 at its 200 W limit, under `fukikae.gpulock`; vLLM held GPU 1 throughout), branch `spike/zonos3-mini` at fe4a7bb. Sets: Sol Bianca S00E01 (a copy of the Part A "after" workspace `~/fukikae-ws/t-ident/sb-final`, its cards and IndexTTS2 dub as they were) and Fairy Tail E12, ja to en. No Fairy Tail workspace with a translation or cards survived the scratch cleanup, so E12 is a fresh copy of `~/fukikae-ws/fairytail` with the anime profile, a new registry (the old one's exemplars predate the current diarize and couldn't be re-embedded) and `fukikae run --to timefit`; its characters carry the subtitle names (Natsu, Lucy, Young Makarov and so on). Workspaces, renders and logs are under `/nvme/fukikae/tmp/zonos3/`; the ledger rows are `2026-11-13-zonos3-mini-spike.jsonl` (six `eval tts` rows) and the driver scripts are in `scripts/zonos3-spike/`. Listening page: `/nvme/fukikae/tmp/zonos3/listen/index.html`._

**Call: drop it.** ZONOS3-Mini's clones are further from the characters than IndexTTS2's on every measure and both sets: similarity to the references is 0.092 lower on Sol Bianca and 0.110 lower on Fairy Tail (lower for 23 of 24 characters), held-out similarity 0.101 and 0.056 lower. That is two to four times the gap that kept the TensorRT engine opt-in (-0.025 against the references, -0.03 to -0.06 held-out), and it is worst exactly where the issue hoped ZONOS would help: thin references (-0.22 and -0.21) and the shouted-only Young Makarov (-0.21). It is twice as fast and fits slots better, which doesn't matter for a voice that isn't the character's.

## What ran

The worker (`workers/zonos3`, `.venvs/zonos3`, `fukikae backends setup zonos3`) runs Zyphra's plain-PyTorch reference (`zonos3.reference`) from their repository at commit `b171b19`, with the model snapshot `Zyphra/ZONOS3-Mini@f68e6e47` (LM, dual codec generator and SSL teacher, the default KEEP-2 iSTFT decoder, the MECT-B2 speaker encoder; about 11 GB) and the steering bundle of that commit, checked by digest. Torch 2.9.1 (cu128), transformers 4.57.3 and the rest are in `workers/zonos3/uv.lock`. The worker maps every `hf://Zyphra/ZONOS3-Mini/...` location the package resolves to the pinned local snapshot and refuses anything else; it runs offline with telemetry off like every worker. It declares no capability (CD20). It clones from the card's reference clips joined in card order (up to 30 s; IndexTTS2 reads 15 s), as a raw reference prefix (`mode = "reference"`, the checkpoint's default) or as a MECT-B2 speaker embedding (`mode = "embedding"`). No stock voice, default voice or designed voice was used anywhere (D41).

Neither runtime the card suggests could stand up here. SGLang-Omni and Zyphra's Mini-SGLang CUDA server JIT-compile TileLang, quack and FlashInfer kernels against a CUDA 12.8 toolkit, which this host doesn't have (no `/usr/local/cuda*`, no `nvcc`), and their exact-mode experts are Hopper-only (`env.sh` falls back to Triton kernels before sm_90). The reference is the numerical oracle both servers are tested against, so the voices are the model's; only the speed would differ.

Likeness, WER, fit and emotion agreement come from `fukikae eval tts --backend <b>` (8 cues a character, seed 0, cards cloned from each character's references, WeSpeaker similarity, deterministic ASR, emotion2vec), the same cues and references for every backend. Speed, pace, emotion and thin references come from `scripts/zonos3-spike/spike.py` through the same Synthesizer classes and the tts stage's trimming.

## 1. Likeness

| set, backend | cues | held-out sim (ceiling) | sim to references | WER | fit | stretch p90 | cps | emotion agreement |
|---|---|---|---|---|---|---|---|---|
| Sol Bianca, indextts2 | 80 | 0.454 (0.611) | 0.498 | 0.124 | 0.58 | 1.54 | 12.9 | 0.397 |
| Sol Bianca, zonos3 reference | 80 | 0.353 (0.611) | 0.406 | 0.087 | 0.80 | 1.19 | 15.7 | 0.187 |
| Sol Bianca, zonos3 embedding | 80 | 0.462 (0.611) | 0.520 | 0.433 | 0.35 | 2.79 | 9.7 | 0.189 |
| Fairy Tail E12, indextts2 | 105 | 0.368 (0.622) | 0.529 | 0.073 | 0.86 | 1.07 | 14.7 | 0.416 |
| Fairy Tail E12, zonos3 reference | 105 | 0.312 (0.622) | 0.419 | 0.081 | 0.94 | 1.00 | 17.2 | 0.402 |
| Fairy Tail E12, zonos3 embedding | 105 | 0.470 (0.622) | 0.581 | 0.717 | 0.46 | 2.48 | 9.6 | 0.330 |

Held-out similarity covers 9 of 10 Sol Bianca characters and only 4 of 14 on E12 (the rest have no speech left after the references), so similarity to the references is the steadier figure. In reference mode it is lower for 9 of 10 Sol Bianca characters (mean -0.092, range -0.263 to +0.054) and all 14 Fairy Tail characters (mean -0.110, range -0.277 to -0.018); Young Makarov drops from 0.617 to 0.340. ZONOS3 speaks faster (15.7 and 17.2 characters a second against 12.9 and 14.7), which is why its fit is better (section 4); its WER is level or better.

Embedding mode scores the highest similarity and is unusable: given a Japanese speaker's embedding and an English line it speaks Japanese-sounding syllables. faster-whisper hears E12 c0332 ("Erza... How long are you going to tantalize me?") as Japanese (p 0.86, "あらさから頃はちぎどとさ..."), WER is 0.43 and 0.72 with median WER 0.42 and 1.0, and renders run at 9.7 characters a second. The similarity is high because the speech is in the speaker's own language. The reference-mode renders of the same cues are English. Embedding mode is out, which matters for section 6.

## 2. Emotion transfer

Twenty lines per set the tag stage marked emotional (angry, sad, fearful, surprised, happy, disgusted, spread over labels and the sampled characters), four ways: IndexTTS2 with the line as emotion audio (the default path); ZONOS3 plain; ZONOS3 steered from the tag (EARS mood directions at dose 3 x 0.6 to 0.7, plus `delivery.loud` for angry and `mood.valence` for happy, the bundle's own steer-then-hold release); ZONOS3 with the line itself added to the front of the references ("a reference of its own"). Scored by emotion2vec on the render, render level and median F0 against the plain take, and WeSpeaker similarity to the references; 40 lines over both sets:

| arm | label matches the tag | non-neutral label | sim to references | level vs plain | median F0 vs plain |
|---|---|---|---|---|---|
| indextts2 + emotion audio | 9/40 | 29/40 | 0.494 | +2.6 dB | -0.8 st |
| zonos3 plain | 6/40 | 22/40 | 0.408 | 0 | 0 |
| zonos3 steered | 8/40 | 31/40 | 0.366 | +4.4 dB | +5.4 st |
| zonos3 + the line as reference | 5/40 | 25/40 | 0.426 | +0.2 dB | -0.4 st |

Steering does move the delivery (louder, higher, more often labelled non-neutral), but it moves every emotion the same way, towards aroused, and it costs another 0.04 of likeness on a voice that is already 0.09 behind. The line as an extra reference carries nothing measurable. emotion2vec agreement is a weak judge here (IndexTTS2 matches the tag on 9 of 40), so the 40 rows are in the listening page's section 2 for the shouting, crying and whispering verdict; I haven't listened to them.

## 3. Thin references and the shouted-only case

Pipeline cards as the tts stage made them (A2 tiers): on Sol Bianca five extras' cards under the floor or thin (1.2 to 7.2 s of reference, one or two clips) and the 9.1 s thin SPEAKER_01; on E12 the thin and short cards of characters 17, 19 and 21, three thin extras (1.5 to 3.2 s), and Young Makarov's floor card (3 clips, 14.4 s, all shouted: the #8 case). Each rendered on up to three of its own cues by both engines from the same references. Held-out is the card's other lines that overlap none of its references.

| set | renders per engine | indextts2: refs / held-out | zonos3: refs / held-out |
|---|---|---|---|
| Sol Bianca | 12 | 0.590 / 0.506 | 0.368 / 0.317 |
| Fairy Tail E12 | 18 | 0.509 / 0.371 | 0.303 / 0.239 |
| Young Makarov alone | 3 | 0.603 / 0.548 | 0.398 / 0.328 |

ZONOS3 is behind on every thin card, by more than on the full cards. It also fails outright on some: Sol Bianca c0052 (SPEAKER_10) came back as 0.04 s, and E12 c0121 (character 19, 3.0 s of reference) ran to 7.5 s for a 2.2 s IndexTTS2 line.

## 4. Duration

Render length over the cue's slot, on the first 40 sampled Sol Bianca cues and 31 on E12. "Pace" sets ZONOS3's speaking-rate condition row from the slot (the line's UTF-8 bytes over the slot length, the model's own pace input; Zyphra's LACI duration retakes need alignment heads and the server's retake loop and weren't tried):

| set, arm | median ratio | over the slot | over 1.25x | under 0.7x | mean abs log ratio |
|---|---|---|---|---|---|
| Sol Bianca, indextts2 | 0.98 | 20 | 11 | 9 | 0.313 |
| Sol Bianca, zonos3 | 0.82 | 12 | 6 | 14 | 0.432 |
| Sol Bianca, zonos3 paced | 0.76 | 6 | 1 | 12 | 0.298 |
| Fairy Tail E12, indextts2 | 0.81 | 9 | 1 | 10 | 0.268 |
| Fairy Tail E12, zonos3 | 0.69 | 5 | 0 | 17 | 0.540 |
| Fairy Tail E12, zonos3 paced | 0.70 | 0 | 0 | 14 | 0.378 |

Pace steering works as advertised for overruns: the takes timefit would have to squeeze over 1.25x go from 6 to 1 and 0 to 0, and nothing paced on E12 overruns. It doesn't stretch short takes (ZONOS3 is short already, and the bundle's notes say a reference-mode voice was validated only on the faster steered buckets). This is the one result in ZONOS3's favour, and it would only matter on a voice that passed section 1.

## 5. Speed and memory

The tts stage's render call timed per line, the worker's load and first line apart; peak VRAM is the worker's own process on GPU 0 (nvidia-smi):

| set, arm | s per line | real-time factor | first line | peak VRAM |
|---|---|---|---|---|
| Sol Bianca (39 lines), indextts2 tf32 | 4.78 | 1.67 | 7 s | 9.2 GB |
| Sol Bianca, zonos3 reference bf16 | 2.33 | 0.88 | 70 s | 9.8 GB |
| Sol Bianca, zonos3 reference f32 | 2.28 | 0.83 | 66 s | 11.6 GB |
| Sol Bianca, zonos3 embedding bf16 | 2.62 | 0.40 | 46 s | 3.9 GB |
| Fairy Tail (30 lines), indextts2 tf32 | 3.70 | 1.60 | 5 s | 9.1 GB |
| Fairy Tail, zonos3 reference bf16 | 1.95 | 0.97 | 65 s | 9.8 GB |
| Fairy Tail, zonos3 reference f32 | 1.71 | 0.80 | 61 s | 11.6 GB |

About twice IndexTTS2's speed a line, and bf16 buys nothing over f32 (the reference decodes one frame at a time in Python). The worker loads in about 11 s, and its first line takes another minute while the 8 GB dual-codec generator loads to encode the first reference; most of the reference mode's VRAM is that encoder. Zyphra's CUDA server with CUDA graphs would be faster still, but needs the toolkit (above).

## 6. Studio fit (Part B, #33 Task 19)

On a clone only the bundle's hidden-space sets apply: delivery (whisper, loud, tempo, pitch, conversational), expressive, disfluency, mood (seven axes), emotions (23 EARS moods) and word emphasis, each settable per character span with `control_spans` and released after an onset by steer-then-hold. Those would make reasonable audition parameters, and they are applied to the character's own clone, never a voice of their own (D41). The character-design sets (identity, pitch, texture, room and both accent sets) are raw speaker-embedding directions and need embedding mode, which can't speak English from a Japanese reference (section 1). Section 2 shows what the hidden controls cost: about 0.04 of likeness at the doses tried, on top of the 0.09 to 0.11 the clone already loses. With the base clone failing the gate, there is nothing to audition on, so I wrote no CD20 ledger note: no capability is recommended.

## What wasn't measured

- The SGLang-Omni and Zyphra CUDA-server runtimes (no CUDA toolkit on the host); speed is the reference runtime's.
- The full-size `full_mr` decoder (4 GB, not fetched); the KEEP-2 default was used throughout. It changes the decoder, not the codes the LM picks, so likeness shouldn't move much, but it is unmeasured.
- LACI duration targets and `speaker_cfg_scale` above 1.
- Listening verdicts: the page has 20 likeness rows, 40 emotion rows, 30 thin-reference rows and 12 pace rows; the call above rests on the numbers alone.

## Cleanup

Everything is under `/nvme/fukikae/tmp/zonos3/` (workspace copies about 8 GB, renders, venv about 7 GB, Zyphra's clone) plus the model snapshot at `/nvme/fukikae/hf-cache` (11 GB) with two symlinks into the default cache (`/nvme/huggingface/cache/hub/models--Zyphra--ZONOS3-Mini`, `/nvme/huggingface/cache/fukikae-zonos3`) and `.venvs/zonos3` and `.venvs/indextts2` symlinks in the worktree.
