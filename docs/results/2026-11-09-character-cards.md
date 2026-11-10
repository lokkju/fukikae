# Character cards and clone-by-default dubbing: results

_Run 2026-11-09 on branch `feat/character-cards` (Task 12 of `docs/superpowers/plans/2026-11-08-character-cards.md`). Every run used scratch copies of the workspaces under `~/fukikae-scratch/`; no original workspace changed._

## The pipeline as built

`fukikae run <workspace> <episode>` runs extract, separate, language, asr, diarize, characters, tag (SenseVoice events, emotion2vec emotion), translate, tts, timefit and mix, and muxes `out/<episode>.fukikae.mkv`.

- **Translate.** With `[translate] source = "auto"` it reads target-language subtitles when there is a dialogue track (ruling CD25, below) and otherwise translates the ASR transcript with a managed llama-server (Qwen3-8B Q6_K, pinned). Both paths budget each cue for the speech rate of the voice that will speak it (CD7, CD8): the backend of the character's card (IndexTTS2's 14.3 characters a second for a clone), else `[tts] clone_backend`'s for a character with no card yet, and Kokoro's (15.5) for extras. Until the final review the budgets used Kokoro's rate for every cue, so clone lines were budgeted about 8% long.
- **Voices.** Each recurring character speaks with a voice card. A character without one gets an automatic card the first time it needs a voice: a clone card of `[tts] clone_backend`, IndexTTS2 in every profile since the verdict (D24). The references are up to three of the character's cleanest turns on the separated vocals across the episodes processed so far, ranked by the CD5 rules. A character with under 6 s of clean speech gets a stock Kokoro voice from its gender's pool, and the report says how much it had. Extras always get stock voices. An automatic card never replaces an existing one (CD2, CD3, CD9).
- **Delivery.** IndexTTS2 receives the source line's audio as a separate emotion reference, scaled by the card's `emotion_strength`. It gets none where the tag stage marks laughter or crying (CD18).
- **Render guards.** A render far too long or too short for its text is retried once with seed + 1, then flagged as suspect (CD18).
- **Timefit and mix.** Rate fit and time-stretch share `max_speedup` (1.25), then borrowed silence. Loudness is matched by linear gain plus a true-peak limiter (CD10).
- **Cards are portable.** `voices export` and `voices import` move them as packs (Task 10), and `voices clone` makes a card from a clip you cut by hand.

## The candidates (Task 11 numbers)

The `eval tts` rows behind the blind test sampled 8 cues per character (seed 0), scored with deterministic ASR (CD16) and held-out speaker similarity (CD17). **The held-out similarity rests on 3 characters per set (n = 3), so gaps of a few hundredths are noise.**

Fairy Tail E12, ja to en, subtitle path:

| candidate | run | similarity held-out | ceiling | WER pooled | fit | cps | emotion agreement | suspect | WER > 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| kokoro (stock) | 0aabe773 | 0.054 | 0.536 | 0.044 | 0.944 | 16.8 | 0.10 | 0 | 4 |
| indextts2 | 0ae8795e | 0.304 | 0.536 | 0.074 | 0.889 | 14.6 | 0.31 | 1 | 4 |
| cosyvoice3 | eded422d | 0.251 | 0.536 | 0.061 | 0.903 | 16.4 | 0.17 | 0 | 5 |
| qwen3_tts | d89f2158 | 0.335 | 0.536 | 0.037 | 0.889 | 13.7 | 0.21 | 0 | 4 |

BBT S01E01, de to en, ASR path (duration is the dubbed line against the real English line):

| candidate | run | similarity held-out | ceiling | WER pooled | fit | cps | emotion agreement | WER > 0.5 | duration vs English |
|---|---|---|---|---|---|---|---|---|---|
| kokoro (stock) | 0879aa04 | 0.086 | 0.605 | 0.010 | 0.208 | 18.2 | 0.47 | 0 | 1.28 |
| indextts2 | 217d180d | 0.540 | 0.605 | 0.024 | 0.083 | 17.0 | 0.35 | 0 | 1.38 |
| cosyvoice3 | 0242d0d4 | 0.421 | 0.605 | 0.017 | 0.125 | 15.7 | 0.32 | 0 | 1.40 |
| cosyvoice3:zero_shot | 8c9be93d | 0.525 | 0.605 | 0.044 | 0.042 | 13.8 | 0.37 | 2 | 1.75 |
| qwen3_tts | 8e1f33eb | 0.242 | 0.605 | 0.010 | 0.167 | 18.1 | 0.40 | 0 | 1.37 |
| qwen3_tts:xvector | 21785803 | 0.372 | 0.605 | 0.024 | 0.167 | 16.6 | 0.24 | 0 | 1.47 |

The BBT fit column is the German dub budget at work: German lines run longer than the English original, and every candidate's English rendering is longer still, so only 4 to 21% of cues fit inside their slot after rate fit.

## The blind verdict

The operator ranked 12 cues (6 per set) and one 75 s scene by letter before opening the key (`~/fukikae-review/ab-2026-11/verdict.toml`, decoded with `fukikae.eval.ab.read_verdict`). `[overall]` in the file is still empty; the decision below is D24, recorded in the ledger.

| candidate | first places | points (3/2/1 for 1st/2nd/3rd) |
|---|---|---|
| indextts2 | 4 (scene included) | 20 |
| qwen3_tts (icl) | 3 | 17 |
| cosyvoice3:zero_shot | 2 | 10 |
| cosyvoice3 (cross_lingual) | 2 | 9 |
| qwen3_tts:xvector | 2 | 9 |
| kokoro (stock) | 0, never ranked | 0 |

Notes the operator left: FT characters 3 (cue02, c0154) and 2 (cue06, c0331) sound female though the lines are male; IndexTTS2 puts a wrong comma pause before "Wraith" in cue02.

## The decision (D24)

- Default clone backend: IndexTTS2 (`[tts] clone_backend = "indextts2"` in the default, anime and live-action profiles).
- Passed: IndexTTS2, Qwen3-TTS and CosyVoice 3. Either alternate can be given to a character with `voices assign` or `voices clone --backend`.
- No per-character overrides.

Stock stops being the default for characters with enough clean speech, so D12 resumes. Characters under 6 s still get stock, and the report says so.

## The full dubs with the new default

The workspaces were scratch copies: `~/fukikae-scratch/t12/fairytail` (anime profile, `[series]` set) and `~/fukikae-scratch/t12/bbt-de` (live-action). The Fairy Tail copy's registry was ECAPA, so I ran `fukikae registry reembed` on it first (382 exemplars, 13 centroids). Each episode ran as a single `fukikae run` on GPU 0. The muxed files are hard-linked under `~/fukikae-scratch/t12/out/`:

| file | path | cues spoken | characters (clone) | extras cues (stock) | renders retried / suspect | stretched | over after fitting | wall time |
|---|---|---|---|---|---|---|---|---|
| `fairytail-S01E11.fukikae.mkv` | subtitles (forced-flagged full track, CD25) | 341 | 9 | 38 | 10 / 4 | 72 | 7 (6.0 s) | 24 min |
| `fairytail-S01E12.fukikae.mkv` | subtitles | 354 | 12 | 5 | 7 / 1 | 78 | 6 (2.3 s) | 36 min |
| `fairytail-S01E13.fukikae.mkv` | subtitles | 251 | 7 | 83 | 5 / 0 | 45 | 2 (0.5 s) | 25 min |
| `fairytail-S01E14.fukikae.mkv` | subtitles | 323 | 9 | 3 | 6 / 0 | 65 | 3 (1.6 s) | 34 min |
| `bbt-de-S01E01.fukikae.mkv` | ASR + translator (142 over budget) | 334 | 3 | 25 | 5 / 0 | 260 | 51 (28.9 s) | 46 min |
| `bbt-de-S01E02.fukikae.mkv` | ASR + translator (139 over budget) | 273 | 4 | 5 | 5 / 2 | 223 | 55 (29.0 s) | 39 min |
| `bbt-de-S01E03.fukikae.mkv` | ASR + translator (136 over budget) | 338 | 4 | 168 | 2 / 0 | 161 | 48 (21.9 s) | 33 min |

- **Assignments.** 15 Fairy Tail characters and 4 BBT characters got automatic IndexTTS2 clone cards. Every one had at least 6 s of clean speech, so no character fell back to stock; the closest were 6.6 s (FT characters 8 and 15). Cards drew their clips from whichever processed episode had the most clean speech, so E11's characters often got clips from E12 to E14. Lines from `07-tts/assignments.log`:
  - `character 2: clone indextts2, card 152827cd049fb3e5, from 3 reference(s) in S01E14 (...; 13.5 s of 90.6 s eligible); auto and unlocked`;
  - five cards carry the CD5 warning `references may carry emotion: angry` (or happy).
- **Neutral-only renders**, where no emotion audio was passed, have two causes:
  - extras, whose stock Kokoro voices take none;
  - cues whose source span the tag stage marks as laughter (CD18). On BBT E03 that is 168 extras plus 42 laughter drops out of 338 cues. The sitcom's laugh track covers many lines.
- **BBT E03 extras.** 168 of E03's 338 cues went to extras because the characters stage left large clusters unmatched. This is the live-action profile's known German-dub weakness (`docs/results/2026-11-09-live-action-ground-truth.md`).
- **E11 used the subtitle path.** Under the old rule it would have used ASR (next section).

## The held-out row

`fukikae eval tts ~/fukikae-scratch/t12/fairytail --episodes S01E14 --backend indextts2 --split holdout --set fairytail --tag chosen`:

- run once with the committed defaults: 8 cues per character, seed 0, deterministic ASR (CD16), held-out similarity (CD17);
- E14 was never used in any tuning run or in the A/B;
- ledger run `f912c020`, exported with the rest of the `fairytail` tts rows to `2026-11-09-character-cards-tts.jsonl`.

| metric | value |
|---|---|
| characters / cues | 6 / 48 (11 left out for a slot under 1 s; characters 6, 9 and 11 under 6 s of clean speech in E14, so not cloned) |
| speaker similarity, held-out (headline) | 0.449 over 2 characters (6 clips, 36 s); ceiling 0.698 |
| speaker similarity to the references | 0.518 (6 characters) |
| WER pooled / median | 0.104 / 0.000; 5 renders over 0.5 (c0126, c0178, c0234, c0295, c0323) |
| fit within the slot / within budget | 0.75 / 0.94; stretch p90 1.19 |
| emotion agreement (emotion2vec) | 0.357 over 42 cues |
| render guard retries / suspect | 0 / 0 |

Against the E12 tuning row (held-out 0.304, ceiling 0.536, WER 0.074, fit 0.889, emotion 0.31), the held-out similarity is higher and so is its ceiling. Held-out similarity covers only 2 characters, so it is as noisy as the n = 3 rows. WER is worse, and most of that comes from character 1, a high-pitched comic voice: WER 0.24 and 9 cps against 13 to 16 for the others. Nothing in this row was re-tuned afterwards.

The production cards overlap this row. They drew some clips from E14 (characters 1, 2, 7, 13 and 15), because the Fairy Tail copy had E14's artifacts before the dubs ran. The eval builds its own clone cards from E14 with the same CD5 rules, so for the two characters with a held-out figure (2 and 13) its references and the production references come from the same episode, and partly from the same turns. The held-out exemplars still exclude the eval's own picks and every rendered cue's delivery span (CD17), so the 0.449 is not similarity to the prompt. It does mean E14 is no longer a clean test of the production voices: what was cloned from E14 is scored on E14. The row measures IndexTTS2 cloning from one episode and speaking that episode's lines. That is what the production path does for a character first heard in that episode; it is not a cross-episode generalisation figure.

## Operator notes

### The female-sounding characters 3 and 2 (FT E12 cue02 and cue06)

In both clips the clone copies the wrong speaker. Each cue's character is a cluster that merged several voices, and the reference clips came from someone other than the cue's speaker. I checked every pick against the official English subtitles' speaker names (Crunchyroll ASS `Name` field). The clips are in `~/fukikae-review/t12-refs/`, readable by the owner only.

- **cue02 (c0154), "Don't you die, Wraith!", spoken by YOUNG MAKAROV.** In the A/B workspace (`~/fukikae-scratch/t7/e12-ws`), character 3 is diarize cluster SPEAKER_13. That cluster is mostly the opening theme's singer (87 to 165 s) plus single lines from Young Makarov, Natsu, Laxus, Juvia, Lucy, Wraith and Wendy. The two picks:
  - 654.44 to 659.69 s, transcribed as one word, "くそ". Young Makarov says "Damn it!" in the first 1.9 s; the remaining 3.4 s is no speech. File: `ab_c3_654.4-659.7.wav`.
  - 442.64 to 445.85 s: Natsu, "My friends aren't your vessels". File: `ab_c3_442.6-445.9.wav`.
- **cue06 (c0331), "I'd even settle for Jellal!", spoken by NATSU.** Character 2 is SPEAKER_12. By subtitle time it holds Charle, Wendy and Happy (all voiced by women) and two Natsu spans. The two picks:
  - 1057.10 to 1060.19 s: Charle, "Don't worry about me and go after that guy". File: `ab_c2_1057.1-1060.2.wav`.
  - 1300.30 to 1305.87 s: one diarize turn holding Happy and Natsu. File: `ab_c2_1300.3-1305.9.wav`.

The fresh full run shows the same thing at series level. Matching every Fairy Tail cue to its subtitle speaker across E11 to E14:
- character 2 is Wendy 67, Happy 65, Charle 25, Natsu 15;
- character 3 is Natsu 29 plus Kiria, Erza, Happy and Wendy;
- character 9 is the main Natsu (95 of 150).

In the full run, character 2's card was cloned from Wendy and Charle in E14 (`fullrun_char2_*.wav`) and character 3's from Touka in E13 (`fullrun_char3_*.wav`). So c0154 and c0331 sound female in the full dubs too.

**What I fixed.** The one selection bug: a reference turn whose transcript is far sparser than the speaker's usual rate, and sparse in absolute terms, is now dropped (`select.SPARSE_*`). The "くそ" turn was 0.4 characters a second against the speaker's median of about 5. I also made the stock gender estimate skip theme songs and tagged singing. Character 3 had been read over the opening's singer.

**What that changes for these two cues** (`c0154_char3_fixed.wav`, `c0331_char2_fixed.wav`):
- character 3 keeps only Natsu's 3.2 s, falls under 6 s, and goes to stock. Its F0 of 224 Hz falls in the anime profile's unknown band, so it gets `af_heart`, still a female voice;
- character 2's picks are unchanged, because the Charle clip's text rate is normal.

**What I tried and didn't ship.** A consistency check against the character's embedding centroid does not separate these voices:
- Charle's three spans sit at 0.68 to 0.80 to each other and at 0.72 to 0.75 to the centroid;
- the Natsu spans sit lower;
- character 3's picks (0.61, 0.53) are inside its own cluster's spread (0.25 to 0.74).

The centroid is the merged voice, so preferring it reinforces the majority speaker, who is the wrong one for Natsu's lines. Neither fix is a reference-selection rule, because the voices merge upstream:
- on the subtitle path with named lines, split or check characters by the subtitle speaker name;
- the operator can split the cluster in the registry, then `voices clone --clip`.

Until then, `voices clone <ws> <character> --clip <natsu.wav>` gives a character a hand-picked voice. It cannot give one merged character two voices.

### IndexTTS2's comma pause before "Wraith" (cue02)

The text sent was `Don't you die, Wraith!`, the official subtitle's line unchanged:
- the translate stage passed it verbatim, with no line break and no quote marks;
- the worker's special-token strip (CD20) leaves commas alone;
- IndexTTS2's own `TextNormalizer` returns the same string.

The vocative comma is correct English. IndexTTS2 renders it as a 170 ms silence ("Don't you die" from 0.35 to 1.17 s, "Wraith" from 1.34 s), where Qwen3-TTS and CosyVoice 3 run straight through. The pause comes from the model, not from fukikae. I did not strip vocative commas: that would change the text for every backend to hide one model's prosody.

## Ruling CD25: target subtitles mean the dialogue track

`media.dialogue_track` picks the embedded target-language subtitle track for `source = "auto"`:
- a track titled forced is never chosen, nor one titled signs or songs unless the title also says full, dialogue or complete ("Full Subtitles (incl. Signs)", "Dialogue + Songs"); such a track still needs a known, dialogue-sized count;
- a track with fewer than 3 cues a minute, or under 40% of the densest plausible dialogue track (unflagged, title not ruling it out), is skipped. That track's rate is capped at 30 cues a minute when it sets the bar, and a track denser than the cap ranks after the plausible ones, so a typeset track with a thousand styled events neither sets the bar nor wins;
- a forced flag is believed unless the track reaches 8 cues a minute as well as the measures above;
- a flagged track of unknown size is skipped;
- plain tracks win over CC/SDH, then unflagged tracks win.

Cue counts come from the container's `NUMBER_OF_FRAMES` statistics tag. Without it, density can't be judged and the old order stands. If no track qualifies, the episode falls back to ASR.

On Fairy Tail:
- **E11's only English track** is flagged forced, titled "English", with 344 cues (14.4 a minute; E12's dialogue track has 357). I decided every line beats the flag, so E11 now dubs from the subtitles.
- **E12 to E14** skip the 3-cue "Forced" track and choose the plain dialogue track over E13's and E14's CC tracks.
- **BBT's English tracks** are image subtitles, so BBT stays on the ASR path.

## Known limits

- **The German dub budget.** German lines are longer than the slots of the English original they replaced. The translator ends 40 to 51% of BBT cues over budget. After rate fit, time-stretch at the 1.25 cap, and up to 63 s of borrowed silence per episode, 48 to 55 lines an episode still play past their slot (22 to 29 s in all).
- **Anime voice merging.** Diarization and the characters stage merge voices, as above (Natsu with Happy, Charle and Wendy), and a clone copies whatever voice dominates its references. That is why the female-sounding lines persist. Fixing it is identity work.
- **The female-sounding characters.** The selection fix removes no-speech turns but cannot fix merged identities. Characters 2 and 3 still sound female on Natsu's and Young Makarov's lines.
- **Fingerprints (ruling CD26).** Automatic clone cards are not fingerprinted when they are made, because that would load the clone model during the stage's key computation. `voices export` records a fingerprint for every exported card that lacks one: one render of the test script on the card's backend, under the GPU lock. When that backend can't run on the machine, or the render fails, export lists the cards it ships without one (`--no-record` skips the renders and lists them too), and `--require-fingerprints` refuses the export instead. `voices verify --record` records one at any time. This departs from the plan text, which asked for `verify_card(record=True)` at assignment.
- **`voices clone` takes `--transcript`.** It runs no ASR pass of its own, which also departs from the plan.
- **Unpinned runtime downloads.** Ten older backends still download an unpinned model on a cache miss: ASR (qwen3, faster_whisper), alignment (qwen3, wav2vec2), pyannote diarization, dialogue separation, and the wespeaker, ecapa, wavlm and sherpa embedders. This is open follow-up work; pinning them changes cache keys.
- **Unknown-gender stock voices.** A stock fallback for a character in the anime profile's unknown band (165 to 300 Hz) alternates pools starting with the female one.

## Follow-ups

- **D26 (done):** the performance task: tf32, the tag stage on the GPU, overlapping consecutive episodes, and a rate estimate before the first render. Its timings and the eval gate are in the ledger. The wall times above predate it.
- **D27 (proposed next):** subtitle speaker names for identity. On the subtitle path with named lines (Crunchyroll ASS), use the names to split or check characters, which is the fix for the merged voices described above.
- **D28 (operator, 2026-11-10):** further IndexTTS2 speed levers. Lever 1, rendering grouped by card, is done and measured below; it saves about 2%. Lever 2 is the TensorRT backend trial.
- The ten older backends that download on a cache miss (above).

## D28 lever 1: renders grouped by card

The tts stage now renders each backend's jobs one card at a time (cards in order of first appearance in the episode, each card's cues in cue order; `render_order` in `stages/tts.py`). IndexTTS2 caches the conditioning of one speaker prompt, and a change of speaker reruns w2v-BERT, the semantic codec, CAMPPlus and the mel over the 15 s prompt, plus a `torch.cuda.empty_cache()`. In cue order E14's 320 IndexTTS2 lines change speaker 114 times; grouped, 8 times (9 cards).

Nothing about the output depends on the order. Every render is seeded inside the worker before `infer`, `SpeechRates` keeps its renders per card (and each card still sees its own cues in cue order, so every estimate, factor and cache key is the same), the retry is per job, and the plan was already written in cue order. `run.json` now lists `rendered` and `cached` in cue order instead of render order. The emotion conditioning is not shared between lines in either order: each cue passes its own `ref/<cue-id>.wav`, and IndexTTS2 compares emotion prompts by path.

I measured on scratch copies of the smoke workspace (`~/fukikae-scratch/group-tts/{main,branch}`, driver `~/fukikae-scratch/group-tts/driver.py`): the 40 consecutive spoken E14 cues c0026 to c0065, 7 cards, 20 speaker changes in cue order and 6 grouped (the other cues stay cache hits, so the worker sees only these 40). The driver deletes their cache entries, runs the stage on GPU 0 and times each `render_cue` call; the worker's load (about 41 s, inside the first call) is taken out. Two runs each, alternating:

| code | run 1 | run 2 | mean |
|---|---|---|---|
| main (5a9c2f0) | 3.25 s/line | 3.41 s/line | 3.33 s/line |
| grouped | 3.27 s/line | 3.32 s/line | 3.29 s/line |

All 40 wavs are byte-identical (sha256) across all four runs, and every cache key is the same. The 1% difference is inside the run-to-run spread (main's two runs differ by 5%). A fit over all 156 timed lines (time against rendered audio seconds and whether the speaker changed) puts a speaker change at 0.23 ± 0.05 s, against a 1.30 s fixed cost and 1.01 s per second of audio. For a whole E14 that is 106 fewer changes, about 24 s of roughly 18 minutes of rendering. The research estimate assumed the prompt conditioning was a large share of the 0.35 s "other" per line; on the 3090 it is about 0.2 s, and the per-line emotion pass (also w2v-BERT, over the cue's span) runs whatever the order.

## D28 lever 2: TensorRT

I ran IndexTTS2 through upstream's TensorRT backend (index-tts `backends/trt` at our pin `d9e41aac`, from PR #733 and arXiv 2607.21042) on GPU 0, the 3090 at its 200 W limit. It is twice as fast, and its voices are measurably further from the characters, so it stays opt-in and off.

**What it accelerates.** Everything, not just the GPT. The GPT runs on TensorRT-LLM (beam search with 3 beams and upstream's top-k 30, top-p 0.8, temperature 0.8 and repetition penalty 10), and nine more networks run as plain TensorRT fp16 engines: w2v-BERT, the semantic codec, CAMPPlus, the speaker and emotion perceiver conditioners, the latent projector, the length regulator, the diffusion decoder's DiT and BigVGAN. The weights are the same snapshot, so every IndexTTS2 card renders unchanged. The engines are fp16 at batch 1, the one configuration upstream verified.

**How it runs.** `indextts2_trt` is a tts backend that renders `indextts2` cards (it declares `card_backend = "indextts2"`). A workspace opts in with `[tts] engines = { indextts2 = "indextts2_trt" }`; `eval tts --backend indextts2_trt` and a tts-ab candidate `indextts2_trt` work the same way. The renders carry the engine's name and identity in the cache key (the project digest and the engines' build id), so they never mix with PyTorch renders: on the 40 timing cues all 40 keys differ. TensorRT-LLM needs OpenMPI, which the host doesn't carry, so the worker runs in a local image built from `workers/indextts2_trt` (CUDA 12.8 runtime base and uv pinned by digest, Python 3.12, TensorRT-LLM 0.21, torch 2.7.1, the locked environment, upstream's code cloned at the pin), never pushed. fukikae holds the GPU 0 lock as for any worker and starts `docker run -i --rm` speaking the same JSON lines: one GPU, no network, my uid, no capabilities, no-new-privileges, a pids limit, a read-only image with `/tmp` in memory, the Hugging Face cache and the engines read-only, fukikae's source read-only, and one writable directory, the worker's own, into which each line's reference clips and emotion span are copied and from which the render is copied out through a descriptor opened without following links. The worker gets `FUKIKAE_GPU_INDEX`, the host's index of its card, so the GPU lock's leftover scan on another card doesn't count it (it carries no `CUDA_VISIBLE_DEVICES`). The container is removed by name before the lock is released, and a load removes containers left by a fukikae that is gone. `fukikae backends setup indextts2_trt` checks docker, its NVIDIA runtime and the disk, builds the image, fetches the pinned models and builds the engines under the lock into a fresh directory that replaces the old set only once complete; engines already built for this digest and compute capability are kept unless `--force`.

**Setup cost.** The first image build took 14 minutes (mostly downloading TensorRT-LLM and torch); a rebuild after a lock change takes 2 to 4. Exporting the nine ONNX graphs takes about 9 minutes, converting the GPT checkpoint 46 s, and building the engines 16 minutes (BigVGAN 263 s, CAMPPlus 205 s, the DiT 118 s, w2v-BERT 90 s, the GPT 59 s, the rest 4 minutes). The engines are 3.1 GB (GPT 1.03 GB, w2v-BERT 0.84 GB, the speaker conditioner 0.41 GB, the emotion conditioner 0.33 GB, BigVGAN 0.23 GB, the DiT 0.20 GB, the rest 0.08 GB) under `~/.cache/fukikae/indextts2_trt/`, the image 23.6 GB. Upstream's dependency list didn't build on its own: I pinned the ONNX tools to their versions on upstream's verification date (onnx 1.20.1, onnxscript 0.7.1, onnx-ir 0.2.1; onnx-ir 1.x breaks torch 2.7's exporter, and older onnxscript can't translate the DiT), pyarrow below NumPy 2, added openai-whisper and fugashi (indextts's tokenizer imports them), and gave the image gcc (Triton compiles a helper when TensorRT-LLM is imported) and libpython. The engines take 15.1 GiB of VRAM (15,438 MiB, the nvidia-smi peak, context included), against 9.3 GB for the PyTorch worker, because TensorRT-LLM reserves its KV cache up front.

**Speed.** Same method as lever 1: the 40 consecutive spoken E14 cues c0026 to c0065 on a scratch copy of the smoke workspace (`~/fukikae-scratch/tensorrt/ws`, driver `~/fukikae-scratch/tensorrt/driver.py`), every other cue skipped and the stage run against an empty tts cache, each `render_cue` call timed and the worker's load taken out. Two runs each, alternating.

| backend | run 1 | run 2 | mean | load | real-time factor | audio |
|---|---|---|---|---|---|---|
| indextts2 (PyTorch, tf32) | 3.31 s/line | 3.23 s/line | 3.27 s/line | 39 s | 1.71 | 76.6 s |
| indextts2_trt (fp16 engines) | 1.48 s/line | 1.45 s/line | 1.47 s/line | 28 s | 0.84 | 69.8 s |

That is 2.23x a line, or 2.04x on the real-time factor (render time over audio), against upstream's 2.39x on a 4090. Both are deterministic: each backend's two runs are byte-identical on all 40 wavs. The TensorRT renders are 9% shorter overall (median per cue 0.93 of the PyTorch length, range 0.61 to 1.30), so part of the per-line gain is shorter speech. For a whole E14 (320 IndexTTS2 lines) the saving is about 9.5 minutes of roughly 17.

**Quality gate.** `eval tts` on E14 with each character's assigned card, deterministic ASR and held-out similarity, both backends on the same cues and cards, at 4 and 8 cues a character (ledger `~/fukikae-scratch/tensorrt/ledger.jsonl`):

| run | cues | held-out sim (ceiling) | sim to references | sim to card | WER | median WER | fit | stretch p90 | cps | suspect |
|---|---|---|---|---|---|---|---|---|---|---|
| indextts2, 4/char, `09d59ed3` | 36 | 0.406 (0.651) | 0.493 | 0.491 | 0.105 | 0 | 0.83 | 1.03 | 14.6 | 1 |
| indextts2_trt, 4/char, `08149dee` | 36 | 0.344 (0.651) | 0.468 | 0.482 | 0.092 | 0 | 0.78 | 1.07 | 14.5 | 0 |
| indextts2, 8/char, `5fffa9b3` | 67 | 0.438 (0.673) | 0.494 | 0.453 | 0.103 | 0 | 0.82 | 1.05 | 14.3 | 1 |
| indextts2_trt, 8/char, `cb6a2473` | 67 | 0.410 (0.673) | 0.474 | 0.443 | 0.081 | 0 | 0.82 | 1.05 | 14.6 | 0 |

The smoke run's own PyTorch row (0.432 held-out, ceiling 0.651, WER 0.104) was made before this branch; my PyTorch rows on the same cues come out at 0.406 and 0.105, so I compare the two backends within this run. Every similarity figure is lower on TensorRT in both samples: held-out -0.062 and -0.028, similarity to the references -0.025 and -0.020 (9 and 6 characters, so less noisy than the held-out figure's 3 and 2), similarity to the card -0.009 and -0.010. Character 13 loses 0.06 held-out in both samples and character 15 0.08 against its references. The `accel` engine failed this gate at -0.021 and -0.024 held-out. WER improves (0.092 against 0.105, 0.081 against 0.103) and the render guard flagged nothing, where PyTorch keeps its one long render (c0072). Fit is worse in the 4-a-character sample (0.78 against 0.83) and level in the 8-a-character one (0.82 both); speech rate is level.

Per cue, the PyTorch and TensorRT renders of the same line on the 40 timing cues have a WeSpeaker cosine of 0.650 mean (median 0.678, lowest 0.385 on c0035, 10th percentile 0.472). For scale, two different lines of the same card score 0.501 within PyTorch and 0.460 across the two backends: the same line on the other engine is closer than another line on the same engine, but the TensorRT voice sits a little further from the PyTorch one. Identical output was never expected (fp16 engines, and TensorRT-LLM's own beam search where IndexTTS2 samples within its beams).

**Capabilities.** `indextts2_trt` declares `emotion_audio` because the PyTorch worker does, without an A/B of its own (CD20 asks a backend for one). The evidence is indirect: the same model snapshot, and an impl that hands the cue's source span to upstream's pipeline as the emotion prompt with `emotion_strength` as `emo_alpha`, as the PyTorch impl does; the quality gate above ran with the emotion prompt on. Reopening lever 2 means an emotion A/B on the engine too.

**Recommendation.** Keep the PyTorch worker as the default and don't build an A/B pack: the speed is real, but the similarity loss is consistent across every measure and both samples and at least as large as the one that failed `accel`, so the gate fails before the listening checkpoint. The backend stays as an opt-in for a run where speed matters more than likeness. The likeliest causes are the fp16 conditioners (w2v-BERT, the perceivers and CAMPPlus all shape the voice, and PyTorch runs them in fp32 with tf32 matmuls) and TensorRT-LLM's beam search; an fp32 build of the non-GPT engines (upstream's `PRECISION=fp32` path for those) is the next thing to measure if lever 2 is reopened.

## How to run it

```bash
fukikae backends setup indextts2          # once: worker environment and pinned models (the only download)
fukikae backends setup kokoro             # and sensevoice, emotion2vec, openai_compat
# fukikae.toml: profile = "anime" (or "live-action"); [tts] clone_backend is indextts2 by default
fukikae run <workspace> S01E11            # automatic clone cards on first need; out/S01E11.fukikae.mkv
fukikae voices list <workspace>           # who has which card; voices show / assign / clone / lock
```
