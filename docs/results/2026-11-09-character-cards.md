# Character cards and clone-by-default dubbing: results

_Run 2026-11-09 on branch `feat/character-cards` (Task 12 of `docs/superpowers/plans/2026-11-08-character-cards.md`). Every run used scratch copies of the workspaces under `~/fukikae-scratch/`; no original workspace changed._

## The pipeline as built

`fukikae run <workspace> <episode>` runs extract, separate, language, asr, diarize, characters, tag (SenseVoice events, emotion2vec emotion), translate, tts, timefit and mix, and muxes `out/<episode>.fukikae.mkv`.

- **Translate.** With `[translate] source = "auto"` it reads target-language subtitles when there is a dialogue track (ruling CD25, below) and otherwise translates the ASR transcript with a managed llama-server (Qwen3-8B Q6_K, pinned). Both paths budget each cue for the TTS backend's speech rate (CD7, CD8).
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

The workspaces were scratch copies: `~/fukikae-scratch/t12/fairytail` (anime profile, `[series]` set) and `~/fukikae-scratch/t12/bbt-de` (live-action). The Fairy Tail copy's registry was ECAPA, so I ran `fukikae registry reembed` on it first (382 exemplars, 13 centroids). Each episode ran as a single `fukikae run` on GPU 0. The muxed files are hard-linked under `/home/lokkju/fukikae-scratch/t12/out/`:

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
- a track titled signs, songs or forced is never chosen;
- a track with fewer than 3 cues a minute, or under 40% of the densest track of the language, is signs-and-songs-sized and skipped;
- a forced flag is believed unless the track is dialogue-sized by both measures;
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
- **Fingerprints.** Automatic clone cards are not fingerprinted when they are made, because that would load the clone model during the stage's key computation. `voices verify --record` (or `voices export`) records one later. This departs from the plan text, which asked for `verify_card(record=True)` at assignment.
- **`voices clone` takes `--transcript`.** It runs no ASR pass of its own, which also departs from the plan.
- **Unpinned runtime downloads.** Ten older backends still download an unpinned model on a cache miss: ASR (qwen3, faster_whisper), alignment (qwen3, wav2vec2), pyannote diarization, dialogue separation, and the wespeaker, ecapa, wavlm and sherpa embedders. This is open follow-up work; pinning them changes cache keys.
- **Unknown-gender stock voices.** A stock fallback for a character in the anime profile's unknown band (165 to 300 Hz) alternates pools starting with the female one.

## How to run it

```bash
fukikae backends setup indextts2          # once: worker environment and pinned models (the only download)
fukikae backends setup kokoro             # and sensevoice, emotion2vec, openai_compat
# fukikae.toml: profile = "anime" (or "live-action"); [tts] clone_backend is indextts2 by default
fukikae run <workspace> S01E11            # automatic clone cards on first need; out/S01E11.fukikae.mkv
fukikae voices list <workspace>           # who has which card; voices show / assign / clone / lock
```
