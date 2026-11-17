# Gender bands on the median F0: calibration (identification workflow, Task 2)

_Run 2026-11-11 on CPU, branch feat/identification-part-a, estimator `fukikae.pitch` VERSION 2 (commit ae11bee). Tuning sets: Fairy Tail E11-E13 (ja, `~/fukikae-scratch/t12/fairytail`), The Big Bang Theory S01E01-E02 (en, `~/fukikae-ws/bbt-en`), Kaze S01E01-E03 (ja, `~/fukikae-ws/kaze`). Sol Bianca S00E01 (`~/fukikae-scratch/sol-bianca`) is reported and not tuned on; Fairy Tail E14 was not read. Records: `2026-11-11-gender-bands.jsonl` (2895 rows: per ground-truth line, per character pooled over the set's episodes, per diarize cluster, each with the A0 and pYIN medians and voiced seconds)._

**Verdict: no pair of bands passes the acceptance rule on all three sets, and 165/200 Hz fails on each of them.** The anime profile's `[tts.gender]` override was left in place for the operator's decision (the plan's Step 4 stop); ruling D44 has since removed it (below). The defaults in `fukikae.pitch` and `cards.select` are 165/200 Hz (D43), which is the only pair that puts no main character of BBT or Kaze in the other class.

## Caveats

- Natsu's truth is `m` by ear: a male voice actor, shouting through most of this arc (confirmed in the Task 2 review). The Fairy Tail failure is the limit of a median on shouted speech, not a truth error or an estimator error.
- The low-voice finding that motivated the window correction came from Sol Bianca (SPEAKER_08 read 199.5 Hz against pYIN's 113), so Sol Bianca is not a clean held-out set for the estimator, though the bands were never tuned on it. Fairy Tail E14 stayed untouched by both the fix and the calibration; the reviewer measured 0.57 st mean error against pYIN there (11 clusters).
- Per-character medians pool every ground-truth line of the character, shouted lines and lines that overlap another speaker included, so a character's median is the register of all their speech as subtitled, not of their calm speech.
- The estimator here is VERSION 2. VERSION 3 (Task 3's first commit) adds a valley check that keeps low rumble unvoiced (brown noise: 16.7% voiced at about 271 Hz before, under 5% after); against pYIN it changes Fairy Tail E14 not at all (0.572 st) and Sol Bianca from 0.402 to 0.401 st, so these tables stand.
- VERSION 4 (the Task 3 review fixes, 7c64a44) adds per-candidate loudness groups and the loud-frame voiced share; without groups, as every measurement here reads it, a track is unchanged. I didn't rerun the calibration; instead I compared VERSION 3's `f0_track` (from 1885f93) with VERSION 4's on Fairy Tail E12's vocals (60 s, three turns of 3.5 to 30 s, and 1290 s whole): every frame identical, 64,499 frames on the long span. The tables stand at VERSION 4 too.

## Method

`scripts/gender-calibration/calibrate.py measure` reads each episode's ground-truth subtitle lines (Fairy Tail: the en ASS Name field, E11 extracted from the release's English track, E12 and E13 the cached tracks the names arm used; BBT: the transcript-aligned named ASS in `~/fukikae-ws/subs/bbt/labels`; Kaze: the retimed colour captions in `~/fukikae-ws/subs/kaze-kaoru/aligned` with the workspace's colour map and aliases), and for Sol Bianca the ASR transcript's segments, which carry no names. Per line it runs A0 (`f0_track`) and pYIN (librosa, 60 to 1000 Hz, 2048-sample frames, 20 ms hop, through `uv run --with librosa`); per character it pools the voiced frames of all its lines over the set's episodes; per diarize cluster it reads at most 120 s of turns, as `cards.select.speaker_f0` does. `report` classes every median at the bands given (speaker minimum 1.0 s voiced, line minimum 0.4 s) and applies the plan's rule: no main character (30 s or more) in the other class, at most 10% of main characters' seconds unknown, and at least 90% per-line agreement with pYIN where both are known. `--sweep` tries every band on a 5 Hz grid from 120 to 320 Hz. Truth is `scripts/gender-calibration/truth.toml`, voice register as heard, filled from the names and the casts; entries marked `# check` are mine to confirm by ear. 真風 is the narrator and fortune teller played by 研ナオコ (cast lists), so `f`.

## The estimator fix that came first (Task 1 review)

The review found low voices reading an octave or more high (Sol Bianca SPEAKER_08 199.5 Hz against pYIN's 113). I compared peak rules against pYIN's per-cluster median (up to 30 turns of 1 s or more per diarize cluster) on the tuning sets only. None of the three proposed rules fixed the cause: the Hann window's taper shrinks the autocorrelation at a long lag, so a low voice's period peak falls under its half-period peak. Dividing the autocorrelation by the window's own (Boersma's correction) does; the share was then picked by the same comparison. Mean absolute error in semitones (74 tuning clusters; "speech only" drops two BBT clusters where pYIN reads laughter or effects at 675 and 789 Hz, which every rule misses by 8 to 25 st):

| rule | tuning mean | median | clusters > 2 st | pYIN > 400 Hz (5) | pYIN < 160 Hz (18) | speech only | Sol Bianca (not tuned) |
|---|---|---|---|---|---|---|---|
| Task 1: shortest lag within 0.9 | 1.16 | 0.37 | 12 | 0.57 | 1.53 | 0.84 | 2.02 |
| (a) best peak | 1.11 | 0.42 | 8 | 1.14 | 1.18 | 0.73 | 1.51 |
| (b) within 0.9, not best/k | 1.11 | 0.34 | 10 | 0.94 | 1.31 | 0.76 | 1.74 |
| (c) within 0.95 | 1.11 | 0.34 | 11 | 0.86 | 1.33 | 0.76 | 1.77 |
| window-corrected, within 0.9 | 1.05 | 0.30 | 5 | 1.17 | 0.40 | 0.56 | 0.70 |
| window-corrected, best peak | 2.19 | 1.25 | 16 | 2.57 | 1.23 | 1.68 | 1.73 |
| window-corrected, within 0.85 | 0.91 | 0.31 | 4 | 0.78 | 0.38 | 0.46 | 0.44 |
| **window-corrected, within 0.8 (shipped)** | **0.88** | 0.32 | 5 | 0.65 | 0.46 | 0.49 | 0.33 |

The shipped rule has the lowest tuning mean and keeps the five voices over 400 Hz within 1.3 st (Fairy Tail 548, 465, 443, 428 and 401 Hz read 533, 444, 432, 421 and 372). On Sol Bianca it reads SPEAKER_07, 08 and 11 at 90, 116 and 88 Hz (pYIN 89, 118 and 89; Task 1 read 157, 200 and 152).

## Results at 165/200 Hz

### Fairy Tail E11-E13: FAIL (Natsu)

| character | truth | seconds | A0 | class | pYIN | class |
|---|---|---|---|---|---|---|
| LAXUS | m | 121 | 128 | m | 123 | m |
| MAKAROV (WRAITH) | m | 37 | 129 | m | 132 | m |
| ZEREF | m | 128 | 139 | m | 134 | m |
| WRAITH | m | 244 | 150 | m | 151 | m |
| GRAY | m | 75 | 151 | m | 147 | m |
| JELLAL | m | 54 | 160 | m | 157 | m |
| IGNEEL | m | 46 | 165 | m | 175 | u |
| NEBARU | m | 65 | 170 | u | 168 | u |
| ERZA | f | 176 | 213 | f | 204 | f |
| KIRIA | f | 102 | 258 | f | 266 | f |
| CANA | f | 89 | 267 | f | 257 | f |
| NATSU | m | 328 | 267 | **f** | 274 | f |
| LISANNA | f | 48 | 281 | f | 281 | f |
| YOUNG MAKAROV | f | 37 | 296 | f | 313 | f |
| WHITE MAGE | f | 76 | 327 | f | 317 | f |
| JUVIA | f | 58 | 381 | f | 372 | f |
| LUCY | f | 197 | 381 | f | 390 | f |
| HAPPY | f | 146 | 457 | f | 472 | f |
| TOUKA | f | 147 | 457 | f | 506 | f |
| WENDY | f | 48 | 533 | f | 584 | f |

| truth \ A0 (s) | m | u | f |
|---|---|---|---|
| m | 704 | 65 | 328 |
| f | 0 | 0 | 1125 |

One main character in the other class (Natsu), 2.9% of main seconds unknown (Nebaru), line agreement with pYIN 99.6% (733 lines), 14.0% of line seconds unknown (in the band or under 0.4 s voiced). No band passes: Natsu's lines in this battle arc are mostly shouted (pYIN line medians: 58 of 121 over 250 Hz, 25 over 400), and his median sits above Erza's (204 to 213 Hz), so any band that makes him `m` makes her `m` too. pYIN agrees with A0 on him; this is the median's limit, not the estimator's, and his truth stays `m` (by ear).

### The Big Bang Theory S01E01-E02: FAIL (unknown share)

| character | truth | seconds | A0 | class | pYIN | class |
|---|---|---|---|---|---|---|
| Howard | m | 133 | 148 | m | 144 | m |
| Raj | m | 41 | 158 | m | 145 | m |
| Sheldon | m | 649 | 182 | u | 169 | u |
| Leonard | m | 643 | 195 | u | 178 | u |
| Penny | f | 308 | 302 | f | 292 | f |

| truth \ A0 (s) | m | u | f |
|---|---|---|---|
| m | 174 | 1291 | 0 |
| f | 0 | 0 | 308 |

Nobody in the other class, 72.8% of main seconds unknown (Sheldon and Leonard), line agreement 97.0% (429 lines), 29.9% of line seconds unknown. The two leads are tenors whose medians pYIN also puts inside 165-200. On this set alone the passing band nearest 165/200 is a single cut at 200 Hz (line agreement 90.4%). A0 reads both leads 1.3 to 1.6 st above pYIN on this set.

### Kaze E01-E03: FAIL (unknown share)

| character | truth | seconds | A0 | class | pYIN | class |
|---|---|---|---|---|---|---|
| 信右衛門 | m | 49 | 162 | m | 164 | m |
| 真風 | f | 173 | 198 | u | 204 | f |
| 直美 | f | 166 | 246 | f | 246 | f |
| りん | f | 516 | 258 | f | 265 | f |

| truth \ A0 (s) | m | u | f |
|---|---|---|---|
| m | 49 | 0 | 0 |
| f | 0 | 173 | 682 |

Nobody in the other class, 19.1% of main seconds unknown (真風, a low female narrator at 198 Hz), line agreement 96.3% (188 lines), 22.8% of line seconds unknown. On this set alone the nearest passing band is 165/195.

### Sol Bianca S00E01 (reported, not tuned on)

No names, so no truth: line agreement with pYIN 95.9% (513 lines), 9.6% of line seconds unknown. Clusters of 30 s or more: SPEAKER_02 421 Hz (pYIN 397), 03 271 (269), 04 222 (225), 05 232 (239), 06 333 (341), 12 254 (246), 14 239 (246) and 16 432 (438) are `f`; SPEAKER_07 93 (89) is `m`; SPEAKER_13 200 (203), the mixed cluster, and SPEAKER_15 193 (194), the hallucination, are `u`, as the spec expected.

## Other bands tried

| bands | Fairy Tail | BBT | Kaze |
|---|---|---|---|
| 165/180 (Task 1's `cards.select`) | Natsu other; 2.9% u | **Sheldon and Leonard other** | pass |
| 165/200 (D43) | Natsu other; 2.9% u | 72.8% u | 19.1% u |
| 165/300 (anime override, now on the median) | 38.0% u | 72.8% u | 94.6% u |
| 175/210 | Natsu other; 0% u | 72.8% u | 19.1% u |
| 197/197 (single cut) | Natsu other | line agreement 89.5% | pass |

BBT and Kaze share the live-action profile: BBT needs `male_below` over 195 Hz (Leonard) and Kaze needs `female_above` under 198 Hz (真風), which leaves a 2 Hz window with no unknown band at all, and BBT's line agreement falls to 89.5% there. I don't treat that as a calibrated band. The anime profile's 300 Hz, tuned on the 20th percentile, puts 38% of Fairy Tail's main speech in the unknown band on the median.

## What changed and what waits

- `cards.select` and `cards.auto.gender_thresholds` default to `fukikae.pitch.MALE_BELOW` and `FEMALE_ABOVE` (165/200). The old `cards.select` 180 Hz put both BBT leads in the other class; 200 leaves them unknown.
- `gender_thresholds` warns once per run when a workspace (not a profile) sets `[tts.gender]`.
- `profiles/anime.toml` kept `female_above = 300` until the operator decided. On these numbers 165/200 is the better of the two on every set (one main character wrong against 38% unknown), but neither passes, and the plan stopped here. D44 settled it (next section).

## Ruling D44 (2026-11-12): one band rule on every profile

The operator ruled that every profile uses 165/200 Hz on the median F0, and the anime profile's `female_above = 300` override is gone. `profiles/anime.toml` no longer has a `[tts.gender]` table, so `cards.auto.gender_thresholds` returns `fukikae.pitch`'s `MALE_BELOW` and `FEMALE_ABOVE` on every shipped profile; a workspace can still set `[tts.gender]` and still gets the median warning.

The measurement above is why. No pair of bands met the plan's acceptance rule on all three tuning sets, so there was no calibrated band for the anime profile to keep in place of the default. Of the two candidates, 165/200 was the better one on every set: on Fairy Tail it puts one main character (Natsu, shouting) in the other class, while 165/300 leaves 38.0% of main speech in the unknown band, and on Kaze 300 Hz leaves 94.6% unknown. The 300 Hz value was tuned on the old 20th-percentile estimate and was never recalibrated for the median.

After Part A the override barely mattered anyway. Since D41 nothing in the pipeline chooses a voice by these bands: the tts stage's line gate (`GATE_BANDS`) and the reference registers (`select.speaker_register`) read `fukikae.pitch`'s constants directly, and only `eval tts`'s stock baselines read `[tts.gender]`. Keeping a second rule that only one diagnostic read would have meant two answers to "what register is this speaker" depending on where the question was asked. One rule is simpler to reason about, and it is the one rule the studio (Part B) has to show and let the operator override.

No cached decision depends on the override. The tts stage key's `gender_gate` holds `GATE_VERSION`, `GATE_BANDS`, `pitch.VERSION` and `LINE_MIN_VOICED_S`, all constants that D44 doesn't change; `STAGE_KEYS` keeps `[tts.gender]` out of the stage's options key; the register memo in `cards.auto` is keyed on the constant bands; the pitch cache holds F0 tracks, not classes; and `asrclean` reads no band. The stock baselines `eval tts` picks with `gender_thresholds` go into a scratch store on every run, and a render's key holds the voice it was made with, so a baseline that now picks a different stock voice renders under a new key. No VERSION bump was needed.
