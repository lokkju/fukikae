# Per-turn F0 spread guard: rejected (identification workflow, Task 15 follow-up)

_Run 2026-11-13 on CPU, branch feat/pitch-spread-guard off the Part B head ba0c47f, estimator `fukikae.pitch` VERSION 4. Workspaces (scratch copies only): Sol Bianca S00E01 at `/nvme/fukikae/tmp/rev16/ws`; Fairy Tail E11-E14 at `/nvme/fukikae/tmp/rev16/ft` for the season index; Fairy Tail E12 for `eval gender` at a full copy of `/nvme/fukikae/tmp/zonos3/ws/ft` (`/nvme/fukikae/tmp/pitch-guard/ft-e12`), because `rev16/ft` has no translate or tts output and the Task 11 arm (`ft-t11`) no longer exists. Every pitch cache was under `/nvme/fukikae/tmp/pitch-guard/`, never a workspace's own. Script: `scripts/pitch-guard/measure.py`, which carries the guard itself (wrapping `pitch.turn_classes`) so it runs on the unchanged pipeline: `measure.py turns <ws> <cache> <out.jsonl> --guard 20 [--pyin 16]` for the season suggestions and per-turn spreads, `measure.py gate <ws> <episode> <cache> <out.json> --guard 20` for the line gate; both reproduce the patched code's numbers below exactly._

**Verdict: the guard regresses Part A and is not committed.** The controller's ruling assumed real turns stay well under about 20 semitones from p10 to p90. They don't: on Sol Bianca 113 of the 653 turns with 0.4 s or more voiced (250 s) read over 20 st, and pYIN agrees with our class on 61 of the 99 classified ones (146 s). At 20 st the guard drops SPEAKER_14's false split, as ruled, but it also strips 17 of Sol Bianca's 25 gated lines (A3) and 5 of Fairy Tail E12's 16, takes Sol Bianca SPEAKER_13 (the cluster the spec says is bimodal) from 25.9 s low to 6.0 s, and leaves three Fairy Tail extras with no register at all. `eval gender`'s headline numbers don't move, because it can't see the change; the line gate the next tts run would apply can, and there the guard reopens the wrong-register lines Part A closed. No threshold fixes it (the sweep below). The implementation and its tests, written test first, are in `/nvme/fukikae/tmp/pitch-guard/rejected-guard.patch`.

## The guard as built

A turn (`pitch.turn_classes`, so the season index's split suggestions, the tts line gate and speaker registers, and the reference gates' majority register) whose voiced frames spread over `TURN_SPREAD_MAX_ST` (20) semitones, `12 log2(p90 / p10)`, is classed `u` instead of by its median; `classify` takes the limit as an optional argument and a pooled speaker or season summary takes none; the names stage's split pieces use it too, with `pitch.VERSION` in that stage's key while a split file exists; `pitch.VERSION` 4 to 5. The other keys that name `pitch.VERSION` were already there: the tts line gate, asrclean, the season index's digest and `eval gender`'s options; the pitch cache key covers the reference registers (`cards.auto.references_class`) and the turn tracks. The failing tests first (a 26 st synthetic turn classes `u`, an 8 st one keeps `f`, `turn_classes` on a 90-then-400 Hz turn gives `u` and on a 220-then-350 Hz turn `f`, the names key follows `pitch.VERSION` only with a split), then the code; the full suite passed with it (see the ledger line).

## Spread of real turns

Every turn of every speaking character in the season index's reading (one track per character per episode), classified at 165/200 Hz with 0.4 s voiced; pYIN (librosa, 60-800 Hz, 2048-sample frames, as Task 15) on every turn of 16 st or more:

| | Sol Bianca | Fairy Tail E11-E14 |
|---|---|---|
| turns, of them 0.4 s or more voiced | 1142, 653 | 1400, 831 |
| spread p50 / p90 / p95 / p99 (st) | 11.4 / 25.2 / 29.2 / 34.8 | 10.4 / 22.4 / 28.0 / 33.7 |
| over 20 st and classified m or f | 99 turns, 207.0 s | 101 turns, 275.6 s |
| of those, pYIN finds no voice (under 0.4 s voiced) | 34 (47.0 s) | 38 (42.6 s) |
| of those, pYIN gives the same class | 61 (145.5 s) | 60 (225.3 s) |
| of those, pYIN gives the other class or u | 4 (14.5 s) | 3 (7.6 s) |

By threshold, of the classified turns over it (pYIN no voice : pYIN same class, in turns):

| guard | Sol Bianca | Fairy Tail |
|---|---|---|
| 16 st | 41 : 123 | 52 : 123 |
| 20 st | 34 : 61 | 38 : 60 |
| 24 st | 27 : 36 | 26 : 28 |
| 28 st | 19 : 20 | 18 : 14 |
| 30 st | 11 : 14 | 13 : 7 |

At every threshold at least as many turns the guard removes are real speech of the class we gave them as are pYIN-silent ones, so p90/p10 doesn't separate the two. The wide spreads on real turns come from our estimator: a few frames an octave or two up (SPEAKER_07, char 8, reads 67-74-507 Hz on a turn pYIN reads 67-72-77 Hz) widen p90 without moving the median, which the median-based class is built to tolerate.

## Part A measures, before and after (guard at 20 st)

`fukikae eval gender` (bands 165/200 Hz, lines under 0.4 s voiced u, `--pitch-cache` in scratch, `--no-ledger`), before at ba0c47f and after with the rejected patch applied:

| measure | Sol Bianca before | after | Fairy Tail E12 before | after |
|---|---|---|---|---|
| strict mismatches | 25 of 369 | 25 of 369 | 10 of 261 | 10 of 261 |
| single 180 Hz cut | 44 of 414 | 44 of 414 | 28 of 313 | 28 of 313 |
| references passing the pipeline's gates | 27 of 27 | 27 of 27 | 51 of 51 | 51 of 51 |
| references within 4 st of the majority (diagnostic) | 22 of 27 | 21 of 27 | 34 of 51 | 32 of 51 |
| gated lines (recorded in plan.json) | 25 | 25 | 16 | 16 |
| emotion audio against a wrong-gender line | 0 | 0 | 0 | 0 |
| speakers with a register (majority m or f) | all | all | all | 3 extras lose theirs (below) |

Sol Bianca's before reproduces the Task 11 numbers (25 of 369, 27 of 27, 0). Fairy Tail E12's before is not Task 11's 13 of 274: these are the zonos3 spike's renders of E12, scored before and after on the same renders. The headline rows can't move: `eval gender` bands each source line on its median itself (`eval.gender.band`), reports the gated lines tts recorded, and counts wrong-gender prompts against lines the guard has just made `u`. What moves is the line gate (A3) the next tts run computes, measured with `measure.py gate` (`stages.tts.gender_gate` both ways, operator genders from a registry copy):

| | Sol Bianca | Fairy Tail E12 |
|---|---|---|
| lines the gate would hold, no guard | 25 | 16 |
| with the guard at 20 st | 8 | 11 |
| line classes changed | 85 of 552 | 49 of 354 |
| speaker registers changed | none | SPEAKER_04~makarov m to u, SPEAKER_06~nabaru f to u, SPEAKER_06~wizard-a f to u |

pYIN on the cue spans of the lines the guard would release: on Sol Bianca, 9 of the 17 have a pYIN median in the same band with 0.4 s or more voiced (c0049 at 282 Hz in SPEAKER_08's male register; c0157, c0291, c0329, c0365, c0389, c0422, c0444 and c0472 at 121-164 Hz in chars 9, 7, 10 and 6's female registers), 6 have pYIN medians in the male band on less voicing (c0135 148, c0400 136, c0450 103, c0474 109, c0541 69 Hz, and c0475 166 Hz just inside the gap), and 2 have none (c0347, c0449). These are char 10's male butler and lecturer and char 7's low lines, the ones Task 11 lists among the worst cues and Part B's split is meant for. Released, they would pass their male source as emotion audio into a female card again, and the wrong-gender prompts metric would not count them, since their line class is now `u`. On Fairy Tail E12, 4 of the 5 released lines have pYIN in our band (c0143, c0190, c0225 f against male registers; c0317 m against a female one); c0294 reads 458 Hz by pYIN on 0.23 s voiced.

## Season index bimodal suggestions

| | no guard | guard 20 st | guard 24 st | guard 26 st |
|---|---|---|---|---|
| SB char 7 SPEAKER_14 | m 11.0, u 27.2, f 43.9 | gone | gone | gone |
| SB char 10 SPEAKER_13 | m 25.9, u 35.7, f 61.8 | m 6.0, u 99.8, f 17.5 | m 6.0, u 91.7, f 25.6 | gone |
| FT E12 char 1 SPEAKER_06 | m 17.9, u 23.2, f 38.3 | m 11.8, u 46.5, f 21.1 | m 14.5, u 31.2, f 33.6 | as 24 |
| FT E12 char 8 SPEAKER_09 | m 46.3, u 46.4, f 8.9 | unchanged | unchanged | unchanged |
| FT E12 char 10 SPEAKER_05 | m 20.4, u 3.6, f 7.9 | unchanged | unchanged | unchanged |
| FT E12 char 11 SPEAKER_07 | m 27.1, u 18.3, f 12.3 | m 27.1, u 24.9, f 5.8 | m 27.1, u 22.0, f 8.7 | m 27.1, u 18.9, f 11.8 |
| FT E13 char 12 SPEAKER_08 | m 16.1, u 23.0, f 31.4 | m 16.1, u 27.8, f 26.6 | m 16.1, u 24.5, f 29.9 | as 24 |
| FT E14 char 1 SPEAKER_01 | m 25.4, u 21.9, f 21.8 | m 19.8, u 40.1, f 9.3 | m 23.5, u 32.2, f 13.4 | as 24 |

The no-guard column reproduces the season.json both copies hold (two on Sol Bianca, Fairy Tail's six). SPEAKER_14 drops as ruled, but the three turns Task 15 named (1155.1, 2836.3, 2838.7 s, pYIN silent) are three of the twelve (15.8 s) the guard takes from it; of the other nine, one more is pYIN silent (782.0 s) and five have a pYIN median in the same band on 0.4 s or more voiced (641.8, 1118.2, 1180.9 and 1449.4 s at 211-277 Hz, 2092.4 s at 127 Hz). On SPEAKER_13 the guard takes 21 turns (64.2 s), among them 9 of its low turns (19.9 s), 6 of which pYIN reads at 105-156 Hz on 0.4 s or more voiced. Fairy Tail keeps six suggestions at every guard tried, but its true two-voice clusters lose seconds from both parts.

## Threshold

Sol Bianca's three pYIN-silent turns read 26.3, 28.2 and 29.0 st, so removing SPEAKER_14's split needs a guard under 26.3 st; SPEAKER_13 stops being bimodal at 26 st and is down to 6.0 s low at 24 st and 20 st. No value keeps SPEAKER_13's low voice while dropping SPEAKER_14, and every value up to 30 st releases gated lines that pYIN confirms (11 of 25 still gated at 24 st, 12 at 26 st on Sol Bianca). 20 st, the ruling's starting point, is the value measured above.

What the measurement points to instead, for the controller: the turns the guard was meant for are the ones where pYIN finds no voice, and those are separable by the voiced fraction of our frames inside the p10-p90 band, or by a frame-to-frame continuity check, rather than by the spread of the percentiles. That is a different estimator change and needs its own ruling.
