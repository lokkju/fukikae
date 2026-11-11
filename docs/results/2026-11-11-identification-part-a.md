# Identification workflow Part A: Sol Bianca S00E01 and the Fairy Tail names arm (Task 11)

_Run 2026-11-11 on the GPU host (RTX 3090, index 0 under the GPU lock), branch feat/identification-part-a at 72021ee. Sol Bianca's "after" is a fresh copy of `~/fukikae-scratch/sol-bianca` at `~/fukikae-ws/t-ident/sb-final` (extract, separate, language, ASR and diarize hard-linked; registry, cards, caches and every later stage copied), run with `fukikae run <copy> S00E01`: asrclean, translate, tts, timefit and mix reran (characters and names were current). Fairy Tail is a fresh copy of the D27 names arm (`~/fukikae-ws/t-ident/ft-names-a1`, built with `scripts/names-eval/make_arm.sh`) at `ft-t11`, run with `fukikae run <copy> S01E12 --to tts`: asrclean, tag, translate and tts reran (names and characters were current). Scores are `fukikae eval gender` (A0 estimator, bands 165/200 Hz, lines under 0.4 s voiced are u) and `fukikae eval names`. Ledger rows (gender and names, tags `t11-*`) are in `2026-11-11-identification-part-a.jsonl`. Nothing was tuned on Sol Bianca or E14._

**Verdict on the spec's Sol Bianca targets:** stock-voiced cues 51 to 0, extras disagreement 13 of 22 to 3 of 22, wrong-gender emotion prompts 23 to 0 and char 11 gone all pass. Strict mismatches stay at 25 against a target of 15 or fewer, a miss, and reference purity on chars 1, 2 and 10 is not 100% by the evaluation's 4-semitone measure while the pipeline flags none of them, a second miss. On Fairy Tail, cue accuracy and the names metric's reference purity don't move; gender mismatches on E12 rise from 10 to 13 (a third miss); E14's purity is 63.9% against D27's 73.9%, the drop the ledger already recorded at Task 3.

## Sol Bianca: the measures

The "before" is the original v1 dub, scored by the same `eval gender` (so the numbers differ from the diagnostic's pYIN figures: 25 of 348 strict here against its 35 of 402).

| measure | before (v1) | after (v2) | target |
|---|---|---|---|
| cues | 570 | 552 | |
| strict mismatches | 25 of 348 (7.2%) | 25 of 369 (6.8%) | 15 or fewer: **missed** |
| single 180 Hz cut | 53 of 414 (12.8%) | 44 of 414 (10.6%) | |
| extras: render against source | 13 of 22 | 3 of 22 | 3 or fewer: met |
| stock voice against source | 14 of 23 | none | |
| stock-voiced cues | 51 | 0 | 0: met |
| gated lines (A3) | 0 | 25 | |
| emotion audio against a wrong-gender line | 23 | 0 | 0: met |
| cues in song spans | 16 | 0 | |
| asrclean drops | - | 3 segments (18 cues) | |
| character 11 | 12 cues | gone | gone: met |
| references passing purity | 17 of 27 | 22 of 27 | 100% on chars 1, 2, 3, 10 or flagged: **missed** |
| render-vs-source cosine, clones | 0.33 (519 cues) | 0.34 (552 cues) | |
| render-vs-source cosine, stock / former stock | 0.07 (51 cues) | 0.44 (39 cues) | reported, no target |

The 18 dropped cues are asrclean's three segments: お疲れ様でした over the ending's first song (3564-3685 s) and two ご視聴ありがとうございました segments over the second (3775-3819 s). They took char 11's 12 cues and three each from chars 2 and 3.

### Why strict mismatches didn't fall

20 of the 25 after-mismatches are gated lines: a source line in the other band from its character, rendered in the character's own clone without emotion audio, as the spec decided (A3, D41). Before, those lines passed their wrong-gender source as `emotion_audio`, which pulled the render toward the source's register and often read as a match. The spec predicted this ("this routing will fix fewer of the counted mismatches"). Without the gated lines the after count is 5: char 8's four renders that come out 259-377 Hz from a male clone (c0056, c0103, c0214 and c0314; the first three were as high in v1, and c0314 read 84 Hz there) and one SPEAKER_10 line (c0057, 159 against 202 Hz).

Gated lines by speaker: char 10 8, char 7 6, char 6 5, char 2 2, chars 1 and 9 and extras SPEAKER_08 and SPEAKER_10 one each.

### Per speaker

| who | cues before | strict before | 180 Hz before | cosine before | cues after | strict after | 180 Hz after | cosine after |
|---|---|---|---|---|---|---|---|---|
| character 5 | 126 | 0/108 | 0/108 | 0.29 | 126 | 0/108 | 0/108 | 0.29 |
| character 1 | 82 | 0/38 | 14/67 | 0.32 | 82 | 0/44 | 10/68 | 0.32 |
| character 2 | 66 | 2/21 | 3/25 | 0.28 | 63 | 1/26 | 1/28 | 0.28 |
| character 7 | 45 | 0/24 | 4/34 | 0.34 | 45 | 5/28 | 8/33 | 0.33 |
| character 6 | 40 | 4/25 | 7/29 | 0.29 | 40 | 5/26 | 7/29 | 0.28 |
| character 10 | 36 | 2/18 | 5/32 | 0.43 | 36 | 6/22 | 8/31 | 0.41 |
| character 4 | 35 | 0/29 | 0/29 | 0.47 | 35 | 0/30 | 0/30 | 0.46 |
| character 3 | 33 | 0/21 | 0/21 | 0.30 | 30 | 0/20 | 0/20 | 0.29 |
| character 8 | 33 | 3/24 | 3/24 | 0.49 | 33 | 4/25 | 4/25 | 0.48 |
| character 9 | 23 | 1/17 | 2/18 | 0.33 | 23 | 1/18 | 2/19 | 0.35 |
| character 11 | 12 | 0/1 | 0/3 | 0.10 | 0 | - | - | - |
| extra unknown | 13 | 0/0 | 0/0 | 0.12 | 13 | 0/0 | 0/0 | 0.39 |
| extra SPEAKER_08 | 6 | 5/6 | 5/6 | 0.00 | 6 | 1/6 | 1/6 | 0.47 |
| extra SPEAKER_10 | 6 | 1/5 | 2/6 | 0.05 | 6 | 2/5 | 3/6 | 0.46 |
| extra SPEAKER_11 | 5 | 0/4 | 0/4 | 0.05 | 5 | 0/4 | 0/4 | 0.50 |
| extra SPEAKER_01 | 4 | 3/3 | 4/4 | 0.08 | 4 | 0/3 | 0/3 | 0.47 |
| extra SPEAKER_09 | 3 | 3/3 | 3/3 | -0.01 | 3 | 0/3 | 0/3 | 0.41 |
| extra SPEAKER_00 | 2 | 1/1 | 1/1 | 0.03 | 2 | 0/1 | 0/1 | 0.40 |

Render medians moved where the cards changed: char 1 212 to 256 Hz (source 236), char 2 279 to 368 (source 349), char 7 237 to 266 (source 234), char 10 193 to 238 (source 203). The former stock extras now sit on their own register: SPEAKER_08 234 to 117 Hz (source 116), SPEAKER_09 146 to 306 (294), SPEAKER_01 229 to 136 (142), SPEAKER_11 146 to 87 (96). The clones' cosine against their own source lines didn't move for the re-picked characters (char 2 0.28, char 10 0.43 to 0.41): the emotion prompt dominated that measure before and still does.

The former stock cosine of 0.44 is inflated for the 13 cluster-less lines: each is cloned from its own source span (A2's per-span rule), so the render is compared against the clip it was cloned from. The six extras clusters alone read 0.40 to 0.50.

### References, tiers and card changes

The tts stage re-picked four cards for failing the reference gates (A1), all off-register; char 3's card stayed (its references pass every gate the pipeline has):

| char | before: reference medians (Hz) | after: reference medians (Hz) | re-pick note |
|---|---|---|---|
| 1 | 179, 171, 170 (opening narration) | 236, 231, 216 (scene speech) | 6 off-register, 1 song-like refused; 13.3 s of 72.3 s eligible |
| 2 | 192, 198, 421 | 421, 289, 393 | 3 off-register refused; 11.1 s of 11.1 s eligible |
| 7 | 180, 198, 293 | 198, 293, 226 | 2 off-register refused; needs a split |
| 10 | 133, 209, 148 | 209, 318, 194 | 4 off-register refused; needs a split |

Purity by the evaluation's measure (within 4 semitones of the speaker's majority median, voiced at least 50%, outside song-like audio):

| char | majority (Hz) | passing before | passing after | failing after | pipeline flag after |
|---|---|---|---|---|---|
| 1 | 264 | 0/3 | 2/3 | 236 Hz clip read as song-like | none |
| 2 | 408 | 1/3 | 2/3 | 289 Hz (6 st low) | none |
| 3 | 420 | 2/2 | 2/2 | | none |
| 7 | 274 | 1/3 | 2/3 | 198 Hz (5.6 st low) | none |
| 8 | 92 | 2/3 | 2/3 | 116 Hz (4 st high) | none |
| 10 | 233 | 1/3 | 2/3 | 318 Hz (5.4 st high) | none |

The gap is a definition gap, not a pipeline fault: the ruling at Task 3 made the register gate refuse cross-class candidates only (the other band, or between bands at full confidence), while the evaluation measures the spec's 4-semitone distance. A 289 Hz clip for a 408 Hz speaker is the same class and passes the pipeline. Char 1's failing clip is the reverse: the pipeline's song gate refused one song-like candidate and kept this one, which the evaluation's song check calls song-like. Both go to the ledger as a ruling for the controller (tighten the gate to the spec's distance, or report purity by the gate's own rule).

Tiers after: all ten characters and SPEAKER_09 and SPEAKER_11 at the floor; SPEAKER_08 (4.7 s), SPEAKER_10 (7.2 s) and SPEAKER_00 (3.5 s) short; SPEAKER_01 thin (9.1 s from 10 turns); the 13 cluster-less lines thin, each from its own 0.3-1.3 s span. Seven of those spans are unvoiced or music by the voicing check (c0028, c0146, c0221, c0222, c0260, c0288, c0420, c0460 flag as impure), so their clones may be nobody's voice.

### Worst cues after

| cue | time | who | src Hz | render Hz | gated | text |
|---|---|---|---|---|---|---|
| c0347 | 38:41.2 | character 2 | 135 | 580 | yes | Foolish ones, I curse you |
| c0056 | 8:24.7 | character 8 | 90 | 377 | no | Then what if he said 'Tres'? |
| c0541 | 57:12.0 | character 7 | 77 | 286 | yes | It's memory, far away |
| c0103 | 11:42.2 | character 8 | 115 | 368 | no | Well done |
| c0314 | 36:20.2 | character 8 | 86 | 259 | no | Nothing at all |
| c0474 | 50:13.6 | character 10 | 98 | 248 | yes | They live together there. |
| c0214 | 24:22.0 | character 8 | 140 | 348 | no | So that ship |
| c0412 | 43:28.8 | character 6 | 135 | 324 | yes | How many times have you worried me? |
| c0049 | 8:00.8 | extra SPEAKER_08 | 253 | 109 | yes | That's a fine gun, isn't it? |
| c0295 | 35:10.4 | character 7 | 134 | 302 | yes | 哈，切恩萨姆斯特 |
| c0285 | 34:35.6 | character 6 | 154 | 320 | yes | 嗯，是吗？ |
| c0291 | 34:52.4 | character 7 | 130 | 266 | yes | 打开后面的隔间 |
| c0329 | 37:06.5 | character 10 | 130 | 244 | yes | It's impossible, Miss. You've been locked in. |
| c0539 | 57:05.5 | character 7 | 156 | 283 | yes | Major, that light |
| c0365 | 40:07.8 | character 10 | 154 | 264 | yes | His despair was that deep. |

Char 10's gated lines are the male butler and lecturer inside her cluster (「無理ですお嬢様」, the Dante lecture). v1's mixed card made him male; v2's female card makes him female, without emotion audio. Part B's split is the fix.

## Fairy Tail: regression check on the names arm

Only E12 reran. Its names and characters stages were current, so its assignments are byte-identical and cue accuracy cannot move; the renders it replaced were made before the arm's resolve (the t12 cards and character ids), so the E12 gender before-and-after includes D27's identity change as well as Part A.

`eval names` (protocol A; purity is the metric's own recomputation of each character's picks with the current picker):

| split | cue accuracy before | after | purity before | after | D27 names arm (Nov 10) |
|---|---|---|---|---|---|
| tune E11-E13 | 97.1% (941 cues) | 97.1% | 99.8% (267.3 pure, 0.5 foreign s) | 99.8% | 97.1%, 99.8% (122.1, 0.3) |
| holdout E14 | 32.7% (321 cues) | 32.7% | 63.9% (55.0, 31.1) | 63.9% | 32.7%, 73.9% (25.7, 9.1) |
| E11 alone | 96.5% | 96.5% | 100.0% | 100.0% | |
| E12 alone | 97.4% | 97.4% | 99.8% | 99.8% | |
| E13 alone | 97.6% | 97.6% | 99.9% | 99.9% | |

Because that purity is a recomputation, before and after are the same by construction; the comparison that shows Part A is the D27 column. Tune purity holds at 99.8%, over about twice D27's reference seconds. E14's purity falls from 73.9% to 63.9%: this is the merged-cluster drop the ledger recorded at Task 3 (0.685 to 0.611, then 0.639), where the register gate reads the majority of a cluster that holds another speaker. It is a miss against "must not drop on any episode"; E14 is held out and was not tuned on.

The purity of the cards the characters actually hold (every card reference's span against the aligned ASS names, same episodes): tune 71.7% (49.9 pure, 19.7 foreign s) before, 99.8% (190.3, 0.3) after; E14 62.2% both times, since no E14-only card changed.

E12 cue gender (`eval gender`, the before being the arm's existing E12 renders):

| measure | before | after |
|---|---|---|
| strict mismatches | 10 of 264 (3.8%) | 13 of 274 (4.7%): **rise** |
| single 180 Hz cut | 26 of 321 (8.1%) | 32 of 319 (10.0%) |
| extras disagreement | 5 of 5 | 1 of 10 |
| stock-voiced cues | 5 | 0 |
| emotion audio against a wrong-gender line | 20 | 0 |
| gated lines | 0 | 17 |
| references passing purity | 19 of 32 | 34 of 55 |

7 of the 13 after-mismatches are gated lines, and most of those are shouts the median reads in the other band, rendered correctly in the character's own voice: Laxus's "Roaring Thunder!" (383 Hz), Gray's "I do not!" (432 Hz), Wraith at 209 and 322 Hz, Natsu at 164 Hz. The metric counts a correct render of a shouted line as a mismatch, which is Natsu's pitch limit from the calibration (D44). The other six are Nebaru (character 1) five times, the same five as before on an unchanged card (a near-even bimodal cluster, no cross-class gate by the Task 4 ruling), and one Jellal line on his new card.

E12's card changes, each with its reason (the registry is per show, so the same cards serve E13):

| character | card before | card after | reason |
|---|---|---|---|
| Lucy (4) | 85c90823 | e3cfa201 | references not name-confirmed (D29 #4): S01E13 spans named juvia and cana |
| Cana (5) | 153a6eb4 | 2a3886d4 | references not name-confirmed: a span named juvia |
| Wraith (8) | d3c97d61 | 40ac5dc4 | below the clone floor (2 clips, 6.6 s; D32) |
| Gray (11) | f4a94639 | 169c078c | not name-confirmed (young goldmine), below the floor (7.2 s), and a reference the gates refuse (A1: 531.12-535.02 off-register, 184 Hz) |
| Erza (12) | 5488f645 | 6150494c | references not name-confirmed: a span named kiria |
| Kiria, Juvia, Young Makarov, Wendy, Jellal, Makarov (Wraith), Happy, Charle, Touka, Young Yajima, Young Porlyusica, Young Goldmine, Young Rob | none | first cards | the arm's resolve left these characters without a card (D27); Wendy, Young Yajima, Young Porlyusica and Young Goldmine are short and flagged, Young Rob thin |
| extras (Young Bob, Wizard A and B, Makarov, Nabaru) | stock | per-cluster clones | A2 (D41); four thin, one short |

Nebaru, Natsu, Laxus, character 10, Zeref and White Mage kept their cards (the last two are flagged below the floor but don't speak in E12). No card changed for a reason the spec doesn't name.

E13 was not rerun: before, it has 83 stock-voiced cues and 57 of 214 strict mismatches (26.6%), 55 of them on one extras cluster (SPEAKER_03, a female speaker on a male stock voice); that is the episode where A2 should matter most on Fairy Tail, and it is left for a later render.

## Limits

- **Char 3's insert song (#37).** Her first reference is still the 9.8 s English insert song from mid-episode. The song detector covers the head and the ending only, and the sung-ness gate doesn't catch this clip, so neither the pipeline nor the evaluation flags it.
- **Identity.** Chars 7 and 10 need a split (the stage says so); chars 2 and 3 are probably one speaker split in two (centroid cosine 0.76); 13 lines have no cluster. Part A routes what it can and flags the rest; the studio (Part B) is where the operator splits, merges and names.
- **Translation in Mandarin.** The translator returned Chinese for a block of about 30 cues (c0283-c0312, 34:25-36:10) in both dubs (21 cues in v1, 30 in v2), and IndexTTS2 speaks it. That is a translate-stage fault outside this plan.
- **One bug found and fixed on the way.** With D41 the profiles set no `[tts] backend`, and `pause_caps` and `tts_engines` resolved the stage without one, so tts on any such workspace stopped after prepare with "no backend configured for 'tts'". Fixed in 72021ee (`stages.tts.stage_options`). `fukikae backends setup`'s option lookup (`cli._worker_setup_options`) has the same blind spot and returns no options for the clone backend; not fixed here.
- **Character 11 stays in Sol Bianca's registry** with its retired stock card and no cues; `voices list` still says the next tts run re-picks it, which no run will do while it has no cues.
