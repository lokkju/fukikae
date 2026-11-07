# Identity profiles: sweeps, picks and held-out confirmation

_Run 2026-11-07 on ml-01 (RTX 3090, index 0), branch feat/speaker-identity. Sweeps used the tuning episodes only (no `--holdout`). Each holdout was scored once, after the profiles were committed (`df7692b`); nothing was re-tuned on it. Recurring means 5 s of labeled speech per episode (Ruling R6)._

## Sweeps

Embedder WeSpeaker throughout. Anime sets (`fairytail` ja, `fairytail-en` dub, E12 then E13, `--gt ass`): the shipped pyannote diarization and the `t0.6` exclude_overlap-off variant workspaces from Task 6, each with centroid and with joint (`episode_cannot_link` on), thresholds 0.50 to 0.75 in steps of 0.05. Live action (`grid`, E02 then E04, `--gt sdh --subtitle-track 2`): the shipped diarization with centroid, joint with cannot-link off (the candidate), and joint with it on for reference. Clean means passing and in no merge. Driver and raw lines: `~/fukikae-ws/segmatch/t9sweep.py`, `t9.jsonl`, `t9tab.py`.

Selection rule for anime: clean characters summed over both sets, then fewest merges summed, then mean fraction.

### fairytail (ja)

| Segmentation | Matcher | Thr | Frac | Pass/rec | Clean | Merges | Share | Cov |
|---|---|---|---|---|---|---|---|---|
| diarization | centroid | 0.50 | 0.44 | 4/9 | 4 | 1 | 72% | 36% |
| diarization | joint (cannot-link on) | 0.50 | 0.56 | 5/9 | 5 | 1 | 81% | 41% |
| diarization | centroid | 0.55 | 0.44 | 4/9 | 4 | 1 | 72% | 36% |
| diarization | joint (cannot-link on) | 0.55 | 0.56 | 5/9 | 5 | 1 | 81% | 41% |
| diarization | centroid | 0.60 | 0.44 | 4/9 | 4 | 1 | 72% | 36% |
| diarization | joint (cannot-link on) | 0.60 | 0.56 | 5/9 | 5 | 1 | 81% | 41% |
| diarization | centroid | 0.65 | 0.33 | 3/9 | 3 | 1 | 70% | 36% |
| diarization | joint (cannot-link on) | 0.65 | 0.44 | 4/9 | 4 | 1 | 80% | 41% |
| diarization | centroid | 0.70 | 0.33 | 3/9 | 3 | 1 | 70% | 36% |
| diarization | joint (cannot-link on) | 0.70 | 0.44 | 4/9 | 4 | 1 | 80% | 41% |
| diarization | centroid | 0.75 | 0.33 | 3/9 | 3 | 1 | 73% | 40% |
| diarization | joint (cannot-link on) | 0.75 | 0.44 | 4/9 | 4 | 1 | 77% | 41% |
| pyannote-t0.6-eo0 | centroid | 0.50 | 0.44 | 4/9 | 4 | 1 | 72% | 34% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.50 | 0.56 | 5/9 | 5 | 3 | 81% | 42% |
| pyannote-t0.6-eo0 | centroid | 0.55 | 0.44 | 4/9 | 4 | 1 | 72% | 34% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.55 | 0.56 | 5/9 | 5 | 3 | 81% | 42% |
| pyannote-t0.6-eo0 | centroid | 0.60 | 0.44 | 4/9 | 4 | 1 | 72% | 34% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.60 | 0.56 | 5/9 | 5 | 3 | 81% | 42% |
| pyannote-t0.6-eo0 | centroid | 0.65 | 0.44 | 4/9 | 4 | 1 | 72% | 35% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.65 | 0.56 | 5/9 | 5 | 3 | 81% | 42% |
| pyannote-t0.6-eo0 | centroid | 0.70 | 0.33 | 3/9 | 3 | 1 | 72% | 40% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.70 | 0.44 | 4/9 | 4 | 1 | 76% | 42% |
| pyannote-t0.6-eo0 | centroid | 0.75 | 0.33 | 3/9 | 3 | 1 | 70% | 40% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.75 | 0.44 | 4/9 | 4 | 1 | 74% | 41% |

### fairytail-en (English dub)

| Segmentation | Matcher | Thr | Frac | Pass/rec | Clean | Merges | Share | Cov |
|---|---|---|---|---|---|---|---|---|
| diarization | centroid | 0.50 | 0.70 | 7/10 | 6 | 1 | 72% | 43% |
| diarization | joint (cannot-link on) | 0.50 | 0.80 | 8/10 | 7 | 1 | 81% | 51% |
| diarization | centroid | 0.55 | 0.70 | 7/10 | 6 | 1 | 72% | 43% |
| diarization | joint (cannot-link on) | 0.55 | 0.80 | 8/10 | 7 | 1 | 81% | 51% |
| diarization | centroid | 0.60 | 0.70 | 7/10 | 6 | 1 | 72% | 43% |
| diarization | joint (cannot-link on) | 0.60 | 0.80 | 8/10 | 7 | 1 | 81% | 50% |
| diarization | centroid | 0.65 | 0.70 | 7/10 | 6 | 1 | 78% | 48% |
| diarization | joint (cannot-link on) | 0.65 | 0.80 | 8/10 | 7 | 1 | 81% | 50% |
| diarization | centroid | 0.70 | 0.60 | 6/10 | 6 | 0 | 74% | 48% |
| diarization | joint (cannot-link on) | 0.70 | 0.70 | 7/10 | 7 | 0 | 77% | 50% |
| diarization | centroid | 0.75 | 0.60 | 6/10 | 6 | 0 | 74% | 48% |
| diarization | joint (cannot-link on) | 0.75 | 0.70 | 7/10 | 7 | 0 | 77% | 50% |
| pyannote-t0.6-eo0 | centroid | 0.50 | 0.60 | 6/10 | 5 | 1 | 71% | 42% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.50 | 0.80 | 8/10 | 7 | 1 | 86% | 51% |
| pyannote-t0.6-eo0 | centroid | 0.55 | 0.60 | 6/10 | 5 | 1 | 71% | 42% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.55 | 0.80 | 8/10 | 7 | 1 | 86% | 50% |
| pyannote-t0.6-eo0 | centroid | 0.60 | 0.60 | 6/10 | 5 | 1 | 71% | 42% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.60 | 0.80 | 8/10 | 7 | 1 | 86% | 50% |
| pyannote-t0.6-eo0 | centroid | 0.65 | 0.70 | 7/10 | 6 | 1 | 84% | 48% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.65 | 0.80 | 8/10 | 7 | 1 | 86% | 50% |
| pyannote-t0.6-eo0 | centroid | 0.70 | 0.60 | 6/10 | 6 | 0 | 80% | 48% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.70 | 0.70 | 7/10 | 7 | 0 | 82% | 50% |
| pyannote-t0.6-eo0 | centroid | 0.75 | 0.60 | 6/10 | 6 | 0 | 80% | 48% |
| pyannote-t0.6-eo0 | joint (cannot-link on) | 0.75 | 0.70 | 7/10 | 7 | 0 | 82% | 50% |

### grid (Korean live action)

| Segmentation | Matcher | Thr | Frac | Pass/rec | Clean | Merges | Share | Cov |
|---|---|---|---|---|---|---|---|---|
| diarization | centroid | 0.50 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | joint (cannot-link off) | 0.50 | 1.00 | 2/2 | 2 | 0 | 81% | 14% |
| diarization | joint (cannot-link on) | 0.50 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | centroid | 0.55 | 0.50 | 1/2 | 1 | 0 | 73% | 12% |
| diarization | joint (cannot-link off) | 0.55 | 1.00 | 2/2 | 2 | 0 | 81% | 14% |
| diarization | joint (cannot-link on) | 0.55 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | centroid | 0.60 | 0.50 | 1/2 | 1 | 0 | 73% | 12% |
| diarization | joint (cannot-link off) | 0.60 | 1.00 | 2/2 | 2 | 0 | 81% | 14% |
| diarization | joint (cannot-link on) | 0.60 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | centroid | 0.65 | 0.50 | 1/2 | 1 | 0 | 73% | 12% |
| diarization | joint (cannot-link off) | 0.65 | 1.00 | 2/2 | 2 | 0 | 81% | 14% |
| diarization | joint (cannot-link on) | 0.65 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | centroid | 0.70 | 0.50 | 1/2 | 1 | 0 | 73% | 12% |
| diarization | joint (cannot-link off) | 0.70 | 1.00 | 2/2 | 2 | 0 | 78% | 14% |
| diarization | joint (cannot-link on) | 0.70 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |
| diarization | centroid | 0.75 | 0.50 | 1/2 | 1 | 0 | 73% | 12% |
| diarization | joint (cannot-link off) | 0.75 | 1.00 | 2/2 | 2 | 0 | 78% | 14% |
| diarization | joint (cannot-link on) | 0.75 | 0.50 | 1/2 | 1 | 0 | 73% | 13% |

## Picks

**Anime: joint, episode cannot-link on, shipped pyannote diarization, threshold 0.6.** It sums to 12 clean characters over the two sets (ja 5, dub 7) with 2 merges and a mean fraction of 0.68. The same totals hold at 0.50, 0.55 and 0.60. At 0.65 ja drops to 4 clean (11 total, still 2 merges); at 0.70 and 0.75 ja stays at 4 and the dub's CANA+JUVIA merge disappears (11 clean, 1 merge, mean fraction 0.57). By the rule, 0.50 to 0.60 tie and 0.6 is the top of that tie, the nearest to the 0.65 drop on ja. Centroid sums to 10 clean with 2 merges. The exclude_overlap-off variant ties on clean characters (12) but adds two merges on ja (TOUKA, LUCY and JUVIA on one id, the Task 6 finding), so the shipped diarization wins on the second rule. The dub's 0.86 share on the variant does not change the pick, since the rule does not use share.

**Live action: joint, cannot-link off, shipped diarization, threshold 0.6.** 2 of 2 recurring characters pass with no merge at every threshold from 0.50 to 0.75 (0.60 is a middle value; speech share falls from 81% to 78% at 0.70 and 0.75); centroid and joint with cannot-link on both stay at 1 of 2. This rests on two recurring characters at 12 to 14% coverage, one character is the whole difference, and the pick is provisional (Ruling R16).

**Default: centroid at 0.6.** Summed over all three tuning sets, centroid gives 11 clean characters with 2 merges at 0.50, 0.55 and 0.60 (4 + 6 + 1), and 10 at 0.65. 0.6 is the top of that plateau and today's value.

**Language variants: none.** Both anime tracks plateau at the same thresholds (0.50 to 0.60), so no per-language override would move majority fraction by 5 points on its own set.

## The profiles

All three name `backend` in every table (Ruling R3; `[characters]` names `matcher`): `[separation] demucs` (Ruling R15), `[diarize] pyannote`, `[embed] wespeaker`, `[characters] matcher = "centroid"`, `[registry] match_threshold = 0.6`, `min_new_seconds = 20`. ASR is left to the stages' fallbacks. The characters stage cannot run joint (it needs every episode at once), so the stage keeps centroid and joint is `fukikae registry resolve`, which reads `[resolve]`. The live-action profile's header says PROVISIONAL. The batch settings live in each profile's `[resolve]` table (anime: joint, 0.6, `episode_cannot_link = true`; live-action: joint, 0.6, `episode_cannot_link = false`), which `registry resolve` reads through the normal layering; flags on the command override it (Ruling R17). `[resolve]` selects its matcher by `matcher`, as `[characters]` does, so it is exempt from the `backend` rule.

Changing the default embedder from ECAPA to WeSpeaker changes the diarize stage's config key, so a workspace without a `profile` redoes diarization on its next run.

## Held-out confirmation

Scratch workspaces `~/fukikae-ws/t9hold/<set>` carry the profile, re-ran extract (anime only) and diarize with the profile's embedder, then `registry resolve` on the tuning pair, `registry resolve` on the held-out episode alone (the tuning actors are fixed anchors, Ruling R13), and `eval identity --from-assignments --tune ... --holdout ...`. Outputs: `~/fukikae-ws/segmatch/t9/<set>-holdout.txt`. Because pyannote and the cluster embeddings were re-run through the pipeline, the tuning rows below can differ slightly from the sweep (they matched on clean counts).

| Profile | Set | Tune (frac, merges) | Held-out episode | Recurring | Passing | Fraction | Merges | Share | Strict | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anime | fairytail (ja) | 0.56, JUVIA+LUCY | E14 | 11 | 6 | 0.55 | CHARLE+HAPPY+WENDY; CANA+LUCY; KIRIA+NATSU | 70% | 0.21 | 50% | fail |
| anime | fairytail-en | 0.80, CANA+JUVIA | E14 | 10 | 6 | 0.60 | HAPPY+LUCY+WENDY | 66% | 0.29 | 55% | fail |
| live-action (provisional) | grid | 1.00, none | E05 | 5 | 4 | 0.80 | 새하+어진 | 80% | 0.67 | 16% | fail |
| default | fairytail (ja) | 0.44, JUVIA+LUCY | E14 | 11 | 6 | 0.55 | CHARLE+HAPPY+WENDY; CANA+LUCY; KIRIA+NATSU | 70% | 0.21 | 47% | fail |
| default | fairytail-en | 0.70, CANA+JUVIA | E14 | 10 | 6 | 0.60 | HAPPY+LUCY+WENDY | 66% | 0.29 | 55% | fail |

Neither anime profile reaches the bar on E14. In both tracks id 2, which the tuning pair built for HAPPY, absorbs WENDY (106 s in E14 on ja, 138 s on the dub, against 7 to 24 s in E12 and E13) and, with her, CHARLE on ja and LUCY on the dub; id 2 is only 56% (ja) and 74% (dub) WENDY. The dub also fails LUCY, WHITE MAGE, CHARLE and NEBARU on E14; ja fails LUCY, WHITE MAGE, KIRIA and JUVIA. One hypothesis is that a speech-heavy character the tuning pair holds only as a few seconds gets pulled onto the nearest well-populated id; no ablation was run, so this is not established. The grid result is not a gate: 5 recurring at the holdout's definition against 2 in tuning, and the 새하+어진 merge (id 2, 48% 어진) is a pair the tuning pair never exercised. All three ship as measured, with the failures stated here and no re-tuning on E14 or E05.

The default profile (WeSpeaker, centroid at 0.6, the per-episode characters stage) was held out on the same two E14 tracks the anime profile used. Copies of the anime holdout workspaces at `~/fukikae-ws/t9hold-default/<set>` dropped `profile = "anime"` (so the default applies) and their registries and characters outputs; the diarize outputs were reused, since the two profiles resolve the same diarize and embed settings, so no stage but characters re-ran and nothing was embedded. `fukikae run` ran the characters stage on E12, E13 and then E14, and `eval identity --from-assignments --tune S01E12,S01E13 --holdout S01E14 --gt ass` scored it once. Outputs: `~/fukikae-ws/segmatch/t9/<set>-default-holdout.txt`. The tuning rows match the sweep's centroid rows at 0.6 (ja 0.44 with JUVIA+LUCY, dub 0.70 with CANA+JUVIA). It fails on both tracks with the same fraction, merges, share and strict score as the anime profile's joint resolve (ja 0.55, dub 0.60), id 2 again holding WENDY with HAPPY and CHARLE (ja) or HAPPY and LUCY (dub): on E14 the per-episode centroid stage and the batch joint resolve end in the same place.

The anime evidence is a single show (Fairy Tail 100 Years Quest, two audio tracks of the same episodes) with 9 to 10 recurring characters, so the two anime holdouts are not independent.

## What remains

- Live-action ground truth (D9): Grid's CC track tags speakers on 12 to 16% of speech, which leaves the profile resting on a handful of characters.
- The anime female voices (TOUKA, LUCY, JUVIA, CANA on ja; WENDY, HAPPY, CHARLE on E14) were not separated by any of the embedders and matchers tested in these tasks, on one show; LDA adaptation (Tasks 6 and 7) and Task 8's dialogue stem did not close it.
- Face and active-speaker detection as the next lever (`docs/ideas.md`).
