# Subtitle speaker names on Fairy Tail E11 to E14: results

_Run 2026-11-10 on the GPU host (RTX 3090, index 0), branch `plan/subtitle-speaker-names`, Task 11 of `docs/superpowers/plans/2026-11-10-subtitle-speaker-names.md`. Every run used copies under `~/fukikae-scratch/names-t11/`; no original workspace changed. The commands are `scripts/names-eval/run_arms.sh`; the ledger rows are exported to `2026-11-10-subtitle-speaker-names.jsonl` (tags `t11b-names-<language>-<arm>`). Every table is the t11b re-measure: the arms rebuilt and rescored on b8afe4c (the remainder rule, below) and 9ad5be5 (eval names follow-ups); the t11 rows are retracted._

## Setup

- **ja set** (`fairytail`): ten copies of `~/fukikae-scratch/t12/fairytail` (E11 to E14 on the anime profile: Demucs, pyannote, WeSpeaker, centroid characters stage, joint resolve at 0.6). Its manifests are complete and current, so `scripts/names-eval/check_keys.py` printed `ok` for extract to diarize in every copy and nothing before `names` reran.
- **en set** (`fairytail-en`): the English dub track of the same four episodes, with the same Crunchyroll English ASS as names and ground truth. I copied `~/fukikae-ws/t9hold/fairytail-en` (E12 to E14 only) to `en-base` and ran E11 through extract, separate, ASR and diarize on GPU 0 (D29 #7). E12 to E14 reran separate, ASR and diarize as well: their separate keys predate commit e9359b5 (cache keys include the resolved model identity), so the stage was stale although the model (Demucs) is the same. The en numbers are therefore on a fresh diarization of all four episodes, not on the t9hold turns the identity-profile results used. en-base kept t9hold's registry, as the ja copies keep t12's; resolve replaces every actor these episodes made.
- **Copies.** `scripts/names-eval/make_arm.sh` hard-links the arm (`cp -al`) and then makes real copies of `fukikae.toml`, the registry, the hash cache, cards, caches and every stage directory from `04c-names` on. It then refuses if `fukikae.toml`, the registry, the hash cache, the cards, a manifest or a `05-characters` file still has a second link; the other copied directories are not checked. Stage outputs are written by rename, so the hard-linked audio, stems, ASR and diarize files stay shared and untouched.
- **Offsets.** Every Fairy Tail episode's `names.json` records `offset_s = 0.0` on both sets, so no gap between an arm and its baseline comes from an offset.
- **Order.** Per arm: `fukikae run <ws> S01E11 S01E12 S01E13 S01E14 --from names --to characters` (episodes in order), then resolve, then `eval identity --from-assignments` and `eval names`.

## Arms

| arm | `[names]` | resolve | scored |
|---|---|---|---|
| baseline | `use = "off"` | E11-E13, then E14 alone | `--tune E11,E12,E13 --holdout E14` |
| names (protocol A) | `use = "auto"`, `holdout_episodes = ["S01E14"]` | E11-E13, then E14 alone | `--tune E11,E12,E13 --holdout E14` |
| ceiling | `use = "auto"` | E11-E13, then E14 alone | `--tune E11,E12,E13 --ceiling E14` |
| hidden (protocol B) | `use = "auto"`, `hide_fraction = 0.2`, `hide_seed = 0` | E11-E14 together | `--tune E11,E12,E13,E14` |
| hidden-base | `use = "off"`, `hide_fraction = 0.2`, `hide_seed = 0` | E11-E14 together | `--tune E11,E12,E13,E14` |

Every arm is a fresh copy built from its source with only the `[names]` table appended (Task 10 review B6). Protocol B tunes on all four episodes (B5). The two protocol B arms carry identical `hide_fraction` and `hide_seed`; `eval names` re-draws the off arm's hidden lines from its own config and has nothing to check that against when `names.json` holds no hidden lines, so I checked the exported ledger rows instead: all eight protocol B `names` rows (ja and en, tune and hidden splits) record `hide_fraction 0.2, hide_seed 0`, with `use` `auto` or `off`. The spec's seed is 0.

The ceiling arm grades E14 on the names its own run read: read it as an upper bound, never as a result. The ja and en sets are the same four episodes in two audio languages.

## Detection (`fukikae names scan`)

Same verdicts on both sets; the en scan differs only in the share of named-line time on speech (87% on E13 and E14 against 78% and 81% on ja).

| episode | verdict | track | named lines | keys | top key | offset | named-line time on speech (ja) |
|---|---|---|---|---|---|---|---|
| S01E11 | full | 2 (en ASS "English", the forced-flagged full track) | 337 of 341 (99%) | 27 | natsu 22% | +0.00 s | 77% |
| S01E12 | full | 4 (en ASS) | 354 of 354 (100%) | 26 | wraith 12% | +0.00 s | 77% |
| S01E13 | full | 4 | 250 of 251 (100%) | 19 | touka 18% | +0.00 s | 78% |
| S01E14 | full | 4 | 321 of 323 (99%) | 20 | wendy 26% | +0.00 s | 81% |

E12 suggests `nabaru` (3 lines) as one edit from `nebaru` (18 lines); `labels-aliases.yaml` already maps it for the ground truth, and the pipeline left it alone (no fuzzy matching). The names stage took 0.08 to 0.14 s of stage time per episode (the track was already cached); a whole names plus characters pass ran in about 1.5 s per episode.

## Protocol A: held-out E14

### ja

| arm | split | recurring | passing | fraction | merges | modal share | strict |
|---|---|---|---|---|---|---|---|
| baseline | tune | 5 | 3 | 0.60 | 0 | 82.9% | 0.17 |
| baseline | holdout S01E14 | 12 | 6 | 0.50 | 8 | 65.3% | 0.19 |
| names | tune | 5 | 5 | 1.00 | 0 | 100.0% | 1.00 |
| names | holdout S01E14 | 12 | 6 | 0.50 | 8 | 46.1% | 0.25 |
| ceiling | tune | 5 | 5 | 1.00 | 0 | 100.0% | 1.00 |
| ceiling | ceiling S01E14 | 12 | 12 | 1.00 | 0 | 100.0% | 1.00 |

baseline holdout merges: CHARLE + HAPPY (id 2); CHARLE + JUVIA (id 2); CHARLE + WENDY (id 2); HAPPY + JUVIA (id 2); HAPPY + WENDY (id 2); JUVIA + WENDY (id 2); CANA + LUCY (id 4); KIRIA + NATSU (id 9)

names holdout merges: CANA + LUCY (id 5); CHARLE + HAPPY (id 31); CHARLE + JUVIA (id 31); CHARLE + WENDY (id 31); HAPPY + JUVIA (id 31); HAPPY + WENDY (id 31); JUVIA + WENDY (id 31); KIRIA + NATSU (id 37)

Per character on E14 (share of the character's E14 speech on its modal tuning id; F is a fail):

| character | baseline | names | ceiling |
|---|---|---|---|
| NATSU | 0.0% (modal 20) F | 0.0% (modal 9) F | 100.0% (modal 9) |
| LUCY | 67.3% (modal 4) F | 0.0% (modal 4) F | 100.0% (modal 4) |
| WENDY | 84.4% (modal 2) | 0.0% (modal 22) F | 100.0% (modal 22) |
| HAPPY | 100.0% (modal 2) | 0.0% (modal 18) F | 100.0% (modal 18) |
| WHITE MAGE | 100.0% (modal 15) | 100.0% (modal 15) | 100.0% (modal 15) |
| CANA | 0.0% (modal 19) F | 100.0% (modal 5) | 100.0% (modal 5) |
| KIRIA | 49.8% (modal 6) F | 49.8% (modal 21) F | 100.0% (modal 21) |
| NEBARU | 94.5% (modal 1) | 94.5% (modal 1) | 100.0% (modal 1) |
| SKULLION | 0.0% (modal 17) F | 100.0% (modal 27) | 100.0% (modal 27) |
| JUVIA | 0.0% (modal 4) F | 0.0% (modal 29) F | 100.0% (modal 29) |
| CHARLE | 100.0% (modal 2) | 100.0% (modal 31) | 100.0% (modal 31) |
| GRAY | 81.3% (modal 11) | 81.3% (modal 11) | 100.0% (modal 11) |

`eval names` (cue accuracy speech-weighted; purity is pure / (pure + foreign) in ground-truth seconds; max chars/name lists modal names that more than one character stands for):

| arm | split | cues | cue accuracy | hidden accuracy | purity | pure s | foreign s | unverifiable s (share) | max chars/name |
|---|---|---|---|---|---|---|---|---|---|
| baseline | tune | 941 | 62.9% | - | 66.0% | 75.4 | 38.8 | 1.4 (1.2%) | natsu 2 |
| baseline | holdout | 321 | 25.8% | - | 36.4% | 12.7 | 22.1 | 0.2 (0.7%) | natsu 2 |
| names | tune | 941 | 97.1% | - | 99.8% | 122.1 | 0.3 | 10.4 (7.8%) | touka 2 |
| names | holdout | 321 | 32.7% | - | 73.9% | 25.7 | 9.1 | 0.2 (0.7%) | touka 2 |
| ceiling | tune | 941 | 97.1% | - | 99.8% | 122.1 | 0.3 | 10.4 (7.8%) | touka 2 |
| ceiling | ceiling | 321 | 83.9% | - | 99.5% | 59.9 | 0.3 | 0.9 (1.5%) | touka 2 |

### en

| arm | split | recurring | passing | fraction | merges | modal share | strict |
|---|---|---|---|---|---|---|---|
| baseline | tune | 5 | 2 | 0.40 | 0 | 69.3% | 0.00 |
| baseline | holdout S01E14 | 12 | 6 | 0.50 | 6 | 61.5% | 0.31 |
| names | tune | 5 | 5 | 1.00 | 0 | 98.7% | 0.50 |
| names | holdout S01E14 | 12 | 6 | 0.50 | 6 | 58.5% | 0.25 |
| ceiling | tune | 5 | 5 | 1.00 | 0 | 98.7% | 0.50 |
| ceiling | ceiling S01E14 | 12 | 12 | 1.00 | 0 | 100.0% | 1.00 |

baseline holdout merges: HAPPY + JUVIA (id 2); HAPPY + LUCY (id 2); HAPPY + WENDY (id 2); JUVIA + LUCY (id 2); JUVIA + WENDY (id 2); LUCY + WENDY (id 2)

names holdout merges: HAPPY + JUVIA (id 33); HAPPY + LUCY (id 33); HAPPY + WENDY (id 33); JUVIA + LUCY (id 33); JUVIA + WENDY (id 33); LUCY + WENDY (id 33)

| character | baseline | names | ceiling |
|---|---|---|---|
| WENDY | 84.2% (modal 2) | 84.2% (modal 33) | 100.0% (modal 33) |
| NATSU | 78.4% (modal 8) F | 78.4% (modal 21) F | 100.0% (modal 21) |
| LUCY | 0.0% (modal 35) F | 0.0% (modal 24) F | 100.0% (modal 24) |
| HAPPY | 100.0% (modal 2) | 0.0% (modal 27) F | 100.0% (modal 27) |
| CANA | 100.0% (modal 12) | 100.0% (modal 12) | 100.0% (modal 12) |
| WHITE MAGE | 28.7% (modal 4) F | 28.7% (modal 48) F | 100.0% (modal 48) |
| KIRIA | 100.0% (modal 5) | 100.0% (modal 32) | 100.0% (modal 32) |
| SKULLION | 0.0% (modal 26) F | 0.0% (modal 38) F | 100.0% (modal 38) |
| GRAY | 100.0% (modal 11) | 100.0% (modal 11) | 100.0% (modal 11) |
| CHARLE | 0.0% (modal 2) F | 88.5% (modal 42) | 100.0% (modal 42) |
| NEBARU | 89.8% (modal 20) | 89.8% (modal 31) | 100.0% (modal 31) |
| JUVIA | 0.0% (modal 12) F | 0.0% (modal 40) F | 100.0% (modal 40) |

| arm | split | cues | cue accuracy | hidden accuracy | purity | pure s | foreign s | unverifiable s (share) | max chars/name |
|---|---|---|---|---|---|---|---|---|---|
| baseline | tune | 941 | 63.7% | - | 84.2% | 195.7 | 36.7 | 9.8 (4.0%) | lucy 3, natsu 2, touka 2, wraith 3 |
| baseline | holdout | 321 | 27.0% | - | 56.8% | 54.1 | 41.1 | 2.2 (2.2%) | lucy 3, natsu 2, touka 2, wraith 3 |
| names | tune | 941 | 97.4% | - | 99.7% | 246.3 | 0.9 | 10.9 (4.2%) | juvia 2, nebaru 2 |
| names | holdout | 321 | 44.6% | - | 68.8% | 74.0 | 33.6 | 2.2 (2.0%) | juvia 2, nebaru 2 |
| ceiling | tune | 941 | 97.4% | - | 99.7% | 246.3 | 0.9 | 10.9 (4.2%) | juvia 2, nebaru 2 |
| ceiling | ceiling | 321 | 83.9% | - | 99.6% | 142.8 | 0.6 | 2.9 (2.0%) | juvia 2, nebaru 2 |

### What protocol A says

The names arm's held-out row is no better than baseline's on identity: 6 of 12 recurring characters pass in both arms on both sets, with the same merges (ja 8, en 6). The modal-id share falls (ja 65.3% to 46.1%, en 61.5% to 58.5%). The tuning registry is cleaner (ja 5 of 5 against 3 of 5, en 5 of 5 against 2 of 5), and that is what costs share on E14: in baseline, Wendy, Happy and Charle are one character (2) in tuning as well, so E14's merged diarize cluster lands on the modal id of all three. On ja all three pass in baseline; with names, tuning gives Wendy (22), Happy (18) and Charle (31) their own characters, E14's merged cluster still goes to one of them (Charle, 31), and only Charle passes. On en, baseline passes Wendy and Happy and fails Charle (0.0% on 2); with names, the merged cluster goes to Wendy (33), so Wendy passes and Happy fails, and Charle, outside that cluster on en, passes on its own character (88.5% on 42).

That is the structural limit, and the reason protocol A measures acoustic generalisation only: names act on the speech they cover, through sub-clusters cut from diarize clusters. In an episode whose names the pipeline never reads, nothing splits a diarize cluster that holds three voices, and the characters stage and resolve can only send that cluster whole to one character. A cleaner registry helps E14 only where E14's clusters are themselves clean (ja Cana and Skullion, en Charle pass with names and fail without).

Cue accuracy and reference purity move where identity doesn't. On the held-out episode, cue accuracy rises from 25.8% to 32.7% (ja) and from 27.0% to 44.6% (en), and E14 reference purity from 36.4% to 73.9% (ja) and 56.8% to 68.8% (en). With E14's names read (ceiling), identity passes 12 of 12 on both sets, cue accuracy reaches 83.9%, and purity 99.5% (ja) and 99.6% (en).

## Reference purity, before and after

Per character, the share of the auto-picked reference seconds (`character_references`, computed over the split's episodes, not rendered) under lines of the character's own modal name. Baseline characters have no names, so they appear as ids.

| set | split | baseline | names | ceiling |
|---|---|---|---|---|
| ja | tune (E11-E13) | 66.0% (38.8 s foreign) | 99.8% (0.3 s foreign) | 99.8% |
| ja | E14 | 36.4% (22.1 s foreign) | 73.9% (9.1 s foreign) | 99.5% (0.3 s foreign) |
| en | tune (E11-E13) | 84.2% (36.7 s foreign) | 99.7% (0.9 s foreign) | 99.7% |
| en | E14 | 56.8% (41.1 s foreign) | 68.8% (33.6 s foreign) | 99.6% (0.6 s foreign) |

Unverifiable seconds (references under no subtitle line) are 0.7 to 7.8% of all reference seconds in every cell; the per-split numbers are in the tables above. Task 9 measured 51.0% on the old t12 cards (references picked before names existed); this table recomputes the picks from each arm's assignments.

ja, E14 per character:

| character | baseline | names | ceiling |
|---|---|---|---|
| Cana | - | 47.3% (4.0 s foreign) | 100.0% (0.0 s foreign) |
| Charle | - | 51.1% (3.6 s foreign) | - |
| Happy | - | - | 99.5% (0.0 s foreign) |
| Kiria | - | - | 100.0% (0.0 s foreign) |
| Lucy | - | - | 100.0% (0.0 s foreign) |
| Nebaru | - | 100.0% (0.0 s foreign) | 100.0% (0.0 s foreign) |
| Skullion | - | 82.6% (1.2 s foreign) | 100.0% (0.0 s foreign) |
| Wendy | - | - | 100.0% (0.0 s foreign) |
| White Mage | - | 96.2% (0.2 s foreign) | 96.2% (0.2 s foreign) |
| character 1 | 100.0% (0.0 s foreign) | - | - |
| character 15 | 96.2% (0.2 s foreign) | - | - |
| character 2 | 0.0% (7.4 s foreign) | - | - |
| character 4 | 0.0% (7.7 s foreign) | - | - |
| character 7 | 0.0% (6.8 s foreign) | - | - |

en, E14 per character:

| character | baseline | names | ceiling |
|---|---|---|---|
| Cana | - | 91.1% (1.5 s foreign) | 100.0% (0.0 s foreign) |
| Charle | - | 100.0% (0.0 s foreign) | 100.0% (0.0 s foreign) |
| Erza | - | 0.0% (7.7 s foreign) | - |
| Gray | - | 100.0% (0.0 s foreign) | 100.0% (0.0 s foreign) |
| Happy | - | - | 100.0% (0.0 s foreign) |
| Juvia | - | - | 100.0% (0.0 s foreign) |
| Kiria | - | 100.0% (0.0 s foreign) | 100.0% (0.0 s foreign) |
| Lucy | - | - | 97.8% (0.3 s foreign) |
| Natsu | - | 72.1% (3.8 s foreign) | 97.5% (0.4 s foreign) |
| Nebaru | - | 100.0% (0.0 s foreign) | 100.0% (0.0 s foreign) |
| Skullion | - | - | 99.8% (0.0 s foreign) |
| Touka | - | 0.0% (17.0 s foreign) | - |
| Wendy | - | 73.1% (3.6 s foreign) | 100.0% (0.0 s foreign) |
| White Mage | - | - | 100.0% (0.0 s foreign) |
| character 11 | 100.0% (0.0 s foreign) | - | - |
| character 12 | 91.1% (1.5 s foreign) | - | - |
| character 15 | 0.0% (17.0 s foreign) | - | - |
| character 2 | 17.2% (11.1 s foreign) | - | - |
| character 20 | 100.0% (0.0 s foreign) | - | - |
| character 5 | 100.0% (0.0 s foreign) | - | - |
| character 7 | 0.0% (7.7 s foreign) | - | - |
| character 8 | 72.1% (3.8 s foreign) | - | - |

## Protocol B: hidden lines

### ja

| arm | split | recurring | passing | fraction | merges | modal share | strict |
|---|---|---|---|---|---|---|---|
| hidden-base | tune | 4 | 2 | 0.50 | 0 | 80.5% | 0.00 |
| hidden | tune | 4 | 4 | 1.00 | 0 | 91.4% | 0.17 |

| arm | split | cues | cue accuracy | hidden accuracy | purity | pure s | foreign s | unverifiable s (share) | max chars/name |
|---|---|---|---|---|---|---|---|---|---|
| hidden-base | tune | 1262 | 59.3% | 59.4% | 62.5% | 83.6 | 50.1 | 1.6 (1.2%) | natsu 2 |
| hidden-base | hidden | 235 | 59.4% | - | 62.5% | 83.6 | 50.1 | 1.6 (1.2%) | natsu 2 |
| hidden | tune | 1262 | 91.0% | 63.8% | 94.7% | 161.1 | 9.0 | 6.2 (3.5%) | touka 2 |
| hidden | hidden | 235 | 63.8% | - | 94.7% | 161.1 | 9.0 | 6.2 (3.5%) | touka 2 |

### en

| arm | split | recurring | passing | fraction | merges | modal share | strict |
|---|---|---|---|---|---|---|---|
| hidden-base | tune | 4 | 2 | 0.50 | 0 | 74.0% | 0.00 |
| hidden | tune | 4 | 4 | 1.00 | 0 | 90.5% | 0.12 |

| arm | split | cues | cue accuracy | hidden accuracy | purity | pure s | foreign s | unverifiable s (share) | max chars/name |
|---|---|---|---|---|---|---|---|---|---|
| hidden-base | tune | 1262 | 62.0% | 63.7% | 77.8% | 219.0 | 62.5 | 10.4 (3.6%) | lucy 2, natsu 2, touka 3, wraith 3 |
| hidden-base | hidden | 235 | 63.7% | - | 77.8% | 219.0 | 62.5 | 10.4 (3.6%) | lucy 2, natsu 2, touka 3, wraith 3 |
| hidden | tune | 1262 | 89.9% | 57.2% | 95.8% | 282.0 | 12.5 | 9.5 (3.1%) | juvia 2, nebaru 2, touka 2, white mage 2 |
| hidden | hidden | 235 | 57.2% | - | 95.8% | 282.0 | 12.5 | 9.5 (3.1%) | juvia 2, nebaru 2, touka 2, white mage 2 |

The `hidden` rows' purity is computed over the same four episodes as the `tune` rows (references aren't per line), so it repeats the tune purity and is not a hidden-line measure.

On the hidden lines alone, names help little on ja and still cost on en: ja 59.4% (off) against 63.8% (names), and en 63.7% against 57.2%, a loss of 6.5 points. Purity rises as in protocol A (62.5% to 94.7% ja, 77.8% to 95.8% en), and so does the identity harness (2 of 4 to 4 of 4 on both).

Per character, hidden accuracy off to names:

| character | ja | en |
|---|---|---|
| natsu | 66% to 32% | 58% to 31% |
| lucy | 81% to 35% | 72% to 46% |
| happy | 0% to 71% | 0% to 39% |
| young makarov | 0% to 78% | 0% to 78% |
| cana | 49% to 86% | 82% to 70% |
| wraith | 89% to 89% | 97% to 30% |

Natsu is still the clearest case. His named lines fall in 4 to 7 different diarize clusters per episode (ja hidden arm: E12 SPEAKER_09, 10, 12 and 13; E13 seven clusters), so the names stage cuts a `~natsu` sub-cluster out of each, and his hidden lines stay in each cluster's unnamed remainder. Of his 32 hidden lines on ja, 23 sit in a remainder that kept embedding rows of its own (on en, 21): the remainder rule doesn't touch those, and the matcher sends them to whoever the remainder sounds like, or makes a new character of them (9 on ja, 12 on en). Without names, the one cluster that is mostly Natsu goes to one character whose modal name is natsu, and his hidden lines in that cluster score. The section below has what the remainder rule changed.

Under `use = "auto"` on a full track this touches only the 0 to 1.2% of lines that are unnamed; on a partial track it is the main effect.

## The remainder rule (Task 11 ruling, amending R1)

Task 11 first ran with R1: a remainder with no embedding rows and no pin took its parent diarize cluster's acoustic match. In a cluster names split between several voices, that sent the unnamed speech to whoever the whole cluster sounded like. Commit b8afe4c replaces it, in the characters stage and in `registry resolve` alike (`fukikae.registry.siblings`): such a remainder takes the character holding at least 60% of its parent's named seconds (source `voice`, with a report line saying it follows its named siblings); when the parent's named speech splits with no character at 60%, it is `extra`; only a parent with no named sub-cluster gives it the parent's match, as R1 did. A role label is not a named sub-cluster (it names no character, D29 #6), so a parent holding only role labels keeps R1's path; a named key that went extra (under `min_new_seconds`) counts in the parent's named seconds, held by no character.

Rowless remainders with named siblings in the final assignments (after resolve, which has no pins; every one agrees with the rule):

| arm | ja: follow / extra | en: follow / extra |
|---|---|---|
| names | 19 / 3 | 24 / 4 |
| ceiling | 24 / 5 | 33 / 6 |
| hidden | 4 / 0 | 8 / 2 |

The followers took their sibling at 68 to 100% of the parent's named seconds. Among the extras are ceiling E14 SPEAKER_08 (Wendy 56% of 243 s, Happy and Charle the rest) and E11 SPEAKER_11 (Igneel 58% of 68 s). The hidden arms have few: a hidden line leaves its speech, rows included, in the remainder, so most of their remainders are matched on their own rows.

| measure | set | before (R1) | after (rule) |
|---|---|---|---|
| protocol B hidden accuracy | ja | 60.7% | 63.8% |
| protocol B hidden accuracy, Natsu | ja | 10% | 32% |
| protocol B hidden accuracy | en | 54.9% | 57.2% |
| protocol B hidden accuracy, Natsu | en | 16% | 31% |
| protocol B cue accuracy (tune, all cues) | ja / en | 90.4% / 89.5% | 91.0% / 89.9% |
| protocol A E14 identity (passing of 12) | ja / en | 6 / 6 | 6 / 6 |
| protocol A E14 modal share | ja / en | 46.1% / 58.5% | 46.1% / 58.5% |
| protocol A E14 cue accuracy | ja / en | 32.7% / 44.6% | 32.7% / 44.6% |
| reference purity, tune | ja / en | 99.8% / 99.7% | 99.8% / 99.7% |
| reference purity, E14 held out | ja / en | 73.9% / 68.8% | 73.9% / 68.8% |
| reference purity, protocol B | ja / en | 94.7% / 95.8% | 94.7% / 95.8% |
| characters per name over 1, names arm | ja / en | 3 / 5 names | 1 / 2 names |

The rule was ruled after the protocol B and held-out results above were seen (Natsu's 10% and 16% hidden accuracy under R1 prompted it), so its protocol B "after" numbers are not clean held-out numbers: the hidden lines informed the rule they now grade. Rule changes stopped there, with the remaining gap recorded under Limits, so that no further rule is tuned on the hidden or held-out data.

The baseline arms (`use = "off"`) read no names, so they have no remainders: their assignments are byte-identical to Task 11's on all four episodes of both sets, and every number in their rows is unchanged. Protocol A doesn't move because E14 is held out (no remainders there) and the tuning registry's characters keep their names; purity doesn't move because a character with name-confirmed turns draws its references from those alone, never from a remainder. What the rule removes is the over-split: on ja the leftover character 2 (the old Wendy, Happy and Charle merge, which three E11 and E13 remainders kept alive by their parent's match) and character 10 are gone, and on en characters 2, 13, 16 and 20.

The eval names follow-ups (9ad5be5) moved nothing on Fairy Tail: the hidden tally now counts a cue only when a hidden line of its own key overlaps it (`cues._hidden`, as the subtitle path tests it), and both off arms score the same 235 cues as before; cues of an episode whose names weren't used are attributed with the alignment shift, and no Fairy Tail episode needed one (every baseline number is unchanged); each `names` row's detail records `hide_fraction` and `hide_seed`.

## Coverage

ja, ceiling arm (every episode named):

| episode | verdict | offset s | named share | sub-clusters | cut turns | dropped row s | unvoiced | hidden | suggestions |
|---|---|---|---|---|---|---|---|---|---|
| S01E11 | full | 0.0 | 98.8% | 72 | 4 | 8.8 | 6 | 0 | - |
| S01E12 | full | 0.0 | 100.0% | 82 | 5 | 15.1 | 4 | 0 | ['nabaru', 'nebaru'] |
| S01E13 | full | 0.0 | 99.6% | 47 | 6 | 18.6 | 4 | 0 | - |
| S01E14 | full | 0.0 | 99.4% | 57 | 13 | 60.9 | 7 | 0 | - |

en, ceiling arm:

| episode | verdict | offset s | named share | sub-clusters | cut turns | dropped row s | unvoiced | hidden | suggestions |
|---|---|---|---|---|---|---|---|---|---|
| S01E11 | full | 0.0 | 98.8% | 69 | 4 | 12.6 | 7 | 0 | - |
| S01E12 | full | 0.0 | 100.0% | 85 | 25 | 104.8 | 3 | 0 | ['nabaru', 'nebaru'] |
| S01E13 | full | 0.0 | 99.6% | 59 | 16 | 91.2 | 5 | 0 | - |
| S01E14 | full | 0.0 | 99.4% | 64 | 21 | 103.1 | 7 | 0 | - |

ja, hidden arm:

| episode | verdict | offset s | named share | sub-clusters | cut turns | dropped row s | unvoiced | hidden | suggestions |
|---|---|---|---|---|---|---|---|---|---|
| S01E11 | partial | 0.0 | 80.9% | 68 | 3 | 5.0 | 5 | 61 | - |
| S01E12 | partial | 0.0 | 81.6% | 72 | 4 | 12.6 | 3 | 65 | - |
| S01E13 | partial | 0.0 | 80.9% | 46 | 3 | 14.5 | 2 | 47 | - |
| S01E14 | partial | 0.0 | 80.2% | 55 | 7 | 31.3 | 7 | 62 | - |

en, hidden arm:

| episode | verdict | offset s | named share | sub-clusters | cut turns | dropped row s | unvoiced | hidden | suggestions |
|---|---|---|---|---|---|---|---|---|---|
| S01E11 | partial | 0.0 | 80.9% | 62 | 3 | 8.8 | 6 | 61 | - |
| S01E12 | partial | 0.0 | 81.6% | 77 | 18 | 75.7 | 3 | 65 | - |
| S01E13 | partial | 0.0 | 80.9% | 50 | 11 | 71.6 | 5 | 47 | - |
| S01E14 | partial | 0.0 | 80.2% | 59 | 12 | 76.6 | 7 | 62 | - |

The dub cuts many more turns (E12: 25 against 5) and drops more row seconds (E12: 104.8 s against 15.1 s). The ASS is timed to the Japanese audio, and the English dub runs its lines together, so more diarize turns hold two named lines.

## Characters per name

Several characters standing for one modal name (an over-split registry) each score as that name in `eval names`. ja names arm: touka 2. en names arm: juvia 2, nebaru 2. The baselines over-split too (ja natsu 2; en lucy 3, wraith 3, natsu 2, touka 2). Before the remainder rule the ja names arm also had happy 2 (characters 2 and 18) and young goldmine 2, and en cana, natsu and white mage 2 and nebaru 3: character 2, the old Wendy, Happy and Charle merge, survived only because three rowless remainders in E11 and E13 took it by their parent's match. With the rule they follow their named siblings and character 2 is pruned.

## The character 2 case

In t12 (no names) character 2 held Wendy, Happy and Charle, and its automatic card was cloned from Wendy's turns; Natsu's E12 line c0331 ("I'd even settle for Jellal!") was spoken by character 2. In the ja names arm E11 created Wendy (22), Happy (18) and Charle (31) as their own characters, c0331 is now character 9 (Natsu), and the cards the tts stage re-picked for stale references were Lucy (4), Cana (5), Gray (11) and Erza (12), each cloned from turns now named for another character (log lines in `07-tts`). With E14 held out, E14's merged cluster still goes to one character (Charle, 31), so Natsu's E14 lines are on character 37 (a resolve-created actor whose modal name is touka) rather than 9; with E14 named (ceiling), all of Natsu's E14 speech is on 9.

## Defect found and fixed: dangling actors after a held-out resolve

The first run of the arms left assignments naming characters the registry no longer held: three clusters of ja E11 and E13 named character 2, and five clusters of en E12 and E13 named characters 2, 5, 7 and 20, all in the names arms. A rowless remainder (no embedding rows of its own) takes its parent cluster's acoustic match and adds no exemplar (R1), so character 2's exemplars all came from E14; resolving E14 alone deleted them and pruned character 2. Commit 0f9e4ee makes both `registry resolve` and the characters stage skip any orphan another episode's `assignments.json` still names (`fukikae.registry.usage.actors_in_use`), with a test for each path. I rebuilt and rescored every arm on that commit (the t11 re-run): no arm had a dangling id, and every number matched the first run (the eval scores the ids in the assignments either way). The 36 first-run ledger rows are retracted, and so are the t11 re-run's: the tables above are t11b, re-measured on b8afe4c and 9ad5be5.

## Listening check (D29 #9)

I ran the ja names arm through tag, translate and tts for E12 to E14 on GPU 0 (`fukikae run <ws> S01E12 S01E13 S01E14 --to tts`; the normal stage re-picked the stale automatic cards of Lucy, Cana, Gray and Erza and replaced nothing curated). IndexTTS2 ran in the main checkout's worker environment (`FUKIKAE_WORKER_CHECKOUT=~/projects/fukikae`; `workers/` is identical on this branch, and the worktree has no `.venvs/indextts2`). A first attempt without it made stock fallback cards, which the second run replaced as designed. Eight renders were flagged suspect after the seed retry.

Where the cues of the two characters went (ground truth from the subtitle Name, before = t12, after = names arm):

| character | episode | cues | before to after |
|---|---|---|---|
| Natsu | E12 | 40 | 26 already on 9; 14 moved to 9 from 2 (7), 8 (3), 3 (3), 11 (1) |
| Natsu | E13 | 27 | 8 already on 9; 19 moved to 9 from 3 (11), 2 (3), none (3), 7 (1), 11 (1) |
| Natsu | E14 (held out) | 22 | 15 moved from 9 to 37; 5 from 2 to 31 (Charle); 1 from 13 to 38; 1 stayed on 11 |
| Young Makarov | E12 | 16 | all 16 moved to 30 from 1 (10), 3 (3), 10 (2), none (1) |

In the episodes whose names were read, every Natsu and Young Makarov cue is now on its own character. In the held-out E14, 15 Natsu cues that t12 had on character 9 moved to character 37, whose card was cloned from 9.3 s of E13 speech under no subtitle line ("references are not name-confirmed"). That is the protocol A merge (KIRIA + NATSU on 37) heard in the dub, and the listening pack includes three of those cues.

The pack is `~/fukikae-review/names-natsu/` (6.9 MB): 10 Natsu cues (E12, E13 and E14 in turn, longest first) and 10 Young Makarov cues (all E12; he speaks in no other episode), each with `original.wav` (the Japanese line, the tts emotion reference) and the two renders as `A.wav` and `B.wav` in a shuffled order (seed 0). `index.html` plays them, `README.txt` says what to listen for, and `key.toml` holds which letter is before and which is after, with the characters and cards. Built by `scripts/names-eval/listening_pack.py`. No verdict yet: the listening is the operator's.

The pack was rendered from the Task 11 (R1) names arm, kept as `~/fukikae-scratch/names-t11/ja-names-t11`. `scripts/names-eval/pack_check.py` attributes each of the 20 pack cues in that arm (agreeing with its `cues.en.json` on every cue of E12 to E14) and in the t11b arm built with the remainder rule, and compares the reference picks of the character each lands on: all 20 keep their character (9, 30, and 37 and 31 on E14) and every one of those characters keeps the same reference picks, so no pack cue would get a different character or card. The pack stands as it is.

## Where the numbers come from

- **Ledger rows** (`2026-11-10-subtitle-speaker-names.jsonl`, tags `t11b-names-<language>-<arm>`): every split-level row of the identity and `eval names` tables (recurring, passing, fraction, merges, modal share, strict; cues, cue and hidden accuracy, purity and its seconds, characters per name).
- **`eval identity --json`** outputs, `~/fukikae-scratch/names-t11/out-t11b/<language>-<arm>.identity.json`: the merges lines and the per-character E14 tables (share and modal id), rendered by `scripts/names-eval/tables.py`. These rows predate 269539b, which records each recurring character's modal id and share in the row's detail from now on; the existing rows were not rewritten.
- **`eval names --json`** outputs, `<language>-<arm>.names.json` beside them: the per-character purity tables (`purity_per_character`, `purity_seconds_per_character`) and the per-character hidden accuracy (`hidden.per_character`), also through `tables.py`.
- **Each arm's `04c-names/names.json`**: the coverage tables (`tables.py`); the detection table is `fukikae names scan`'s output.
- **One-off queries over the arms' `assignments.json`, `names.json` and `names-report.json`**, not kept as scripts: the rowless remainder counts and their follower shares, Natsu's clusters and where his hidden lines sit, the characters per name before the rule, the character 2 case and the listening cue moves. The listening pack's checks are `listening_pack.py` and `pack_check.py`.
- **The "before (R1)" column** of the remainder rule table: the retracted t11 rows, with their json outputs and assignments kept under `~/fukikae-scratch/names-t11/before-t11b/`.

## Limits and follow-ups

- One series, four episodes, two audio languages of the same episodes. Every recurring-character count is small (5 in tuning, 12 on E14), so one character is 8 to 20 points of fraction.
- Protocol A cannot show a gain in identity on an unnamed episode while diarization merges voices there; the gain names give is in the episodes that have them. A follow-up that would move E14 is an acoustic one (diarization or embedding that separates Wendy, Happy and Charle), not a names one.
- Protocol B's loss on en, and Natsu's on both sets, now come from remainders that kept embedding rows of their own: the remainder rule decides only rowless ones, and the matcher sends the others to whoever they sound like or makes new characters of them. Nothing here changes a threshold; every threshold stayed at the spec's value. The remainder rule itself came after the protocol B and held-out results (above), so its protocol B gain is not a clean held-out measurement; a fix for rowed remainders needs new held-out data to be graded on.
- The en E12 to E14 diarization was redone (stale separate key), so the en baseline here is not comparable row for row with `2026-11-07-identity-profiles.md`.
- The t11b ledger rows were written from a dirty tree (commit 9ad5be5 plus the uncommitted `run_arms.sh` and `pack_check.py` edits and this doc); the code under test is 9ad5be5.
