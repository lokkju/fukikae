# Live-action ground truth: test sets, profile pick and held-out verdicts

_Run 2026-11-08 and 2026-11-09 on the GPU host (RTX 3090, index 0), branch feat/ground-truth. The sweep used tuning episodes only. The live-action profile was committed in `6bb10c2` before any held-out episode was scored, and each held-out episode was scored once after that, with nothing re-tuned (Ruling R18). Recurring means 5 s of labelled speech per episode compared (R6); a set passes when 90% of its recurring characters keep 80% of their speech on one id and no two share a modal id. Ledger rows (658, exported with `fukikae eval report --export`; the tags are listed under "Reproducing"): `docs/results/2026-11-09-live-action-ground-truth.jsonl` holds the baselines, diagnostics, held-out and clip-set rows (172), and `2026-11-09-live-action-ground-truth-sweep-ct{0.4,0.5,0.6}.jsonl` the sweep, 162 rows per clustering value, split to stay under the repository's 500 KB file limit. Kaze, Kaoru's held-out E11 is still downloading and is **pending**; it gets scored once, with the profile committed here, in a follow-up._

## Test sets and how each label set was made

| Set | Language | Labels | Tuning | Held-out | Recurring (tuning) |
|---|---|---|---|---|---|
| `bbt` (en, de, es) | en, de, es | bigbangtrans transcripts aligned to the Blu-ray's English subtitles | S01E01-E02 | S01E03 | 4 on each track |
| `bazinga-tbbt` (PAL, film) | en | Bazinga gold speakers, word-level | S01E01-E03 | S01E05 | 4 |
| `bazinga-office` (PAL, film) | en | Bazinga gold speakers, word-level | S01E01-E03 | S01E05 | 3 on PAL, 4 on film |
| `kaze` (風、薫る) | ja | ARIB colour captions plus （名前） prefixes, retimed to the WEB-DL | S01E01-E03 | S01E11 (pending) | 1, thin |
| `grid` | ko | Korean CC speaker prefixes | S01E02, S01E04 | S01E05 | 2, thin |
| `aib` (Alice in Borderland) | ja | Netflix Japanese SDH prefixes | S01E01-E02 | not scored | 2, thin, diagnostic only (GT8) |

**The Big Bang Theory.** The release's English subtitle track is image-based (`dvd_subtitle`), and I OCR'd it: ffmpeg renders one frame per subtitle event and tesseract 5.5 reads it (407, 364 and 379 events on E01-E03; the usual slip is `|` for `I`, which touches no name). The cues keep the disc's own timing. `fukikae eval transcript-align` then puts the speaker names from the bigbangtrans.wordpress.com transcripts onto those cues: 98%, 97% and 97% of cues (96%, 95% and 94% of words) on E01-E03. The aliases are Leslie to Lesley, Dr Gablehouser to Gablehouser, and the buzzer, phone and "Voice" lines to `?`. The English timing scores all three audio tracks, which share the video timeline: auto-label's overlap is 83%/81% (en), 86%/85% (de) and 80%/80% (es) on E01/E02 at an offset of 0.0 to -0.2 s.

**How accurate the transcript aligner is.** Bazinga's gold TBBT S01 labels give an independent check on E01-E03 (Task 7c). I scaled them to film speed (1.04271), retimed them onto our English audio with `subalign --speech` (offset +0.76 to +0.90 s, scale 1.000 to 0.999) and compared speakers frame by frame where both sides name exactly one speaker: **98.17% agreement over 1958 s**, which is about 73% of the three episodes' ~2700 s of dialogue. Line level, 1039 of 1058 of our lines (98.2%) go to the gold speaker holding most of their time. The 19 disagreeing lines are one-word cues whose display runs into the next turn, repeated short tokens ("Hi." four times) matched to the wrong transcript instance, a few slips in Bazinga's own boundaries, and Kurt, whom the transcript calls both "Enormous man" and "Man".

**Bazinga.** `fukikae eval bazinga-labels` turns Bazinga's word annotations (revision 96db30ff) into named ASS: consecutive words of one speaker form a turn, a gap over 1.0 s starts a new one, `all`, `multiple_persons` and `unknown` speakers are dropped, and a turn's span comes only from words at confidence 0.9 or above (0.1 marks evenly spaced filler). TBBT keeps 2546-3282 words per episode and The Office 2595-3216. The audio is the PAL DVD's, 4.27% fast and about +0.71 semitone. Each show has two workspaces: `-pal` as shipped, and `-film`, slowed with `asetrate=16000*0.95904` and labels scaled by 1.04271. VAD onsets against turn starts on S01E01 give a median gap of 0.07-0.08 s, a lag within ±0.02 s and drift at most 6e-5 s/s in all four. The two conditions are reported with their own recurring lists: on The Office, Pam has 4.3 s of labelled speech in E02 on PAL and 6.1 s on film; the 5 s bar makes her recurring only on film. I didn't normalise the bar for that, and only the PAL condition enters the pick.

**Kaze, Kaoru.** NHK's ARIB captions colour the main cast (yellow りん, cyan 直美, green 真風) and prefix some other lines with （名前）. `fukikae eval colour-map` builds the series-wide map from the named lines of E01-E15 with no conflicts; that map is ground-truth construction and may use the held-out episode's named lines (controller ruling). Labels are the colour map plus the prefixes, and a line whose colour and prefix disagree is dropped (GT4); white stays unmapped and out of the bar. Green narrates E01-E04 and is an on-screen fortune teller tagged （真風） from E05; WeSpeaker scores E01 narration against E05 narration at 0.907 and against E05's on-screen 真風 at 0.821, against 0.17-0.38 for りん and 直美. Green scores as 真風 throughout (the operator's check by ear is queued). The captions come from a different capture than the WEB-DL audio. `eval subalign` retimed each broadcast .ass to the WEB-DL's .srt by text: 100% inliers on all 15 episodes, offset -0.50 to -0.55 s, scale within 6e-5 of 1, rms 7-13 ms. Against speech, the retimed captions still **lead by a median 0.24-0.31 s** on E01-E03 (pyannote turns and an energy VAD agree). Moving every caption +0.25 s changes no verdict and no share (りん 0.79 at ct 0.6 and 0.84 at ct 0.4), and the retimed files stay the ground truth. The harness's count is thin: only りん reaches 5 s in all three tuning episodes (45, 27 and 31 s), and every Kaze command needs `--force` (overlap 57-65%, long display times).

**Grid** is the Korean set from the identity plan: Korean CC on subtitle track 2, 12-14% coverage.

**Alice in Borderland** is a diagnostic only (GT8). Its SDH tags are sparse inside long diarized spans: 19 of 103 segments carry a label on E01, coverage is 12%, and no diarized speaker holds a majority of any lead's speech. Two characters recur on about 26 s between them. It scores 0 clean at every swept cell (aib rows in the export, tags `t8-diag-aib-ct*`).

## Baselines (tuning episodes, before the pick)

Live-action here is the provisional profile (pyannote at its shipped 0.6, `registry resolve` joint at 0.6 with `episode_cannot_link = false`); default is centroid at 0.6 per episode. Cells are clean / merges / fraction.

| Set | provisional live-action | default |
|---|---|---|
| bbt en | 4 / 0 / 1.00 pass | 4 / 0 / 1.00 pass |
| bbt de | 2 / 1 (Leonard + Sheldon) / 1.00 fail | 2 / 1 / 1.00 fail |
| bbt es | 1 / 3 (Howard, Leonard, Sheldon on one id) / 1.00 fail | 1 / 3 / 0.75 fail |
| bazinga-tbbt PAL / film | 4 / 0 / 1.00 pass, both | 4 / 0 / 1.00 pass, both |
| bazinga-office PAL | 3 / 0 / 1.00 pass | 2 / 0 / 0.67 fail |
| bazinga-office film | 3 / 0 / 0.75 fail (4 recurring) | 3 / 0 / 0.75 fail |
| kaze | 0 / 0 / 0.00 fail (りん 0.79) | 0 / 0 / 0.00 fail |
| aib (diagnostic) | 0 / 1 / 0.00 fail | 0 / 0 / 0.00 fail |

The de and es merges happen in diarization: one pyannote speaker holds 93-97% of both Leonard's and Sheldon's labelled speech on both dubs, while English splits them (78% and 82% on two speakers). Nearest-centroid accuracy on the same labelled spans is 96-99%, because classifying with known labels is easier than clustering without them. Every Office failure is Dwight's S01E01 landing on a different id from his E02/E03 id, a split, never a merge.

## Sweep and pick

The sweep crosses three pyannote `clustering_threshold` values (0.6, community-1's shipped value; 0.5; 0.4) with three matchers (centroid; joint with `episode_cannot_link` off; joint with it on) and six matcher thresholds (0.50 to 0.75), on tuning episodes, WeSpeaker throughout: 540 runs on diarization segmentation (tags `t8-sweep-ct*`, and `t8-diag-aib-ct*` for aib), each a ledger row. Each clustering value re-diarized every gate set's tuning episodes; GT6 puts the option in the diarize cache key, and a staleness check confirmed that only diarize re-ran.

The rule (Task 9's): most clean characters summed over the gate sets, then fewest merges, then mean fraction. The gate sets are bbt-en, bbt-de, bbt-es, kaze, grid, bazinga-tbbt-pal and bazinga-office-pal. BBT counts three times, once per audio track, as Fairy Tail's two tracks did in Task 9. Film conditions are reported alongside and aib is diagnostic; neither enters the sums.

| clustering | matcher, threshold | clean | merges | mean fraction |
|---|---|---|---|---|
| 0.4 | joint, cannot-link on, 0.50-0.60 | 18 | 1 | 0.857 |
| **0.5** | **centroid 0.50-0.60; joint, cannot-link on, 0.50-0.60** | **17** | **1** | **0.714** |
| 0.4 | centroid 0.50-0.55 | 17 | 2 | 0.857 |
| 0.6 | joint, cannot-link off, 0.60 (the provisional setting) | 16 | 4 | 0.857 |

**GT12 overrides the literal rule.** By the rule, ct 0.4 wins on one clean character. The controller treats ct 0.4 against ct 0.5 as a tie and picks ct 0.5. The whole difference between 18 and 17 is りん on Kaze crossing the 80% bar (0.79 at 0.6, 0.74-0.80 at 0.5, 0.84 at 0.4) on the thinnest gate set, with one recurring character and labels that need `--force`. And 0.4 is outside the range VBx was tuned over (its declared search space is `Uniform(0.5, 0.8)`); pyannote accepts the value, but the model was never tuned at it. A tie-break that rests on one character's share on that set shouldn't ship an untuned value. If ct 0.4 is right, held-out data will show it and a later plan can switch; this one doesn't re-tune.

**At ct 0.5**, centroid and joint with cannot-link on tie at 17 / 1 / 0.714 for every matcher threshold from 0.50 to 0.60, and both drop to 16 at 0.65 (The Office PAL loses Dwight). Ties go to the simpler matcher, centroid; 0.6 is the top of the plateau and today's value, as Task 9 chose. Joint with cannot-link off is worse at ct 0.5 (15 / 4 at 0.60): it re-merges the Spanish leads that the lower clustering value split.

**The pick:** pyannote `clustering_threshold = 0.5` in `[diarize]`, the centroid matcher at 0.6 in the characters stage. `[resolve]` keeps the tied joint cell (0.6, `episode_cannot_link = true`) for anyone who runs `fukikae registry resolve` on a live-action workspace; it was `false` before.

Per set at ct 0.5 with centroid at 0.6, against the provisional setting:

| Set | ct 0.5, centroid 0.6 | provisional (ct 0.6, joint, cannot-link off, 0.6) |
|---|---|---|
| bbt en | 4 / 0 / 1.00 | 4 / 0 / 1.00 |
| bbt de | 2 / 1 / 0.75 | 2 / 1 / 1.00 |
| bbt es | 3 / 0 / 0.75 | 1 / 3 / 1.00 |
| grid | 1 / 0 / 0.50 | 2 / 0 / 1.00 |
| bazinga-tbbt PAL | 4 / 0 / 1.00 | 4 / 0 / 1.00 |
| bazinga-office PAL | 3 / 0 / 1.00 | 3 / 0 / 1.00 |
| kaze | 0 / 0 / 0.00 | 0 / 0 / 0.00 |
| bazinga-tbbt film (alongside) | 4 / 0 / 1.00 | 4 / 0 / 1.00 |
| bazinga-office film (alongside) | 4 / 0 / 1.00 | 3 / 0 / 0.75 |

**Spanish is fixed from ct 0.5.** From 0.5 down, pyannote gives Leonard and Sheldon separate speakers in both tuning episodes (E01 0.89/0.70, E02 0.76/0.81 on their modal speakers), and es goes from 1 clean / 3 merges to 3 clean / 0 merges under centroid. Howard then fails on share (0.53).

**German's leads stay unseparated at every setting.** E01 keeps both leads on one pyannote speaker even at 0.4 (0.90/0.96), and E02 splits only partly; every matcher cell ends with Leonard and Sheldon on one id; the best de cell is 2 / 1 at all three clustering values. A diagnostic pass with subtitle segmentation (`--segmentation subtitles`, which takes pyannote out of the loop; tag `t8-diag-subtitles`) shows why: on German line segments the two leads land on one id up to matcher threshold 0.65 and separate only at 0.70, where every character's lines scatter and nothing passes. English line segments never put the two on one id. In WeSpeaker space at line length, the German voices for Leonard and Sheldon sit closer than any useful matcher threshold. The merge has a cause beyond diarization, and no swept setting fixes it.

**Grid** drops to 1 / 0 / 0.50 in every ct 0.5 cell and recovers at 0.4; with two recurring characters on 12% coverage, that is one character (새벽 at 0.78).

**Language variants: none ship.** A variant needs 5 points of fraction on its language's sets with no fewer clean characters. en, es, ja and ko have no better ct 0.5 cell than the pick. German's literally does: joint with cannot-link off at 0.60 gives 2 / 1 / 1.00 against the pick's 2 / 1 / 0.75. That gain is Sheldon's E02 speech joining the merged Leonard + Sheldon id instead of staying unassigned; the merge and the failing verdict are the same. It is also a joint setting, which only `registry resolve` runs: a `[resolve.lang.de]` table would never be read on the centroid path the pick ships. I recorded it and shipped no variant.

## Held-out verdicts

Each held-out episode ran once after `6bb10c2`: in a hard-linked copy of the ct 0.5 sweep workspace with the workspace's `[diarize]` override removed, which leaves every setting to the committed profile. The tuning episodes' diarizations were fresh under the profile's key, which confirms the profile resolves to the swept setting. The held-out episode was diarized at 0.5, then `fukikae run <ws> <ep> --to characters` on the tuning episodes in order and the held-out episode last, and `fukikae eval identity --from-assignments --tune ... --holdout ...` (tags `t8-holdout`, and `t8-holdout-pal` / `-film` for Bazinga). Grid's separate and asr manifest keys predate GT6; its diarize and characters stages ran directly on the existing vocals (`scripts/ground-truth/stage_direct.py`) to keep Task 9's audio. The default profile was held out once on the same episodes in Task 7's `-default` copies, for comparison. Cells are clean / merges / fraction.

| Language | Set, held-out episode | live-action (pick) | default | verdict (pick) |
|---|---|---|---|---|
| en | bbt S01E03 | 4 / 0 / 1.00 | 4 / 0 / 1.00 | **pass** |
| de | bbt S01E03 | 3 / 1 (Leonard + Sheldon) / 0.80 | 3 / 1 / 1.00 | **fail** |
| es | bbt S01E03 | 4 / 0 / 1.00 | 1 / 1 / 0.50 | **pass** |
| en | bazinga-tbbt PAL S01E05 | 4 / 0 / 0.67 (6 recurring) | 4 / 0 / 0.67 | **fail** |
| en | bazinga-tbbt film S01E05 (alongside) | 4 / 0 / 0.67 (6 recurring) | 4 / 0 / 0.67 | fail |
| en | bazinga-office PAL S01E05 | 3 / 0 / 1.00 | 3 / 0 / 1.00 | **pass** |
| en | bazinga-office film S01E05 (alongside) | 3 / 0 / 1.00 | 3 / 0 / 1.00 | pass |
| ko | grid S01E05 | 4 / 0 / 0.80 (5 recurring) | not run | **fail** |
| ja | kaze S01E11 | pending | pending | pending |
| ja | aib | not held out (GT8) | | |

- **English (BBT)** passes: every lead holds 92-100% on its tuning id, coverage 47%.
- **Spanish** passes, 4 of 4 with no merge (Leonard 0.93, Sheldon 0.91, Howard 1.00). The default profile on the same episode fails at 1 / 1 / 0.50: at ct 0.6 its tuning registry merged the leads, and E03's Leonard and Howard land on no tuning id.
- **German** fails. The tuning registry holds Leonard and Sheldon on one id (id 1); on E03 pyannote at 0.5 does split them, Leonard stays on id 1 (0.89) and Sheldon's 152 s go to a new actor, which leaves 0.01 of his speech on a tuning id. The default profile counts 3 / 1 / 1.00 on the same episode, because at ct 0.6 both leads stay on the merged id; its verdict fails on the same merge.
- **Bazinga TBBT** fails on PAL and film alike, and the default profile fails identically. Six characters recur against four in tuning. Howard keeps only 0.52 (PAL) and 0.49 (film) of his E05 speech on his id, and Raj has no tuning id at all (2 s in E01, too little to recur in tuning) and scores 0.00. Leslie, who has 22 s in tuning, holds 0.98.
- **Bazinga Office** passes on both conditions: Michael, Jim and Pam recur on E05 (Dwight is under 5 s there), each at 0.88 or above.
- **Korean (Grid E05)**, scored because the pick differs from the provisional setting: 4 clean, no merge, 0.80. 종이 fails at 0.00; she has 2.3 s and 3.5 s in the tuning pair, too little for a tuning id, and 8.9 s on E05. The provisional profile also scored 0.80 on E05, but with a 새하 + 어진 merge.
- **Japanese (Kaze E11)** is pending. The tuning registry is already built in `kaze-pick` (E01-E03 at the committed profile; no held-out stage has run), and the follow-up copies E11 through the read-only volume, checks caption onsets as Task 7b did, diarizes it, and runs `scripts/ground-truth/holdout_run.sh kaze-pick` (E11 only, `KAZE_EPS=S01E11`) and `holdout_score.sh` once. With one recurring character in tuning, the E11 verdict can't show over-merge (see Limits).

Summed over the PAL gate sets that have a held-out score (bbt en/de/es, both Bazinga shows, grid), the pick gives 22 clean characters with 1 merge; the default profile, without grid, gives 15 clean with 2 merges where the pick gives 18 with 1. The one remaining merge is German's leads.

## Clip-set embedder results

`fukikae eval embedders` scores R8's nearest-centroid accuracy (speech-weighted) on `noncegeek/cantonese-drama-voice-fine-grained`. 人民的名义 (Mandarin, `renmin`) splits by episode, with no episode on both sides; 15 of its 23 characters appear in two or more episodes and are scored. 神雕侠侣 (Cantonese, `shendiao`) has no episode numbers and splits by emotion folder (train 1732, test 1273 clips); spk03 holds 56% of its clips, which puts chance at about 0.56.

| Embedder | renmin (zh) | shendiao (yue) |
|---|---|---|
| eres2netv2 | 0.957 | 0.994 |
| wespeaker | 0.948 | 0.972 |
| campplus (zh_en model) | 0.944 | 0.992 |
| titanet | 0.937 | 0.976 |
| ecapa | 0.930 | 0.972 |

The best non-WeSpeaker embedder, eres2netv2, leads by 0.9 points on renmin and 2.2 on shendiao, under the 5-point threshold. R10 (WeSpeaker for every genre) stands.

**CAM++'s earlier numbers are superseded.** The `campplus` default used to be the English VoxCeleb model (`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`), and every CAM++ number before 2026-11-08 came from it: 0.340 on renmin, and 21-36% on the anime and Grid cells in `docs/results/2026-11-05-embedders.md`. The default is now `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (GT5), which gives 0.944 on renmin and 65-87% on those anime and Grid cells, still under WeSpeaker by 2-15 points on five of six; R9's drop of CAM++ stands on the new numbers. The ledger's `t6-fix1` row (renmin campplus 0.34) reused English-model embeddings under the new model's name before the cache keyed on model identity, and is invalid; it is left out of the export. The `t6` shendiao rows used an earlier split and aren't comparable with `t6-fix1b`.

## Limits

- **Kaze can't detect over-merge.** It has one recurring character (りん) in its tuning episodes, and the bar measures each character's share on its own modal id; with nobody to merge with, a profile that put the whole cast on one id would still pass Kaze. 直美's nearest-centroid accuracy is 7% (her speech classifies as りん); expect a merge if a later episode makes her recurring. Kaze's whole contribution to the pick is one character's share (0.79 to 0.84), which is why GT12 treats it as noise.
- **BBT is one show counted in three languages**, three of the seven gate sets, and its labels are English-timed: the dubs are scored on the English subtitle timeline, which auto-label's overlap (80-86%) shows fits them. The English labels come from OCR'd image subtitles aligned to fan transcripts; the Bazinga check puts that at 98.2% speaker agreement on the 73% of dialogue both sides label.
- **German stays merged.** No clustering value or matcher in the sweep separates Leonard and Sheldon on the German dub; the subtitle-segmentation diagnostic says the two dubbing voices sit too close in WeSpeaker space at line length. Fixing it needs something this plan didn't sweep (another embedder, or visual cues).
- **SDH names skew towards off-screen speech.** Netflix and Korean CC tag a speaker when the picture doesn't show who is talking, and labelled speech over-represents voice-over and off-screen lines (aib, grid). Colour covers only Kaze's main cast, and unmapped (white) lines are out of the bar.
- **Thin sets.** Kaze (1 recurring), Grid (2) and aib (2) are flagged thin; one character decides each. Held-out recurring sets are larger than tuning ones (TBBT 6 against 4, Grid 5 against 2), and a character with no tuning id fails by construction (Raj, 종이).
- **Bazinga PAL against film.** The Dwight split that separates the conditions in places (kept on PAL and split on film under the provisional profile) is diarization variance, not pitch: no TBBT number moves between conditions, and the one Office difference goes the opposite way from a pitch penalty. The 5 s bar is wall-clock time, and the PAL audio, 4% shorter, can drop a character below it (Pam in Office E02).
- **Kaze timing.** The retimed captions lead speech by 0.24-0.31 s; a +0.25 s shift changes no verdict, but the leading edge of each label is early.
- **Alice in Borderland** is out of the evidence (GT8); Japanese live action rests on Kaze alone until E11 is scored.

## Reproducing

The drivers are in `scripts/ground-truth/` (no media or results): `setup.py` makes the hard-linked workspace copies per clustering value, `diar.sh` and `diar_direct.py` re-diarize, `stale.py` checks which stages a copy would re-run, `sweep.py` runs the matcher sweep through the CLI, and `tab.py` renders the tables from the per-run JSON. The held-out steps are `holdout_setup.py`, `holdout_stale.py`, `holdout_run.sh`, `stage_direct.py` and `holdout_score.sh`. They read `FUKIKAE_WS` (default `~/fukikae-ws`); the shell scripts find the repository from their own path. The repository copy of `tab.py` reproduces the sweep tables byte for byte from the workspace's JSON. The exports hold tags `t6`, `t6-fix1b`, `t7-baseline`, `t7c-baseline-{pal,film}`, `t8-sweep-ct{0.4,0.5,0.6}[-pal|-film]`, `t8-diag-aib-ct*`, `t8-diag-subtitles` and `t8-holdout[-pal|-film]`, for sets `bbt`, `aib`, `kaze`, `grid`, `renmin`, `shendiao`, `bazinga-tbbt` and `bazinga-office`; the matrix is `fukikae eval report --stage identity --tag t8-holdout --tag t8-holdout-pal --tag t8-holdout-film --format md`.
