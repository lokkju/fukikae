# Segmentation and matching variants

_Run 2026-11-05 and 2026-11-06 on ml-01 (RTX 3090, index 0), branch feat/speaker-identity. Tuning pairs only: no `--holdout` was passed and nothing here was chosen by looking at a held-out episode. Match threshold 0.6 for centroid, knn and multi-centroid (the threshold sweep is Task 9)._

## What was compared

On each tuning pair (fairytail E12 then E13, `ass`; fairytail-en E12 then E13, `ass`; grid E02 then E04, `sdh`, `--subtitle-track 2`), every combination of:

- **Segmentation** (8): `diarization` (pyannote community-1 as shipped: clustering threshold 0.6, `embedding_exclude_overlap` on), five tuned pyannote variants `pyannote-t<threshold>-eo<0|1>` (threshold 0.5, 0.6, 0.7 times exclude_overlap on/off, less the shipped cell), `subtitles` (the ground-truth lines) and `asr` (transcript sentences split where 1.5 s word windows differ by cosine distance above 0.8).
- **Embedder** (Ruling R9): ECAPA, WeSpeaker and ERes2NetV2 on the anime sets; WeSpeaker on grid, plus ECAPA there for the baseline.
- **Matcher** (12 settings): `centroid`; `knn` k 3, 5, 9; `multi-centroid` max_centroids 2, 3; `lda` dims 16, 32, all; `joint` threshold 0.5, 0.6, 0.7.

That is 764 harness runs, all scored with `eval identity` semantics (recurring at 5 s per episode, Ruling R6). Raw output: `~/fukikae-ws/segmatch/results.jsonl` (one line per run, with per-character shares), drivers `sweep.py`, `diarize_variants.py`, `asr_split.py`, `tables.py` beside it.

Pyannote variants were re-diarized into `~/fukikae-ws/<set>-pyannote-<variant>/`, which hold symlinks to the originals' `fukikae.toml`, aliases, `vocals_16k.wav` and `transcript.json` and their own `04-diarize/turns.json`; the original workspaces were only read. Community-1 clusters with VBx, which has no `min_cluster_size` (the 3.x agglomerative clustering's parameter); the backend refuses that option on a VBx pipeline, so the grid is threshold times exclude_overlap only.

The ASR split threshold came from the tuning episodes: adjacent 1.5 s ECAPA windows of one speaker on fairytail E12 sit at a median cosine distance of 0.62 (95th percentile 0.78), so the planned 0.5 would have split 84% of same-speaker window pairs; 0.8 splits 4% and still catches 75% of the 12 in-sentence speaker changes. Word windows use the Task 5 genre winner: ECAPA on anime (the default Task 5 kept), WeSpeaker on grid.

## How to read the tables

The bar's fraction rewards collapse: an id that swallows several recurring characters gives each of them a high share on "its" id, and they all pass. So every table carries the merge list, and the ranking column is **clean**: recurring characters that pass and appear in no merge. A run passes the bar only with a fraction of at least 0.90 and no merges. NC is Task 5's nearest-centroid accuracy; it depends on segmentation and embedder only. Different segmentations label different speech, so the recurring count moves (9 on fairytail diarization, 10 or 11 on subtitles and asr); compare clean counts within a set with that in mind.

## Best combinations per genre

### Anime, Japanese audio (fairytail)

| Configuration | Fraction | Pass/rec | Clean | Merges | Share | Strict | Coverage | NC |
|---|---|---|---|---|---|---|---|---|
| baseline: diarization, ECAPA, centroid | 0.33 | 3/9 | 2 | JUVIA+LUCY | 70% | 0.17 | 32% | 68% |
| diarization, WeSpeaker, centroid | 0.44 | 4/9 | 4 | JUVIA+LUCY | 72% | 0.25 | 36% | 70% |
| pyannote-t0.6-eo0, WeSpeaker, lda (16) | 0.44 | 4/9 | 4 | JUVIA+LUCY | 76% | 0.25 | 37% | 71% |
| pyannote-t0.5-eo0, ECAPA, centroid | 0.67 | 6/9 | 4 | JUVIA+LUCY; JUVIA+TOUKA; LUCY+TOUKA | 84% | 0.33 | 35% | 69% |
| pyannote-t0.5-eo0, ECAPA, joint (0.6) | 0.89 | 8/9 | 5 | 6 pairs (CANA, JUVIA, LUCY, TOUKA on one id) | 91% | 0.42 | 39% | 69% |

Nothing reaches the bar, and nothing removes the JUVIA+LUCY merge. Every configuration that raises the fraction above 0.44 does it by putting TOUKA, LUCY and JUVIA (and with joint, CANA) on one id. The one gain that adds no merge is the embedder (WeSpeaker, from Task 5), worth two clean characters; no matcher or segmentation adds a clean character on top of it without adding merges.

### Anime, English dub (fairytail-en)

| Configuration | Fraction | Pass/rec | Clean | Merges | Share | Strict | Coverage | NC |
|---|---|---|---|---|---|---|---|---|
| baseline: diarization, ECAPA, centroid | 0.70 | 7/10 | 6 | CANA+JUVIA | 72% | 0.08 | 43% | 72% |
| diarization, ECAPA, joint (0.7) | 0.70 | 7/10 | 7 | none | 77% | 0.08 | 50% | 72% |
| pyannote-t0.6-eo0, WeSpeaker, joint (0.7) | 0.70 | 7/10 | 7 | none | 83% | 0.00 | 50% | 71% |
| pyannote-t0.6-eo0, WeSpeaker, joint (0.6) | 0.90 | 9/10 | 8 | CANA+JUVIA | 89% | 0.00 | 51% | 71% |
| subtitles, WeSpeaker, joint (0.5) | 0.73 | 8/11 | 8 | none | 81% | 0.00 | 86% | 73% |

The dub is where matching helps. `joint` at 0.7 removes the CANA+JUVIA merge at the same fraction; at 0.6 on the exclude_overlap-off diarization it reaches 0.90 with the merge back. The best merge-free run is subtitles segmentation with WeSpeaker and joint at 0.5: 8 of 11 recurring characters clean, 81% of speech on the modal id, coverage 86%. Subtitle segmentation is an oracle segmentation (it cuts at the ground truth's own line boundaries) and is not available at dubbing time, so it bounds what a better segmenter could give; it is not a candidate profile.

### Live action, Korean (grid)

| Configuration | Fraction | Pass/rec | Clean | Merges | Share | Strict | Coverage | NC |
|---|---|---|---|---|---|---|---|---|
| baseline: diarization, ECAPA, centroid | 0.50 | 1/2 | 1 | none | 73% | 0.17 | 12% | 81% |
| diarization, WeSpeaker, centroid | 0.50 | 1/2 | 1 | none | 73% | 0.17 | 12% | 96% |
| diarization, WeSpeaker, joint (0.6) | 1.00 | 2/2 | 2 | none | 81% | 0.17 | 14% | 96% |
| pyannote-t0.6-eo0, ECAPA, joint (0.5) | 1.00 | 2/2 | 2 | none | 88% | 0.33 | 15% | 81% |

`joint` passes both recurring characters (새벽 goes from 67% to 88% on its id) at every threshold and on every diarization variant, with any embedder. Two characters on 12% coverage: one character is the whole difference.

## Per-character effect on the known confusions

Share of each character's labeled speech on its modal id (and that id); the same id across characters is a merge.

fairytail (ja):

| Configuration | TOUKA | LUCY | JUVIA | CANA | HAPPY | ERZA | Merges |
|---|---|---|---|---|---|---|---|
| diarization, ecapa, centroid | 15% (id 3) | 84% (id 4) | 56% (id 4) | 56% (id 5) | 95% (id 2) | 65% (id 12) | JUVIA+LUCY |
| diarization, wespeaker, centroid | 15% (id 3) | 67% (id 4) | 40% (id 4) | 94% (id 5) | 95% (id 2) | 65% (id 12) | JUVIA+LUCY |
| diarization, wespeaker, lda (dims=16) | 15% (id 3) | 67% (id 4) | 40% (id 4) | 94% (id 5) | 95% (id 2) | 65% (id 12) | JUVIA+LUCY |
| diarization, ecapa, multi-centroid (2) | 15% (id 3) | 84% (id 4) | 56% (id 4) | 56% (id 5) | 95% (id 2) | 65% (id 12) | JUVIA+LUCY |
| pyannote-t0.5-eo0, ecapa, centroid | 63% (id 3) | 86% (id 3) | 82% (id 3) | 96% (id 10) | 100% (id 1) | 63% (id 11) | JUVIA+LUCY; JUVIA+TOUKA; LUCY+TOUKA |
| pyannote-t0.5-eo0, ecapa, knn (k=9) | 63% (id 3) | 86% (id 3) | 82% (id 3) | 96% (id 10) | 100% (id 1) | 63% (id 11) | JUVIA+LUCY; JUVIA+TOUKA; LUCY+TOUKA |
| diarization, ecapa, joint (0.7) | 80% (id 4) | 100% (id 4) | 92% (id 4) | 100% (id 4) | 95% (id 2) | 97% (id 7) | all six CANA/JUVIA/LUCY/TOUKA pairs |
| pyannote-t0.5-eo0, ecapa, joint (0.7) | 63% (id 3) | 91% (id 3) | 82% (id 3) | 96% (id 12) | 100% (id 1) | 94% (id 8) | JUVIA+LUCY; JUVIA+TOUKA; LUCY+TOUKA |
| asr, wespeaker, joint (0.5) | 62% (id 4) | 85% (id 4) | 82% (id 4) | 94% (id 4) | 97% (id 2) | 80% (id 9) | all six CANA/JUVIA/LUCY/TOUKA pairs |

TOUKA rises from 15% only by joining LUCY's id; no run gives TOUKA an id of her own with more than 15%. On the Japanese track these four voices are one voice to every embedder, matcher and segmentation tried. ERZA does gain on her own id under joint (65% to 94-97%), a real improvement hidden inside runs that fail on merges.

fairytail-en:

| Configuration | TOUKA | LUCY | JUVIA | CANA | HAPPY | ERZA | Merges |
|---|---|---|---|---|---|---|---|
| diarization, ecapa, centroid | 0% (no id) | 100% (id 11) | 25% (id 10) | 95% (id 10) | 89% (id 2) | 55% (id 13) | CANA+JUVIA |
| pyannote-t0.6-eo0, wespeaker, lda (16) | 90% (id 13) | 97% (id 11) | 26% (id 10) | 95% (id 10) | 90% (id 2) | 56% (id 14) | CANA+JUVIA |
| diarization, ecapa, joint (0.7) | 46% (id 15) | 100% (id 13) | 25% (id 16) | 52% (id 12) | 96% (id 2) | 81% (id 7) | none |
| pyannote-t0.6-eo0, wespeaker, joint (0.6) | 90% (id 15) | 97% (id 13) | 26% (id 12) | 95% (id 12) | 90% (id 2) | 91% (id 7) | CANA+JUVIA |
| pyannote-t0.6-eo0, wespeaker, joint (0.7) | 90% (id 15) | 97% (id 13) | 26% (id 16) | 54% (id 12) | 90% (id 2) | 91% (id 7) | none |
| subtitles, wespeaker, joint (0.5) | 74% (id 12) | 88% (id 13) | 87% (id 11) | 63% (id 14) | 80% (id 2) | 82% (id 7) | none |

On the dub, TOUKA is fixable: the exclude_overlap-off diarization (t0.6-eo0) gives her 90% on an id of her own with WeSpeaker under lda or joint, where the baseline gave her no id at all. JUVIA stays at 25-26% with any diarization segmentation, which means pyannote itself splits her across clusters; only subtitle segmentation lifts her (87%). Breaking CANA+JUVIA (joint at 0.7) costs CANA (95% to 52-54%).

## Matchers, one by one

- **knn**: no gain at threshold 0.6 on any set. With the local-centroid score it lands on centroid's decisions in most cells; k 3/5/9 move single characters back and forth.
- **multi-centroid**: identical to centroid or within one character everywhere. With two tuning episodes an actor has at most two clusters' rows, so the extra centroids have little to model.
- **lda**: fit on E12's labels, applied to E13. No clean gain on fairytail; on fairytail-en it lifts speech share (72% to 84% on t0.6-eo0) and TOUKA, but not the fraction. dims 16, 32 and all gave the same result in most cells (the E12 labels give few classes).
- **joint**: the only matcher that changes outcomes. It merges more freely than per-episode matching (one clustering pass, centroid linkage); that fixes characters the registry splits (ERZA, 새벽, TOUKA on the dub) and produces the mega-merges on the Japanese track. Threshold 0.7 is merge-free on the dub and grid; 0.5 and 0.6 merge on both anime sets. ERes2NetV2's compressed similarity range puts every character on one id under joint at 0.5-0.6 (36 to 55 merge pairs), so joint with ERes2NetV2 is unusable at these thresholds.

## Segmentations, one by one

- **Pyannote tuning**: under centroid no variant adds a clean character. On fairytail, t0.5-eo0 lifts the fraction from 0.33 to 0.67 and share from 70% to 84% (ECAPA) only by putting TOUKA on LUCY and JUVIA's id; on the dub every variant lowers the centroid fraction (0.70 to 0.56-0.60); on grid share rises a few points (73% to 75-78%) with the same 1/2. The one useful effect is exclude_overlap off at the shipped threshold (t0.6-eo0) under lda or joint on the dub: TOUKA gets an id of her own at 90%. The clustering threshold (VBx's AHC initialization) moves cluster counts by a few and scores by one character.
- **Subtitles and asr under per-episode matchers**: near zero (0.00 fraction, coverage 2-33%) because `group_segments` groups unclustered segments with *average* linkage at distance 0.4, and the mean pairwise cosine of short single-segment embeddings is far below 0.6: almost every segment stays its own group, below `min_new_seconds`, and becomes an extra. Under joint (centroid linkage) the same segments give the best merge-free dub run. The problem is the grouping step, not the segments.
- **asr**: under joint it performs like diarization (dub 7/10 clean, merge-free at 0.5; NC 82%, the highest on the dub), without needing pyannote's clusters. Japanese one-character "words" from Qwen3 make the windows coarse, and anime sentences are short, so most sentences are a single window and are never split.

## Recommendation for Tasks 7 to 9

- **Anime**: carry WeSpeaker, both the shipped pyannote and `exclude_overlap = false` at threshold 0.6 (the latter only pays off with joint or lda, on the dub), and two matchers: `centroid` (the pipeline today) and `joint` at 0.7 (`seed = clusters`). Drop knn and multi-centroid (no gain); keep lda only as an option, not a profile default (it needs labeled episodes, and it gained share but no characters). Drop ERes2NetV2 with joint. The Japanese female-voice collapse (TOUKA, LUCY, JUVIA, CANA) is not fixed by any segmentation or matcher here; Task 8's adaptation is the remaining lever, and the bar on the Japanese track looks out of reach without it.
- **Live action**: carry WeSpeaker, shipped pyannote (variants change nothing on grid), and `joint` at 0.6 or 0.7 alongside centroid for Task 9's threshold sweep. Grid's two recurring characters can't distinguish more than that.
- **Task 7 (pipeline matcher)**: `joint` has no registry state and needs every episode at once; in the pipeline it fits a batch re-resolve (`registry resolve --all`) better than the per-episode characters stage. If it goes in, the cannot-link and centroid linkage are already in `registry/matchers.py`.
- **Fix before Task 9**: `group_segments` should use centroid linkage (as `joint` does) for unclustered segments, or subtitle and asr segmentation can't be compared fairly under per-episode matchers. That changes no diarization result (diarization segments keep their clusters).

Each keep/drop above is a ruling candidate for the ledger with the numbers in this note.
