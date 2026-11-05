# Cross-episode character registry: Fairy Tail: 100 Years Quest S01E12-E13

_Run 2026-11-05 on ml-01 (RTX 3090, index 0), branch feat/foundation at 291171a._

## Setup

- Episodes: Fairy Tail: 100 Years Quest S01E12 "A Card in the Hand" and S01E13 "Scarlet Showdown", one WEBDL-1080p release with Japanese and English audio; the Japanese track was used. S01E14 was run too and resolved against the same registry (7 matched, 3 review, 1 new) but isn't scored here.
- Ground truth: the release's Crunchyroll English ASS names the speaker on every Dialogue line (357 and 254 lines). `fukikae eval auto-label` labels every clean span in `clusters.npz` (up to 50 per cluster) with the name whose lines cover at least 60% of the span, else `?`. Spelling variants were merged (`ERZA/I` with `ERZA`, `MAKAROV(WRAITH)` with `MAKAROV (WRAITH)`), and one alias was applied: `NABARU` to `NEBARU`, a typo seen only in E12 (3 lines) against NEBARU in all of E11-E14.
- Pipeline: Demucs htdemucs_ft separation, Qwen3-ASR (Japanese), pyannote community-1 diarization, ECAPA-TDNN embeddings (speechbrain, VoxCeleb), match threshold 0.6 (the placeholder default), 20 s new-character bar.
- Workspace: `~/fukikae-ws/fairytail` (`fukikae.toml`: `source_language = "ja"`, `[asr.lang.ja] backend = "qwen3_asr"`, `gpu_device = 0`; aliases in `labels-aliases.yaml`).

Commands:

```
fukikae run ~/fukikae-ws/fairytail S01E12 --to characters      # then S01E13, S01E14
fukikae eval auto-label ~/fukikae-ws/fairytail --episodes S01E12,S01E13 --out ~/fukikae-ws/fairytail/labels-auto --report
fukikae eval speakers ~/fukikae-ws/fairytail --labels ~/fukikae-ws/fairytail/labels-auto/labels.yaml
```

## Result

The spec's metric (a recurring character is consistent only when every one of its clips maps to one character id, and any two names sharing an id count as a merge): 12 recurring characters, 2 consistent, fraction **0.17**, 63 merged pairs, verdict **rework** (bar: pass ≥ 0.90 with no merges, rework < 0.75).

That metric fails a character on a single stray clip, so it can't tell an impure diarization cluster from a wrong cross-episode match. Weighting by speech and taking each character's modal id separates the two:

| Character | Speech | Share on modal id | E12 on that id | E13 on that id |
|---|---|---|---|---|
| Lucy | 70.2 s | 84% | 100% | 61% |
| Touka | 57.9 s | 15% | 0% | 16% |
| Laxus | 56.6 s | 89% | 94% | 84% |
| Natsu | 53.8 s | 72% | 88% | 37% |
| Erza | 49.3 s | 65% | 0% | 100% |
| Cana | 43.8 s | 56% | 90% | 0% |
| Happy | 41.6 s | 95% | 100% | 94% |
| Kiria | 35.5 s | 75% | 100% | 0% |
| Juvia | 31.8 s | 56% | 70% | 38% |
| Gray | 25.9 s | 100% | 100% | 100% |
| Charle | 14.9 s | 81% | 100% | 0% |
| Wendy | 9.2 s | 100% | 100% | 100% |

Speech-weighted, 70% of recurring characters' speech sits on their modal id. Character-id purity, the reverse view:

| Id | Speech | Dominant name | Purity | Other names |
|---|---|---|---|---|
| 4 | 101.5 s | Lucy | 58% | Cana 19%, Juvia 18%, Touka 5% |
| 2 | 68.8 s | Happy | 57% | Charle 17%, Wendy 13%, Natsu 7% |
| 7 | 50.6 s | Laxus | 100% | |
| 8 | 42.5 s | Wraith | 84% | Jellal 16% |
| 9 | 40.7 s | Natsu | 95% | Young Bob 5% |
| 11 | 32.4 s | Gray | 80% | Young Goldmine 11%, Townspeople B 9% |
| 12 | 31.9 s | Erza | 100% | |
| 1 | 29.9 s | Young Makarov | 43% | Nebaru 39%, Young Yajima 7% |
| 3 | 27.8 s | Touka | 31% | Laxus 22%, Natsu 11%, Juvia 9% |
| 5 | 27.5 s | Cana | 89% | Juvia 11% |
| 6 | 26.7 s | Kiria | 100% | |
| 10 | 25.4 s | Makarov (Wraith) | 66% | Young Yajima 27%, Makarov 8% |

Within one episode, diarization spreads characters over several clusters (across both episodes: Natsu 7, Touka 5, Juvia 5, Lucy 4, Laxus 4) and some clusters mix voices (E13's Kiria cluster is 38% Kiria, its Cana cluster 46% Cana).

## Reading

Three failure modes, in order of weight:

1. **Similar voices collapse.** Id 2 holds Happy, Charle and Wendy (two cat characters voiced by women and a young girl); id 4 holds Lucy, Cana and Juvia (three young women). ECAPA-TDNN is trained on VoxCeleb's conversational adult speech and doesn't separate stylized young anime voices well. Distinctive voices resolve cleanly: Laxus, Gray, Happy, Erza within E13, Natsu within E12.
2. **Some characters split across episodes.** Erza's E12 speech never reaches her E13 id; Cana, Kiria and Charle get new ids in E13. Either their E12 centroids were built from mixed clusters, or the 0.6 threshold is too strict for their voices; this run doesn't separate the two.
3. **Diarization over-splits and mixes** on dense, scored anime dialogue, which feeds both of the above.

Per the spec's bar this is rework before milestone 5. The ground truth is now automatic, so the rework can be measured directly: an embedder comparison on the labeled spans alone (no diarization) would show whether a stronger embedder (CAM++/ERes2Net from 3D-Speaker, WavLM-based models, TitaNet) separates these voices; subtitle-line segmentation, when a release has named subtitles, would show how much of the failure is pyannote's. A threshold sweep only makes sense after those.

Chimera (2021) S01E01-E02 was also run through characters (9 actors) but has no speaker ground truth; its labeling clips are in `~/fukikae-ws/chimera/labels-video` if a hand-labeled comparison is ever wanted.
