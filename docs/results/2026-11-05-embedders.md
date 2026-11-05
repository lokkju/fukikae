# Speaker embedders compared

_Run 2026-11-05 on ml-01 (RTX 3090, index 0), branch feat/speaker-identity. Tuning pairs only: no `--holdout` was passed and nothing here was chosen by looking at a held-out episode._

## What was compared

ECAPA-TDNN (the baseline) against five new backends and one fusion embedder, each behind its own entry point:

| Backend | Model | Dim | Extra | Runs on | Peak GPU |
|---|---|---|---|---|---|
| `ecapa` | speechbrain/spkrec-ecapa-voxceleb | 192 | pyannote | GPU | 1570 MiB (nvidia-smi delta, from Task 3) |
| `wespeaker` | pyannote/wespeaker-voxceleb-resnet34-LM (pyannote-audio 4.0.7) | 256 | pyannote | GPU | 40 MiB torch peak, 700 MiB declared (estimate with CUDA context) |
| `wavlm_sv` | microsoft/wavlm-base-plus-sv (WavLMForXVector) | 512 | wavlm | GPU | 452 MiB torch peak, 1100 MiB declared (estimate) |
| `campplus` | 3D-Speaker CAM++, `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` | 512 | sherpa | CPU | none |
| `eres2netv2` | 3D-Speaker ERes2NetV2, `3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx` | 192 | sherpa | CPU | none |
| `titanet` | NVIDIA TitaNet-L, `nemo_en_titanet_large.onnx` | 192 | sherpa | CPU | none |
| `fusion` | `members = ["ecapa", "wespeaker"]`: concatenated unit-norm rows, renormalized | 448 | pyannote | GPU | the larger member's |

Pins: `sherpa-onnx 2.0.1` (with `sherpa-onnx-core 2.0.1`) in the new `sherpa` extra; `torch>=2.8` and `transformers>=4.57.6,<5` in the new `wavlm` extra; the lock kept every existing version (torch, transformers and pyannote-audio did not move) and `uv sync --all-extras` resolves. NeMo is not installed: TitaNet-L runs from the ONNX export that sherpa-onnx publishes, so none of NeMo's pins are involved. The sherpa models download on first use (about 221 MB for the four files tried) from the sherpa-onnx `speaker-recongition-models` release into `~/.cache/fukikae/models`. The sherpa backends run on the CPU; "GPU card 0 only" held for everything that touches a GPU. Spans shorter than 0.5 s are wrap-padded to 0.5 s for all five new backends (ECAPA zero-pads to 0.1 s).

Command per cell (workspaces under `~/fukikae-ws/`, outputs in `~/fukikae-ws/embedders/<set>.<segmentation>.<embedder>.{txt,json}`): `fukikae eval identity <ws> --tune A,B --gt <ass|sdh> [--subtitle-track 2] --segmentation <diarization|subtitles> --embedder <name> --matcher centroid --threshold 0.6`. Sets: fairytail (ja, E12 then E13, `ass`), fairytail-en (E12, E13, `ass`), grid (E02, E04, `sdh`, `--subtitle-track 2`). The ECAPA rows on diarization reproduce the Task 4 baselines exactly.

## How to read the tables

Nearest-centroid accuracy (NC) is the primary number (Ruling R8): centroids from the first episode's labeled segments, the second episode's labeled segments classified, speech-weighted, no clustering. The majority fraction, passing/recurring, strict fraction, merges and speech share on the modal id come from the harness's registry simulation with the centroid matcher at 0.6 and are secondary.

Two cautions on the secondary columns. First, 0.6 is ECAPA's operating point. WavLM-SV and the CAM++ file have compressed cosine ranges (unrelated speakers score high), so at 0.6 they merge many characters into one id; their "fraction" on subtitles (wavlm_sv 0.55 to 1.00) comes from one id absorbing everything, with 24 or more merges listed, and is not a pass. Second, on the subtitles segmentation the centroid matcher leaves most segments unmatched for the ECAPA-like embedders (coverage 22 to 39%), which is why their fraction is 0.00 there; the column does not discriminate between them. The merge lists for those cells are collapsed to a pair count. Use NC for the comparison, and read the harness columns only on the diarization segmentation.

Grid has six characters labeled in both episodes on the diarization segmentation (85 spans in all) and two that recur at 5 s; a single character moves any Grid number by 15 to 50 points. Treat it as a cross-check.

## Results


#### fairytail, diarization

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 68% | 0.33 | 3/9 | 0.17 | JUVIA+LUCY | 70% | 32% |
| wespeaker | 70% | 0.44 | 4/9 | 0.25 | JUVIA+LUCY | 72% | 36% |
| wavlm_sv | 49% | 0.33 | 3/9 | 0.17 | JUVIA+LUCY; ERZA+KIRIA | 70% | 35% |
| campplus | 24% | 0.22 | 2/9 | 0.17 | JUVIA+LUCY; CANA+ERZA | 65% | 32% |
| eres2netv2 | 69% | 0.22 | 2/9 | 0.17 | JUVIA+LUCY; JUVIA+TOUKA; LUCY+TOUKA; ERZA+KIRIA | 76% | 33% |
| titanet | 67% | 0.33 | 3/9 | 0.17 | JUVIA+LUCY; ERZA+KIRIA | 70% | 35% |
| fusion | 71% | 0.33 | 3/9 | 0.17 | JUVIA+LUCY | 70% | 33% |

Per-character nearest-centroid accuracy (fairytail, diarization):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| TOUKA | 3% | 11% | 4% | 19% | 4% | 3% | 7% |
| HAPPY | 88% | 88% | 68% | 11% | 94% | 78% | 88% |
| ERZA | 92% | 92% | 74% | 26% | 100% | 92% | 92% |
| LUCY | 66% | 77% | 46% | 18% | 58% | 72% | 77% |
| LAXUS | 84% | 72% | 48% | 48% | 84% | 84% | 84% |
| NATSU | 89% | 78% | 34% | 11% | 89% | 79% | 78% |
| CANA | 89% | 100% | 78% | 28% | 100% | 89% | 100% |
| JUVIA | 100% | 100% | 65% | 41% | 100% | 100% | 100% |
| KIRIA | 100% | 100% | 78% | 0% | 100% | 100% | 100% |
| GRAY | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| CHARLE | 100% | 100% | 100% | 0% | 100% | 100% | 100% |
| WENDY | 100% | 100% | 0% | 100% | 0% | 100% | 100% |

#### fairytail, subtitles

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 64% | 0.00 | 0/11 | 0.00 | none | 18% | 22% |
| wespeaker | 63% | 0.00 | 0/11 | 0.00 | none | 28% | 33% |
| wavlm_sv | 43% | 0.55 | 6/11 | 0.08 | 24 pairs | 80% | 95% |
| campplus | 21% | 0.00 | 0/11 | 0.00 | 37 pairs | 42% | 73% |
| eres2netv2 | 71% | 0.00 | 0/11 | 0.00 | CHARLE+TOUKA | 53% | 55% |
| titanet | 60% | 0.00 | 0/11 | 0.00 | none | 28% | 30% |
| fusion | 63% | 0.00 | 0/11 | 0.00 | none | 22% | 25% |

Per-character nearest-centroid accuracy (fairytail, subtitles):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| TOUKA | 5% | 3% | 12% | 7% | 21% | 3% | 0% |
| ERZA | 80% | 86% | 61% | 29% | 88% | 80% | 83% |
| HAPPY | 83% | 86% | 54% | 6% | 86% | 71% | 83% |
| LAXUS | 74% | 74% | 54% | 34% | 74% | 74% | 74% |
| NATSU | 82% | 70% | 37% | 7% | 84% | 70% | 86% |
| LUCY | 90% | 94% | 27% | 46% | 91% | 81% | 90% |
| CANA | 81% | 81% | 69% | 22% | 88% | 93% | 81% |
| KIRIA | 70% | 68% | 52% | 49% | 91% | 75% | 70% |
| JUVIA | 100% | 100% | 75% | 53% | 100% | 100% | 100% |
| GRAY | 100% | 100% | 100% | 19% | 100% | 82% | 100% |
| CHARLE | 100% | 59% | 0% | 0% | 41% | 59% | 59% |
| WENDY | 61% | 100% | 0% | 0% | 100% | 61% | 100% |
| NEBARU | 100% | 100% | 0% | 0% | 100% | 100% | 100% |

#### fairytail-en, diarization

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 72% | 0.70 | 7/10 | 0.08 | CANA+JUVIA | 72% | 43% |
| wespeaker | 71% | 0.70 | 7/10 | 0.08 | CANA+JUVIA | 72% | 43% |
| wavlm_sv | 59% | 0.70 | 7/10 | 0.08 | CANA+JUVIA | 67% | 38% |
| campplus | 36% | 0.60 | 6/10 | 0.08 | ERZA+KIRIA; CANA+JUVIA | 70% | 39% |
| eres2netv2 | 71% | 0.70 | 7/10 | 0.08 | CANA+JUVIA | 67% | 38% |
| titanet | 78% | 0.60 | 6/10 | 0.15 | HAPPY+TOUKA; CANA+JUVIA | 73% | 43% |
| fusion | 72% | 0.70 | 7/10 | 0.08 | CANA+JUVIA | 72% | 43% |

Per-character nearest-centroid accuracy (fairytail-en, diarization):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| TOUKA | 3% | 5% | 3% | 0% | 5% | 34% | 5% |
| ERZA | 96% | 96% | 91% | 53% | 96% | 96% | 92% |
| HAPPY | 86% | 78% | 21% | 6% | 78% | 78% | 86% |
| LAXUS | 91% | 91% | 78% | 40% | 91% | 92% | 91% |
| CANA | 100% | 100% | 100% | 79% | 100% | 100% | 100% |
| LUCY | 100% | 100% | 100% | 61% | 100% | 100% | 100% |
| NATSU | 100% | 100% | 92% | 27% | 100% | 92% | 100% |
| KIRIA | 100% | 100% | 90% | 59% | 100% | 100% | 100% |
| JUVIA | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GRAY | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| CHARLE | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| WENDY | 100% | 100% | 0% | 0% | 100% | 100% | 100% |
| NEBARU | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

#### fairytail-en, subtitles

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 69% | 0.00 | 0/11 | 0.00 | none | 21% | 26% |
| wespeaker | 73% | 0.00 | 0/11 | 0.00 | none | 34% | 38% |
| wavlm_sv | 59% | 0.73 | 8/11 | 0.15 | 24 pairs | 89% | 98% |
| campplus | 31% | 0.00 | 0/11 | 0.00 | 45 pairs | 43% | 63% |
| eres2netv2 | 75% | 0.36 | 4/11 | 0.00 | none | 57% | 56% |
| titanet | 72% | 0.00 | 0/11 | 0.00 | none | 37% | 39% |
| fusion | 73% | 0.00 | 0/11 | 0.00 | none | 29% | 32% |

Per-character nearest-centroid accuracy (fairytail-en, subtitles):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| TOUKA | 6% | 14% | 12% | 4% | 28% | 11% | 9% |
| ERZA | 94% | 99% | 88% | 53% | 96% | 96% | 100% |
| HAPPY | 82% | 81% | 21% | 0% | 81% | 80% | 83% |
| LAXUS | 78% | 74% | 84% | 52% | 84% | 90% | 74% |
| NATSU | 88% | 90% | 71% | 37% | 76% | 89% | 95% |
| LUCY | 95% | 95% | 95% | 74% | 94% | 97% | 95% |
| CANA | 100% | 100% | 93% | 22% | 100% | 100% | 100% |
| KIRIA | 91% | 100% | 77% | 59% | 95% | 91% | 100% |
| JUVIA | 100% | 100% | 81% | 53% | 100% | 100% | 100% |
| GRAY | 100% | 100% | 100% | 0% | 100% | 100% | 100% |
| CHARLE | 59% | 100% | 0% | 0% | 59% | 59% | 100% |
| WENDY | 39% | 100% | 0% | 0% | 100% | 39% | 100% |
| NEBARU | 100% | 100% | 100% | 0% | 100% | 100% | 100% |

#### grid, diarization

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 81% | 0.50 | 1/2 | 0.17 | none | 73% | 12% |
| wespeaker | 96% | 0.50 | 1/2 | 0.17 | none | 73% | 12% |
| wavlm_sv | 58% | 0.50 | 1/2 | 0.17 | none | 76% | 12% |
| campplus | 30% | 0.50 | 1/2 | 0.17 | none | 74% | 12% |
| eres2netv2 | 85% | 0.50 | 1/2 | 0.17 | none | 73% | 13% |
| titanet | 85% | 0.50 | 1/2 | 0.17 | none | 73% | 13% |
| fusion | 96% | 0.50 | 1/2 | 0.17 | none | 73% | 12% |

Per-character nearest-centroid accuracy (grid, diarization):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| 새벽 | 100% | 100% | 67% | 49% | 100% | 100% | 100% |
| 부국장 | 100% | 100% | 80% | 14% | 100% | 100% | 100% |
| 어진 | 100% | 100% | 51% | 0% | 100% | 100% | 100% |
| 종이 | 0% | 100% | 0% | 0% | 0% | 0% | 100% |
| 새하 | 51% | 100% | 51% | 51% | 100% | 100% | 100% |
| 마녹 | 0% | 0% | 0% | 100% | 0% | 0% | 0% |

#### grid, subtitles

| Embedder | NC overall | Majority fraction | Passing/recurring | Strict | Merges | Speech on modal id | Coverage |
|---|---|---|---|---|---|---|---|
| ecapa | 82% | 0.00 | 0/6 | 0.00 | none | 0% | 0% |
| wespeaker | 93% | 0.00 | 0/6 | 0.00 | none | 0% | 0% |
| wavlm_sv | 59% | 1.00 | 6/6 | 0.73 | 6 pairs | 98% | 90% |
| campplus | 24% | 0.00 | 0/6 | 0.18 | 7 pairs | 42% | 55% |
| eres2netv2 | 89% | 0.00 | 0/6 | 0.00 | none | 22% | 14% |
| titanet | 86% | 0.00 | 0/6 | 0.00 | none | 0% | 0% |
| fusion | 93% | 0.00 | 0/6 | 0.00 | none | 6% | 3% |

Per-character nearest-centroid accuracy (grid, subtitles):

| Character | ecapa | wespeaker | wavlm_sv | campplus | eres2netv2 | titanet | fusion |
|---|---|---|---|---|---|---|---|
| 새벽 | 95% | 95% | 74% | 25% | 91% | 95% | 95% |
| 새하 | 84% | 92% | 56% | 16% | 97% | 97% | 92% |
| 부국장 | 100% | 100% | 84% | 30% | 95% | 94% | 100% |
| 종이 | 35% | 100% | 23% | 55% | 83% | 46% | 100% |
| 어진 | 100% | 100% | 46% | 25% | 100% | 100% | 100% |
| 팀장 | 48% | 100% | 0% | 0% | 74% | 22% | 100% |
| 남자 | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 유령 | 100% | 100% | 100% | 0% | 100% | 100% | 100% |
| 여자1 | 0% | 100% | 0% | 0% | 0% | 100% | 100% |
| 여자2 | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 마녹 | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## CAM++ file check

CAM++ on the `en_voxceleb` file scored worst (24% NC on fairytail diarization). Because that looked like a bad model for this material rather than a bad backend, the sherpa release's `zh_en_16k-common_advanced` CAM++ was run once through the `file` option on fairytail, diarization: NC 66% overall (TOUKA 3%, LUCY 58%, CHARLE 0%, WENDY 100%), 2 of 9 passing, four merges. That is ECAPA's level, no better, and it was run on one cell only. The default `campplus` file stays the one in the tables.

## Findings

- No new embedder separates the confusable groups better than ECAPA by a margin that survives the tuning-pair noise. On fairytail (ja) diarization NC is 71% fusion, 70% wespeaker, 69% eres2netv2, 68% ecapa, 67% titanet, 49% wavlm_sv, 24% campplus; the top five are within 4 points. On fairytail-en diarization, titanet leads at 78% against ecapa's 72%, and all of that lead is TOUKA (34% against 3%); every other character matches or trails ECAPA.
- Happy/Charle/Wendy: on diarization ecapa, wespeaker, titanet and fusion keep HAPPY at 78 to 88% and CHARLE and WENDY at 100% in both Fairy Tail sets (eres2netv2 gains HAPPY at 94% on ja but loses WENDY there; wavlm_sv and campplus lose one or more of the three). On subtitles the group is thin (CHARLE and WENDY have a few seconds each) and swings 39 to 100% between embedders in both directions, ecapa 100/61 and wespeaker 59/100 on ja; wespeaker, fusion and eres2netv2 hold WENDY at 100% in both sets.
- Lucy/Cana/Juvia: wespeaker and fusion raise LUCY from 66% to 77% and CANA from 89% to 100% on ja diarization; on en diarization all of ecapa, wespeaker, eres2netv2, fusion and titanet are at 100% on all three. The CANA + JUVIA merge in fairytail-en and the JUVIA + LUCY merge in fairytail stay for every embedder (wavlm_sv and campplus add more merges), so the registry merges are not removed by a stronger embedder.
- TOUKA stays at 3 to 19% on both Fairy Tail diarization sets for all but titanet on the English dub (34%), and at 0 to 28% on subtitles (eres2netv2 21 and 28%). TOUKA has 5.2 s in E12 and 52.7 s in E13, so the E12 centroid that classifies E13 is built from a single short stretch; I read this as a thin-centroid problem more than a similar-voices problem (inference; I did not test it with a different tune order, which would be a tune-set change).
- wespeaker is the most consistent gain, mostly on Grid: ja diarization 70 against 68, ja subtitles 63 against 64, en diarization 71 against 72, en subtitles 73 against 69, grid diarization 96 against 81, grid subtitles 93 against 82. Outside Grid that is within 4 points either way. It costs 40 MiB of torch memory against ECAPA's 1570.
- fusion (ecapa + wespeaker) tracks wespeaker within a point on every cell and costs ECAPA's memory plus a second model load per episode (the members load one after the other); it adds nothing measurable over wespeaker alone.
- wavlm_sv and campplus are worse than ECAPA on NC in all six cells (wavlm 43 to 59 against 64 to 82, campplus 21 to 36). Neither is worth a profile.
- eres2netv2 is at or above ECAPA on NC in five of six cells (ja diarization 69 against 68, ja subtitles 71 against 64, en diarization 71 against 72, en subtitles 75 against 69, grid 85 against 81 and 89 against 82), about as good as wespeaker on the anime subtitle cells. It is CPU-only, a 71 MB download, and on ja diarization it adds three merges (JUVIA + TOUKA, LUCY + TOUKA, ERZA + KIRIA, besides JUVIA + LUCY) and passes 2 of 9 recurring characters against ECAPA's 3. titanet is ahead in four cells and behind in two (67/60/78/72/85/86 against 68/64/72/69/81/82); its en-diarization lead is the single TOUKA effect noted above, and it also adds ERZA + KIRIA to the ja merges.

## Recommendation

- Anime (fairytail ja and en): keep ECAPA as the default; carry `wespeaker` forward as the one candidate to test further with the matcher and threshold work (it is level on diarization, ahead on three of the subtitle and grid cells, and nearly free on the GPU) and eres2netv2 as a second candidate for the subtitle segmentation (CPU-only, but it adds merges on ja diarization), and treat fusion as optional, since it adds nothing over wespeaker. Drop wavlm_sv and campplus. Do not expect an embedder swap to fix TOUKA or the CANA + JUVIA / JUVIA + LUCY merges; those need either more enrolment speech per character or the matcher work of the next tasks.
- Live action (Grid, thin): `wespeaker` (96% diarization, 93% subtitles against ECAPA's 81% and 82%), on six characters and 85 labeled spans. Flagged thin: the difference is one or two characters (종이 0% to 100%, 새하 51% to 100%) and should be re-checked on a larger live-action set before it drives a default.
- No profile or default was changed in this task. The ledger ruling for the numbers above is recorded in the report.
