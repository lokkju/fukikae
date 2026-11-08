# Ground-truth plan: execution ledger

Plan: `docs/superpowers/plans/2026-11-08-ground-truth.md`. Rulings GT1-GT7 are the controller's; this file records the facts Task 7 onwards produced, with their numbers.

## Task 7: live-action test sets and baselines (BBT en/de/es, Alice in Borderland)

Run 2026-11-08 at dcb4935 (code at 4591303 plus the alignment fix below). Kaze, Kaoru is split off to Task 7b (GT7): its three weekly packs were still arriving on the media share (about 10-16% allocated at 11:12 UTC).

### Media (Step 1)

`df -h /`: 136 GB free before the copy, 129 GB after; 118 GB free at the end of the task.

| Set | Files (library, read-only volume on the NAS media share) | Streams |
|---|---|---|
| BBT S01E01-E03 | `The Big Bang Theory - S01E0N - <title> Bluray-1080p.mkv` (1378 s, 1269 s, 1326 s) | 0 hevc; audio 1 eng, 2 ger, 3 spa (aac); subtitles 4 eng, 5 fre, 6 ger, 7 spa, 8 spa, all `dvd_subtitle` (image) |
| Alice in Borderland S01E01-E03 | `Alice in Borderland (2020) - S01E0N - Episode N WEBDL-720p.mkv` (2912 s, 3040 s, 2549 s) | audio 1 eng, 2 jpn (eac3); subtitle 19 jpn "Japanese" (SDH, SubRip) and 20 jpn "Forced" on all three; no E04 substitute needed |

### BBT labels (Step 3)

- The release's English track is image-based, so I OCR'd it rather than retiming an external file: one frame per subtitle event rendered by ffmpeg, then tesseract 5.5 (`captare-ocr-tesseract:1`, no network). The script is outside the repo at `~/fukikae-ws/subs/bbt/ocr-tools/vobsub_ocr.py`. The cues keep the disc's own timing, so no subalign step. E01 has 407 events, E02 364 and E03 379; every event produced text. The usual OCR slip is `|` for `I`.
- The transcripts are the 17 bigbangtrans S01 pages fetched at Task 4, copied to `~/fukikae-ws/subs/bbt/transcripts/`.
- `eval transcript-align` on E01-E03 only, because only those episodes have media:
  - S01E01: 98% of cues, 96% of words. Leonard 179, Sheldon 144, Penny 85, Howard 29, Receptionist 7, Raj 3, Voice from buzzer 1, Voice 1, Enormous man 1, Man 1.
  - S01E02: 97% of cues, 95% of words. Leonard 138, Sheldon 131, Penny 76, Howard 28, Raj 15.
  - S01E03: 97% of cues, 94% of words. Leonard 188, Sheldon 90, Penny 48, Howard 39, Lesley 24, Raj 12, Class instructor 4, Doug 1, Waitress 1.

  All five main characters speak in every episode. Raj has only 3 lines in E01 because he doesn't talk in front of Penny.
- The subtitle-to-speech alignment check (auto-label) gives 83%/81% overlap for en, 86%/85% for de and 80%/80% for es on E01/E02, all at an offset of 0.0 to -0.2 s. The English timing fits all three dubbed tracks.
- `labels-aliases.yaml` in each BBT workspace has: Leslie -> Lesley, Dr Gablehouser -> Gablehouser, and Voice from buzzer, Voice and Howard's phone -> `?` (not scored). "Dr Koothrappali" in S01E08 is Raj's father, who appears with Mrs Koothrappali, so it gets no alias. Mrs Cooper, Howard's Mother and Christie stay three separate people.

### Alice in Borderland labels (Steps 1 and 5)

- The SDH reader works on stream 19:
  - E01: 528 cues, 180 named lines. カルベ 54 (77 s), アリス 45 (74 s), チョータ 43 (70 s), シブキ 18 (31 s), 始 5, ミナミ 6, チョータの母 4, カズ 2, エミ 2, 有栖賢治 1.
  - E02: 517 cues, 127 lines. アリス 30, カルベ 29, シブキ 12, チョータ 10, ニトベ 10, チシヤ 7, and others.
  - E03: 439 cues, 123 lines. カルベ 36, チョータ 25, アリス 27, シブキ 26, and others.

  The reader treats 合成音声 (the game's synthetic voice) and the sound tags as descriptions.
- Aliases, per GT3: 有栖良平 -> アリス. 有栖賢治 -> 賢治 and 有栖 始 -> 始 join each full name to the same person's given-name tag; both are Arisu's family and stay apart from アリス. 宇佐木重憲 is Usagi's father and stays apart from ウサギ.
- The alignment check is below the 0.6 floor: 66% (E01) and 56% (E02), with the best offset at +0.1 s and +0.0 s. The SDH track includes sound-effect cues and long display times, so the timing is right but the overlap share is low. Every aib command therefore passes `--force`.
- After diarization segmentation only 19 of 103 segments (E01) and 15 of 95 (E02) carry a label, and coverage is 12%. The named lines are sparse inside long diarized spans. Only アリス and カルベ reach 5 s in both tuning episodes; チョータ has 1.7 s in E02. **aib is flagged thin: 2 recurring characters in its tuning episodes, fewer than 3.**

### Kaze, Kaoru: the steps that need no audio (done ahead of Task 7b)

- I ran subalign by text, ARIB .ass against the web release's .srt, on all 15 episodes into `~/fukikae-ws/subs/kaze-kaoru/aligned/S01E<NN>.ass`. Every episode has 100% inliers. The offset is -0.50 to -0.55 s, the scale 0.999986 to 1.000060, the rms 7-13 ms and the max residual 17-30 ms. E01 has 151 anchors and E11 has 144.
- `eval colour-map` over E01-E15 writes `~/fukikae-ws/kaze/labels-colours.yaml`: cyan -> 直美, green -> 真風, yellow -> りん. There are no conflicts. Green is tagged （真風） only from E05 on, and white stays unmapped.
- Labels with GT4 over E01-E15: りん 691, 直美 354, 真風 83, 美津 29, 吉江 25, 安 17, 信右衛門 13, 捨松 13, 環 13, 虎太郎 12, 卯三郎 12, 文 11, and minor characters. Per episode: E01 109 lines (真風 29, all narration), E02 63, E03 78, E11 97.
- 155 furigana events (`\fscy50`) are flagged `ruby` and skipped. Kaze goes through `read_cues` and `read_colour` only, never through `ass_names`' own parser, which is reached only with `--gt ass`.
- Kaze aliases: 槇村 りん -> りん; "りん 安" and "トヨ キク" -> `?`.
- Still open for Task 7b: copying the media, diarizing, the doubled-caption check against speech, the green check by ear, the coverage check and the baselines (`.superpowers/sdd/2026-11-08-ground-truth/task-7b-kaze.md`).

### Code fix from the real-file run

- dcb4935 `fix(align): wav2vec2 alignment models for German and Spanish`. The de and es tracks stopped at the asr stage ("wav2vec2 has no alignment model for language 'de'"). I added `jonatasgrosman/wav2vec2-large-xlsr-53-german` and `-spanish`, plus a case-folded vocabulary lookup, since both vocabularies are lowercase. German ß has no token and is skipped. After the fix, bbt-de S01E01 has 3152 of 3165 words scored.

### Baselines on tuning episodes (Step 6, tag `t7-baseline`)

Live-action is `registry resolve` (joint, 0.6, `episode_cannot_link = false`) followed by `eval identity --from-assignments`. Default is the per-episode centroid characters stage in a `cp -al` copy. Both use WeSpeaker on pyannote diarization with the min-speech at 5 s. All 8 rows are tune rows. S01E03 was diarized but not scored.

| Set | Profile | Recurring | Clean | Merges | Fraction | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| bbt (en) | live-action | 4 | 4 | 0 | 1.00 | 47% | pass |
| bbt (en) | default | 4 | 4 | 0 | 1.00 | 47% | pass |
| bbt (de) | live-action | 4 | 2 | 1 (Leonard + Sheldon) | 1.00 | 53% | fail |
| bbt (de) | default | 4 | 2 | 1 (Leonard + Sheldon) | 1.00 | 53% | fail |
| bbt (es) | live-action | 4 | 1 | 3 (Howard, Leonard, Sheldon on one id) | 1.00 | 50% | fail |
| bbt (es) | default | 4 | 1 | 3 (same) | 0.75 | 50% | fail |
| aib (ja) | live-action | 2 | 0 | 1 (アリス + カルベ) | 0.00 | 12% | fail |
| aib (ja) | default | 2 | 0 | 0 | 0.00 | 9% | fail |

Labelled speech per character in the tuning episodes (E01, E02; live-action rows):

- bbt-en: Sheldon 207.0/150.5 s, Leonard 175.3/122.2, Penny 88.5/66.3, Howard 52.0/14.6, Raj -/8.0, Receptionist 4.6/-.
- bbt-de: Sheldon 215.3/202.2, Leonard 191.2/126.1, Penny 102.1/65.1, Howard 52.4/14.3, Raj 1.6/33.2, Receptionist 6.0/-, Man 1.5/-.
- bbt-es: Sheldon 196.5/173.3, Leonard 197.8/109.0, Penny 95.6/78.1, Howard 51.9/22.5, Raj 1.5/14.7, Receptionist 2.3/-.
- aib: アリス 18.6/7.2, カルベ 8.8/9.6, チョータ 10.2/1.7, 始 4.7/-, ニトベ -/4.2, and the rest under 2.5 s.

Raj recurs nowhere because he has under 5 s in E01 on every track. Ledger option hashes: bbt live-action `ad0ed977`, bbt default `fbfe504c`, aib live-action `c442755e`, aib default `1f881519`. The genre column reads `live-action` in all rows, the default copies included, through `[series] genre`.

On BBT en, live-action and default give identical assignments. On the German and Spanish dubs, both profiles put Leonard and Sheldon (and in es, Howard too) on one id, while nearest-centroid accuracy on the same speech is 96-99%. Correction (Task 7 review): the merge happens in diarization, not at the 0.6 matcher threshold. One pyannote speaker holds 93-97% of both Leonard's and Sheldon's labelled speech in de and es (en splits them, 78% and 82% on two speakers), so the matcher only ever sees one cluster and the two profiles' rows are identical. The 96-99% nearest-centroid accuracy is measured on the labelled spans, not the diarized clusters. Ruling GT9 adds pyannote's clustering threshold to Task 8's sweep.

## Task 7c: Bazinga English sets and the transcript-aligner check (D18)

Run 2026-11-08 at b323629 (`feat(eval): bazinga-labels`). Data: `bazinga/bazinga` at revision 96db30ff, TBBT and The Office S01E01-E05 under `~/fukikae-ws/bazinga/data/`; nothing from it is in git.

### Labels

`fukikae eval bazinga-labels` at `--min-confidence 0.9`, scale 1.0 (`-pal`) and 1.04271 (`-film`), into `~/fukikae-ws/subs/bazinga/<show>-<condition>/S01E<NN>.ass`. Both conditions keep the same words and turns:

| Episode | TBBT kept / low-conf / pseudo, turns (dropped), speakers | Office kept / low-conf / pseudo, turns (dropped), speakers |
|---|---|---|
| S01E01 | 3282 / 593 / 0, 345 (2), 7 | 3052 / 669 / 0, 295 (5), 14 |
| S01E02 | 2705 / 727 / 0, 272 (2), 5 | 3216 / 593 / 0, 277 (1), 12 |
| S01E03 | 2898 / 731 / 0, 292 (2), 9 | 3128 / 610 / 0, 313 (0), 12 |
| S01E05 | 2546 / 551 / 7, 255 (0), 6 | 2595 / 960 / 4, 280 (3), 15 |

S01E04 was converted too and is unused.

### Media and workspaces

Each wav is remuxed into an audio-only .mkv (`~/fukikae-ws/media/bazinga-<show>-<condition>/<Show> - S01E<NN>.mkv`, language eng), because episode lookup by code only finds video suffixes. The film copies go through `asetrate=16000*0.95904,aresample=16000`; ffmpeg rounds the rate to 15345 Hz, so the real stretch is 1.042685 against the labels' 1.04271, a 2.4e-5 mismatch (0.03 s at the end of an episode). Workspaces `bazinga-{tbbt,office}-{pal,film}` (profile live-action, `[series] genre = "live-action"`), diarized on S01E01-E03 and S01E05 at about 4 minutes an episode; `/` went from 125 GB to 116 GB free.

### Film-speed check

VAD (webrtcvad, mode 3) onsets against label turn starts on S01E01 (`~/fukikae-ws/bazinga/tools/vadcheck.py`):

| Condition | Median abs(onset - start) | Within 0.2 s (random times) | Global lag | Drift (robust fit) |
|---|---|---|---|---|
| tbbt-pal | 0.070 s | 79% (50%) | +0.02 s | +8e-6 s/s |
| tbbt-film | 0.070 s | 79% (51%) | +0.01 s | -2.4e-5 s/s |
| office-pal | 0.080 s | 76% (49%) | -0.01 s | -1.3e-5 s/s |
| office-film | 0.080 s | 74% (48%) | -0.02 s | -6.3e-5 s/s |

On TBBT, 8-9 of the 2-minute windows lock onto a spurious +1.1 to +1.5 s lag in both conditions alike; the robust fit drops windows more than 0.3 s from the median. `eval subalign --speech` of each condition's S01E01 labels against its own workspace: scale 1.000000 everywhere, offset +0.02 s (tbbt-pal), +0.00 s (tbbt-film, office-pal, office-film), labels over speech 95%, 94%, 85%, 85%.

### Baselines on tuning episodes (tag `t7c-baseline-pal` / `t7c-baseline-film`)

Ledger sets `bazinga-tbbt` and `bazinga-office`; the condition is the tag's suffix, since a shared tag would put both conditions in one matrix cell. Procedure as Task 7 (live-action: `registry resolve` then `eval identity --from-assignments`; default: `cp -al` copy, profile line removed, characters stage per episode). Option hashes are Task 7's: live-action `ad0ed977`, default `fbfe504c`. JSON results in `~/fukikae-ws/t7c/`.

| Set | Condition | Profile | Recurring | Clean | Merges | Fraction | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|
| bazinga-tbbt | pal | live-action | 4 | 4 | 0 | 1.00 | 55% | pass |
| bazinga-tbbt | pal | default | 4 | 4 | 0 | 1.00 | 55% | pass |
| bazinga-tbbt | film | live-action | 4 | 4 | 0 | 1.00 | 58% | pass |
| bazinga-tbbt | film | default | 4 | 4 | 0 | 1.00 | 58% | pass |
| bazinga-office | pal | live-action | 3 | 3 | 0 | 1.00 | 40% | pass |
| bazinga-office | pal | default | 3 | 2 | 0 | 0.67 | 40% | fail |
| bazinga-office | film | live-action | 4 | 3 | 0 | 0.75 | 41% | fail |
| bazinga-office | film | default | 4 | 3 | 0 | 0.75 | 41% | fail |

TBBT recurring: Leonard, Sheldon, Penny, Howard (Raj has 2 s of labelled speech in E01). Office recurring: Michael, Dwight, Jim, plus Pam on film only: her E02 labelled speech over diarized segments is 4.3 s on PAL and 6.1 s on film, so the 5 s bar itself moves with the condition. Every Office failure is Dwight's S01E01 on a different id from his E02/E03 id (share 0.73, 0.79, 0.63), never a merge. Nearest-centroid accuracy is 97-100% in every run. PAL against film changes no TBBT number; on The Office the one real difference is live-action's Dwight, kept on PAL and split on film.

### Transcript-aligner check (TBBT S01E01-E03)

Bazinga's film-scaled labels retimed onto bbt-en with `subalign --speech`: S01E01 scale 1.000000, offset +0.76 s, 94% over speech; S01E02 0.999001, +0.90 s, 93%; S01E03 0.999001, +0.86 s, 91%. The text route against `release-en` agreed on scale (0.9999, 0.9992, 0.9989) but refused on residuals (1.5-1.7 s max), because word-turn onsets and subtitle cue onsets differ by more than its 0.5 s bound. After the retime the speaker-agnostic lag between the two label sets is +0.04, +0.10 and -0.02 s.

Agreement over frames where both name exactly one speaker (`~/fukikae-ws/bazinga/tools/agree.py`): S01E01 97.7% of 715 s, S01E02 98.4% of 580 s, S01E03 98.5% of 663 s; **98.17% over 1958 s**. Line level (each of our lines against the gold speaker holding most of its time): 1039 of 1058 (98.2%). Name map: leonard_hofstadter Leonard, sheldon_cooper Sheldon, penny Penny, howard_wolowitz Howard, raj_koothrappali Raj, leslie_winkle Lesley, althea Receptionist, kurt Enormous man, instructor Class instructor, doug Doug, waitress Waitress.

The 19 disagreeing lines: one-word cues in fast exchanges whose subtitle display lags the speech into the next speaker's turn ("Done." / "Got it."); runs of repeated short tokens ("Hi." four times) where difflib pairs a cue with the wrong transcript instance; Bazinga's own errors, an apparent speaker boundary one word late (gold gives Sheldon only the word "half" that ends Leonard's line, then gives Leonard "Exactly half.", which the transcript gives Sheldon) and words missing from gold ("Okay.", "Six-thirty."); and Kurt, called "Enormous man" once and "Man" once by the transcript, which is an alias gap, not an alignment error.
