# Ground-truth plan: execution ledger

Plan: `docs/superpowers/plans/2026-11-08-ground-truth.md`. Rulings GT1-GT12 are the controller's; this file records the facts Task 7 onwards produced, with their numbers.

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

## Task 7b: Kaze, Kaoru (tuning episodes and the green check; GT11)

Run 2026-11-09 at b60013d, no code change. Per GT11, E11 (held-out) is not copied, diarized or scored here: week 3 was still downloading. It follows when week 3 completes.

### Media

Week 1 (`Kaze Kaoru Week 01`, 1080p web, Japanese subtitles) was complete on the media share. All five .mkv files were fully allocated (`du -k` x 1024 at or above the apparent size: EP001 560,720,896 against 560,242,625 bytes, and likewise for EP002-EP005). I copied EP001, EP002, EP003 and EP005 through the read-only volume `fukikae-media-ro` and removed it afterwards. The files are mode 0710 for uid 3000, group `users`, so the copy container needed `--group-add 100`. The copies are `Kaze Kaoru - S01E01.mkv`, `S01E02`, `S01E03` and `S01E05`, and their sizes match the source. Broadcast number = episode number (EP001 = S01E01). Each is 899.3 s with one audio stream, index 1, AAC, tagged `jpn` ("日本語 [オリジナル]"), so no `audio_track` override. `df -h /`: 100 GB free before the copy, 98 GB after.

E01-E03 ran `--to diarize` (2 min 38 s for E01, extract through diarize). E05 ran `--to separate` only, for the green check, and was never diarized or scored.

### Doubled captions and timing against speech

- The web release's .srt doesn't repeat a caption; it splits one. Each caption is cut at the next x.530 s boundary and continues with the same text (`00:00:08,023 --> 00:00:08,530` and `00:00:08,530 --> 00:00:12,863`, both "そうだねえ　あの２人の話をしようか。"). The first onset is the caption's onset and the union of the two pieces is its display span.
- `eval subalign --speech` refuses both the .srt and the aligned .ass on E01: share 59% < 60%, at scale 1.001001 (the 1001/1000 candidate) and offset -0.86 s. That fit is noise. At scale 1.0 the share is 0.588 across offsets -0.6 to -0.2 s and 0.587 at 0, so the speech-share objective can't place these captions to better than about half a second, and 1001/1000 wins by 0.0007.
- So I measured onsets directly. For cues that follow at least 1 s without a caption, with the split pieces joined, I took the lag from cue onset to the nearest rising edge of speech within 1.5 s. Speech came from two detectors: pyannote's turns, and an energy VAD on `vocals_16k.wav`. The .srt and the aligned .ass lead speech by a median of:

  | Episode | gap-led cues | pyannote median (IQR) | energy median (IQR) |
  |---|---|---|---|
  | E01 | 20 | +0.25 s (+0.11..+0.44) | +0.28 s (+0.13..+0.41) |
  | E02 | 35 | +0.27 s (+0.19..+0.61) | +0.24 s (+0.19..+0.50) |
  | E03 | 35 | +0.31 s (+0.15..+0.52) | +0.26 s (+0.14..+0.49) |

  The unretimed broadcast .ass on E01 lags by -0.28 s (pyannote) and -0.23 s (energy). The auto-label alignment report agrees in direction: best offset -0.3 s on E01, -0.1 s on E02 and 0.0 s on E03, none applied.
- Verdict: the .srt's first onset is the right one. A repeat's onset would make captions late, and they are early. The captions lead speech by a uniform 0.25-0.3 s, which is over the handoff's 0.2 s bound. I tested whether that matters instead of stopping. With every caption moved +0.25 s (`~/fukikae-ws/subs/kaze-kaoru/diag-shift+0.25/`, scored with `--no-ledger`), the live-action baseline, the ct 0.6 cell and the ct 0.4 cell (centroid, 0.6) give the same verdicts and the same りん shares (0.79, 0.79, 0.84), with coverage within one point. Labels are taken per diarized segment at a 60% overlap, and a quarter-second lead flips none of them. `aligned/` stays the ground truth.

### Green: 真風 or a separate narrator

- Text: E01-E04 green is narration ("そうだねえ　あの２人の話をしようか。", "私は　一体　誰かって？　フフフフフ…"). E05 green has both kinds of line. There is narration in the same persona, with the same wind onomatopoeia and sign-off ("ビョ～　ビョ～と吹いてんのにね～。", "村には　そよそよと穏やかさが戻った。", "続きは　また　風の吹く朝に。"). Then "上々だね。　さて　こちらにも吹くぞ。" leads straight into an on-screen fortune teller tagged （真風） ("（真風）もし　そこのお嬢さん。", "占い師。").
- Embeddings: WeSpeaker (the workspace's embedder, on CPU), every clip cut from its own episode's Demucs vocal stem. Six E01 narration clips, six E05 narration clips and six E05 on-screen （真風） clips, against the first six yellow (りん) and cyan (直美) cues of at least 2.5 s in each episode as negatives:

  | | E01 narration | E05 narration | E05 （真風） on screen |
  |---|---|---|---|
  | E01 narration centroid | 1.000 | 0.907 | 0.821 |
  | E01 りん / 直美 centroid | 0.165 / 0.197 | 0.215 / 0.242 | 0.251 / 0.222 |
  | E05 りん / 直美 centroid | 0.321 / 0.326 | 0.353 / 0.333 | 0.297 / 0.382 |

  The mean clip-to-clip cosine is 0.711 within E01 narration, 0.670 between E01 and E05 narration and 0.554 between E01 narration and E05 （真風）. Each E05 clip against the E01 narration centroid scores 0.68-0.85 (narration) and 0.32-0.80 (on screen; five of six at or above 0.61).
- Decision by the handoff's rule: same voice. Every centroid cosine is above the profile's match threshold of 0.6, and E05's green continues the narrator persona. Green scores as 真風 in every episode, and `aligned/` is used unchanged, so there is no `aligned-gt/`. Clips for the operator to confirm by ear are in `~/fukikae-ws/subs/kaze-kaoru/green/`: `E01-{a,b,c}-narr.wav` (8.00, 12.84, 68.60 s), `E05-{a,b,c}-narr.wav` (202.77, 212.31, 874.92 s) and `E05-{d,e,f}-majikaze.wav` (602.20, 604.77, 629.93 s); the numbers are in `green_cos.txt`.

### Coverage (auto-label, colour map plus （名前） prefixes, GT4)

Every Kaze command passes `--force`. The alignment check gives 65%, 60% and 57% overlap with the best offset at -0.3, -0.1 and 0.0 s, and E03 is under the 0.6 floor. Long display times, not misalignment, as with aib.

| Episode | cue spans | labelled | named lines | labelled speech over diarized speech | furigana events |
|---|---|---|---|---|---|
| E01 | 188 | 87 (46%) | 106 | 290 s of 479 s (61%) | 12 |
| E02 | 167 | 56 (34%) | 62 | 137 s of 436 s (31%) | 8 |
| E03 | 146 | 63 (43%) | 78 | 202 s of 373 s (54%) | 7 |

- Seconds of labelled cue time over diarized speech, at 5 s or more: E01 真風 113.1, りん 111.5, 直美 22.3, 中村 10.0, 信右衛門 8.5, 之宣 7.1, 安 7.0, 美津 6.6. E02 りん 82.7, 直美 20.1, 信右衛門 13.3, 安 5.6. E03 りん 111.9, 直美 51.9, 真風 13.1, 信右衛門 8.5, 虎太郎 5.4. 真風 has 2.3 s in E02.
- The harness counts less, because it labels a diarized segment only when one name covers 60% of it. On its count, only りん reaches 5 s in all three episodes (45.3, 27.1 and 31.1 s). 直美 has 3.8 s in E01, 真風 2.1 s in E02 and 信右衛門 2.2 s in E02. **Kaze is flagged thin: 1 recurring character in its tuning episodes by the harness (R6), fewer than 3.** Three characters (りん, 直美, 信右衛門) pass 5 s per episode on the looser cue-over-speech count.
- Furigana: each of E01-E15 reads to identical labels with and without its `\fscy50` events (155 in all), and no ruby text becomes a line. E06 has a ruby reading りん that happens to equal the name, and it still adds no line. The alias "りん 安" -> `?` dropped 4 lines (E01 3, E02 1).

### Baselines on tuning episodes (tag `t7-baseline`)

Procedure as Task 7. `kaze-default` is a `cp -al` copy with the profile line removed, `registry/` and E01-E03's `05-characters/` deleted, and the characters stage run per episode. Both runs add `--force`. JSON results are in `~/fukikae-ws/kaze/t7b/`.

| Set | Profile | opts | Recurring | Clean | Merges | Fraction | Coverage | Verdict |
|---|---|---|---|---|---|---|---|---|
| kaze (ja) | live-action | `fa8fda9f` | 1 | 0 | 0 | 0.00 | 24% | fail |
| kaze (ja) | default | `07cfff71` | 1 | 0 | 0 | 0.00 | 21% | fail |

りん holds 79% of her speech on one id (live-action: id 3 at 73%, 72% and 94% per episode), just under the 80% bar. Live-action's id 3 is only 65% pure (直美 19%, 真風 8%, 美津 8%). Nearest-centroid accuracy is 61% speech-weighted, and 直美's is 7%: her labelled speech classifies as りん. 真風 is 100% on its own id in both profiles.

### Task 8 Step 1 cells for kaze (tags `t8-sweep-ct{0.6,0.5,0.4}`)

`~/fukikae-ws/t8/ws/kaze-ct{0.6,0.5,0.4}` come from `setup.py`, with kaze added to the t8 drivers (the pre-kaze copies are kept as `*.bak-pre-kaze`). Only diarize was stale at 0.5 and 0.4, and each re-diarization took 32-33 s. Speakers per episode (E01/E02/E03): 9/7/8 at 0.6, 9/6/7 at 0.5, 8/6/8 at 0.4. The flags are `--tune S01E01,S01E02,S01E03 --gt colour --gt-file ~/fukikae-ws/subs/kaze-kaoru/aligned --force --set kaze`. That makes 54 ledger rows, none failed.

| ct | best clean/merges/fraction | cells tied at best | りん share |
|---|---|---|---|
| 0.6 | 0/0/0.00 | all 18 | 0.79 everywhere |
| 0.5 | 0/0/0.00 | all 18 | 0.742, or 0.796 under joint ecl-off at 0.50-0.70 |
| 0.4 | 1/0/1.00 | all 18 | 0.843 everywhere |

Gate sums over bbt-en/de/es, grid, bazinga-tbbt-pal, bazinga-office-pal and kaze (`~/fukikae-ws/t8/tables-diar.md`; pre-kaze tables in `tables-diar.pre-kaze.md`): ct 0.4 joint ecl-on 0.50-0.60 leads at 18 / 1 / 0.857. ct 0.5 centroid and ecl-on 0.50-0.60 follow at 17 / 1 / 0.714, and the provisional ct 0.6 joint ecl-off 0.60 is at 16 / 4 / 0.857. Kaze adds one clean character at ct 0.4 and none elsewhere, so it alone breaks Step 1's ct 0.5 / ct 0.4 tie. That margin is one character crossing the bar, 0.79 to 0.84.

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

## Task 8: pick, profile and held-out verdicts

Run 2026-11-09. Sweep (Step 1) as recorded under Task 7b and in `.superpowers/sdd/2026-11-08-ground-truth/task-8-step1-report.md`; the full tables are reproduced by `scripts/ground-truth/tab.py`. Results doc: `docs/results/2026-11-09-live-action-ground-truth.md`.

### Pick (Step 2)

- Gate sets: bbt-en, bbt-de, bbt-es, kaze, grid, bazinga-tbbt-pal, bazinga-office-pal; BBT counts three times. Film conditions alongside, aib diagnostic (GT8).
- Literal rule: ct 0.4, joint with `episode_cannot_link` on at 0.50-0.60, 18 clean / 1 merge / 0.857. Next: ct 0.5, centroid or joint with cannot-link on at 0.50-0.60, 17 / 1 / 0.714. Provisional (ct 0.6, joint, cannot-link off, 0.60): 16 / 4 / 0.857.
- GT12 overrides the literal rule: ct 0.4 against ct 0.5 is a tie (the one character is りん on Kaze, 0.79 at ct 0.6, 0.74-0.80 at ct 0.5, 0.84 at ct 0.4; 0.4 is outside VBx's `Uniform(0.5, 0.8)` tuning range), and ct 0.5 is picked.
- At ct 0.5, centroid and joint with cannot-link on tie at 17 / 1 / 0.714 for 0.50, 0.55 and 0.60 (both 16 at 0.65). Simpler matcher: centroid. Threshold: 0.6, the top of the plateau and today's value (Task 9's choice in the same situation).
- **Pick** (GT12 clustering value, GT14 matcher and threshold tie-break): `[diarize] clustering_threshold = 0.5`, `[characters] matcher = "centroid"` at `[registry] match_threshold = 0.6`. `[resolve]` holds the tied joint cell: joint, 0.6, `episode_cannot_link = true` (was `false`). That cell was never held out: every held-out run took the centroid path.
- **Language variants: none.** en, es, ja (kaze) and ko (grid) have no ct 0.5 cell above the pick. de has one by the letter: joint, cannot-link off, 0.60 gives 2 / 1 / 1.00 against the pick's 2 / 1 / 0.75. The gain is Sheldon's E02 speech (204 s, unassigned under centroid) joining the merged Leonard + Sheldon id; same merge, same failing verdict. It is a joint setting, read only by `registry resolve`, and a `[resolve.lang.de]` table would never be read on the centroid path the pick ships. Not shipped; for the controller to confirm as a ruling.

### Profile (Step 3)

`6bb10c2 feat(profiles): live-action profile from four-language ground truth`: `src/fukikae/profiles/live-action.toml` (PROVISIONAL dropped), `tests/test_profiles.py` (the profile resolves to the pick for every language; the clustering threshold reaches `PyannoteDiarizer`; `[resolve]` gives joint 0.6 with cannot-link on), README's live-action bullet and a cache note (the new clustering value re-runs diarize in existing live-action workspaces). Committed before any held-out run.

### Held-out (Step 4), once, tags `t8-holdout`, `t8-holdout-pal`, `t8-holdout-film`

Workspaces `~/fukikae-ws/t8/ws/<set>-pick`: `cp -al` of the ct 0.5 copy with its `[diarize]` override removed. Under the committed profile the tuning episodes' diarize keys were fresh (only characters stale), and each held-out episode re-diarized at 0.5 (39-42 s; grid E05 78 s, run directly on the existing vocals through `stage_direct.py`, since grid's separate and asr keys predate GT6). Then the characters stage on the tuning episodes in order and the held-out episode last, and `eval identity --from-assignments`. The default profile was held out once in Task 7's `-default` copies (characters stage on the held-out episode only; their tuning registries are Task 7's).

| Set | Held-out | Pick: recurring, clean / merges / fraction, coverage | Verdict | Default |
|---|---|---|---|---|
| bbt en | S01E03 | 4, 4 / 0 / 1.00, 47% | pass | 4 / 0 / 1.00 pass |
| bbt de | S01E03 | 5, 3 / 1 (Leonard + Sheldon) / 0.80, 34% | fail | 3 / 1 / 1.00 fail |
| bbt es | S01E03 | 4, 4 / 0 / 1.00, 49% | pass | 1 / 1 / 0.50 fail |
| bazinga-tbbt PAL | S01E05 | 6, 4 / 0 / 0.67, 61% | fail (Howard 0.52, Raj 0.00) | 4 / 0 / 0.67 fail |
| bazinga-tbbt film | S01E05 | 6, 4 / 0 / 0.67, 67% | fail (Howard 0.49, Raj 0.00) | 4 / 0 / 0.67 fail |
| bazinga-office PAL | S01E05 | 3, 3 / 0 / 1.00, 39% | pass | 3 / 0 / 1.00 pass |
| bazinga-office film | S01E05 | 3, 3 / 0 / 1.00, 38% | pass | 3 / 0 / 1.00 pass |
| grid | S01E05 | 5, 4 / 0 / 0.80, 16% | fail (종이 0.00, no tuning id) | not run |
| kaze | S01E11 | **pending** (week 3 downloading) | | |

- bbt de: on E03, pyannote at 0.5 splits the leads; Leonard stays on the merged tuning id 1 (0.89) and Sheldon's 152 s go to a new actor (0.01 on a tuning id).
- bazinga-tbbt: Raj has 2 s in E01 and no tuning id; Leslie (22 s in tuning) holds 0.98. Office: Michael, Jim, Pam recur on E05 (Dwight under 5 s).
- grid: the provisional profile scored 0.80 on E05 with a 새하 + 어진 merge; the pick scores 0.80 with none.
- The tune rows that the held-out runs write (pipeline, not harness) match the harness sweep's ct 0.5 centroid 0.60 cells on every set (e.g. bbt de 2 / 1 / 0.75, es 3 / 0 / 0.75, grid 1 / 0 / 0.50).

### Kaze E11, pending

`~/fukikae-ws/t8/ws/kaze-pick` exists with E01-E03's characters stage run at the committed profile (no E11, nothing scored); the default profile's tuning registry is Task 7b's in `~/fukikae-ws/kaze-default`. Follow-up, once each, no re-tuning, from the repository root. Kaze can't show over-merge (one recurring character in tuning).

```bash
# 1. Copy EP011 (week 3 must be under complete/tv/, nothing of it under incomplete/)
docker volume create --driver local --opt type=nfs \
  --opt o=addr=<nas>,ro,nfsvers=4 \
  --opt device=:<media-export> fukikae-media-ro
docker run --rm --user "$(id -u):$(id -g)" --group-add 100 -v fukikae-media-ro:/share:ro alpine:3.20 \
  sh -c 'C=/share/<kaze-dir>; ls -l "$C"/*Week*03*/Kaze.Kaoru.EP011.*.mkv; du -k "$C"/*Week*03*/Kaze.Kaoru.EP011.*.mkv'
docker run --rm --user "$(id -u):$(id -g)" --group-add 100 -v fukikae-media-ro:/share:ro \
  -v ~/fukikae-ws/media:/out alpine:3.20 \
  sh -c 'set -e; f=$(ls /share/<kaze-dir>/*Week*03*/Kaze.Kaoru.EP011.*.mkv); cp "$f" "/out/Kaze Kaoru/Kaze Kaoru - S01E11.mkv"; chmod 0640 "/out/Kaze Kaoru/Kaze Kaoru - S01E11.mkv"'
docker volume rm fukikae-media-ro
# 2. Caption-onset check against ~/fukikae-ws/subs/kaze-kaoru/aligned/S01E11.ass, as Task 7b (pyannote turns and energy VAD)
# 3. Live-action pick (diarizes E11 at clustering 0.5, then characters), scored once
KAZE_EPS=S01E11 scripts/ground-truth/holdout_run.sh kaze-pick
scripts/ground-truth/holdout_score.sh ~/fukikae-ws/t8/ws/kaze-pick
# 4. Default profile (diarizes E11 at 0.6, then characters against Task 7b's registry), scored once
mise -C ~/projects/fukikae exec -- uv run --directory ~/projects/fukikae \
  fukikae run ~/fukikae-ws/kaze-default S01E11 --to characters
scripts/ground-truth/holdout_score.sh ~/fukikae-ws/kaze-default
```

### Reproducibility

The drivers are committed under `scripts/ground-truth/` (`65df647`), with `~/fukikae-ws` replaced by `$FUKIKAE_WS` (default `~/fukikae-ws`) and the repository path found from each shell script's own location; ruff formatting only otherwise. The repository `tab.py` reproduces `tables-diar.md` byte for byte. sha256 of the workspace originals as run:

```
ffbbf367c5eb181eb72812e0a7d064976dcad63579993c5e80b7ebb62eba537e  t8/setup.py
c0324b562ec862807d6bebe42fc243902679fc997c9191db23c3e4331a0687fb  t8/sweep.py
eaef466448df6db85ab30b67bc29ef85c3f8cdf45509ee73a213b03ebcefecb9  t8/diar_direct.py
106fa7a3a918399017dc98d07387ee6f0e0031c3e1e131199c28804d14b92d8f  t8/tab.py
9418cb37203ffacbd8964b2f14ac5273637a4970cbbeb63e52d7957cc254d034  t8/stale.py
51fd70508d2e9260e5b1341ed3c693c7d871ac85bab6ef9ce7c17c1874470d30  t8/diar.sh
0519e49fbbbb7fbc032f201c5c5ec614da8c54861c2bd8aeaa06327e1b37c20c  t8/holdout_setup.py
5189f95327b65a4f90c7a099fd351b141523c02b9aef657210785d14e424fb22  t8/holdout_stale.py
8a937bbeeb24ea61c4af88d64532b972015f6194f40cc6210294af3617e79f60  t8/stage_direct.py
cd802666b9e5297884d10efb39254318618c8f0364f4804df8fe7c84dfd2e316  t8/holdout_run.sh
c32ea2152ba2f68078258914cc5c264c24a0314cf91abe5ca40b0a753f940134  t8/holdout_score.sh
```
