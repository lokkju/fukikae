# Speaker-identity test sets

_Run 2026-11-05 on the GPU host (RTX 3090, index 0), branch feat/speaker-identity._

Five workspaces under `~/fukikae-ws/` (media and workspaces stay outside the repo), each run through `fukikae run <ws> <ep> --to diarize`, labeled from subtitle ground truth with `fukikae eval auto-label`, and scored once with the ECAPA baseline: `fukikae eval identity <ws> --tune A,B --gt <kind> --segmentation diarization --embedder ecapa --matcher centroid --threshold 0.6`. No holdout was scored and nothing here was chosen by looking at one. A recurring character has at least 5 s of labeled speech in each tune episode (Ruling R6). Every set uses Demucs separation, pyannote community-1 diarization and ECAPA embeddings; the ASR backend is noted per set. The baseline outputs are in `~/fukikae-ws/baselines/<set>.{txt,json}`.

## Summary

| Set | Audio | Tune | Holdout | Ground truth | Alignment (tune/holdout) | Recurring at 5 s | Passing | Fraction | Merges | Speech on modal id | Strict | Coverage | Nearest-centroid | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fairytail | ja | E12, E13 | E14 | ass | 77%, 78% / 81% | 9 | 3 | 0.33 | JUVIA + LUCY | 70% | 0.17 | 32% | 68% | fail |
| fairytail-en | en dub | E12, E13 | E14 | ass | 85%, 87% / 87% | 10 | 7 | 0.70 | CANA + JUVIA | 72% | 0.08 | 43% | 72% | fail |
| grid | ko | E02, E04 | E05 | sdh | 75%, 76% / 77% | 2 | 1 | 0.50 | none | 73% | 0.17 | 12% | 81% | fail |
| cyberfarm | ru | E01, E02 | E04 (E03 spare) | sdh | 83%, 75% / 83% | 0 | 0 | 0.00 | none | 71% | 0.50 | 21% | 100% (2 characters) | insufficient |
| linkclick-en | en dub | E04, E05 | E06 | sdh | 67%, 80% / 69% | 0 | 0 | 0.00 | none | 67% | 0.25 | 17% | 28% | insufficient |

Alignment is the share of subtitle time over detected speech at the best global offset; the best offset was within 0.2 s everywhere and none was applied. All sets clear the 0.6 alignment bar. Two sets (cyberfarm, linkclick-en) have no recurring character in the tune pair and are dropped from the comparison tasks; the reason is under each set. The bar is the foundation's: pass needs a fraction of at least 0.90 and no merges, so "fail" is any fraction below 0.90 or any merge.

## fairytail (ja)

- Files: Fairy Tail 100 Years Quest S01E11 to S01E14, `~/fukikae-ws/media/Fairy Tail - 100 Years Quest/Season 1/`. E12 to E14 are the WEBDL-1080p release; E11 is an SDTV release.
- Audio: `jpn`, stream 1 (`source_language = "ja"` picks it; checked with `choose_audio_track`). ASR: `qwen3_asr` for ja. Aligner: default.
- Ground truth: `ass`, the English Crunchyroll track naming a speaker on each Dialogue line. E12 to E14 labeled 3 episodes' clean spans. E11 is run through diarization but excluded: auto-label could not pick its ASS stream (it is the lone "English" track on a different release) and needed `--subtitle-track 2`, and a different release adds nothing to a tune/holdout split.
- Coverage: 32% of speech (labeled in the harness's weighting); E12 152 of 174 diarized segments labeled, E13 116 of 149.
- Aliases accepted (`~/fukikae-ws/fairytail/labels-aliases.yaml`, pre-existing): `NABARU: NEBARU`, a typo on 3 lines of E12 against NEBARU in all of E11 to E14. No new ones. Suggestions declined: `MAKAROV` with `MAKAROV (WRAITH)` (different roles), the `TOWNSPEOPLE`/`WIZARD` letters (different extras), `YOUNG BOB` with `YOUNG ROB` (under 5 s each).
- Baseline: 9 recurring, 3 passing, fraction 0.33, one merge (JUVIA + LUCY on id 4), verdict fail. Nearest-centroid accuracy 68%, with TOUKA at 3% and every other recurring character between 66% and 100%. This reproduces the registry result in `2026-11-05-registry-fairytail.md` on the 5 s bar.

## fairytail-en (English dub of the same files)

- Files: the same E12 to E14. Audio: `eng`, stream 2 (checked). `source_language = "en"`, ASR `faster_whisper`, aligner wav2vec2. E11 was not run for this set.
- Ground truth: `ass`, the same English track and aliases file (copied). Coverage 43%; E12 172 of 200 segments labeled, E13 111 of 139. Alignment is better than the ja track (85 to 87%); I infer that the dub follows the English script, which I have not checked line by line.
- Aliases: `NABARU: NEBARU` (copied); no others, same declines as above.
- Baseline: 10 recurring, 7 passing, fraction 0.70, one merge (CANA + JUVIA on id 10), verdict fail. Nearest-centroid accuracy 72%; TOUKA is again the outlier at 3%, ERZA 96%, HAPPY 86%, LAXUS 91%, the rest 100%.

## grid (ko, live action)

- Files: Grid (2022) S01E02, S01E04, S01E05, Korean audio, 1080p web `.mkv`, in `~/fukikae-ws/media/Grid/`. E02 was copied from the NAS media share (read-only NFS volume, removed afterwards); E04 and E05 were local.
- Audio: `kor`, stream 1 (one audio track). ASR: `qwen3_asr` for ko.
- Ground truth: `sdh`, the `Korean [CC]` track, stream 2. Auto-label's own track choice failed ("no text subtitle stream for sdh ground truth"), so `--subtitle-track 2` is passed to both auto-label and identity. Coverage is thin: 85 of 426 spans labeled in the three episodes, 12% of speech in the tune pair (E02 28 of 127 segments, E04 25 of 140). The CC track names a speaker only where the viewer would not know.
- Aliases: none offered, none accepted.
- Baseline: 2 recurring (부국장, 새벽), 1 passing, fraction 0.50, no merges, verdict fail. Nearest-centroid accuracy 81% over six characters labeled in both episodes (새벽, 부국장, 어진 at 100%, 새하 51%, 종이 and 마녹 0%). Kept, with the caveat that two recurring characters make every fraction on this set a coin with two sides: one character moves it by 0.5.

## cyberfarm (ru, live action): dropped

- Files: The Cyberfarm S02E01 to S02E04 (Venus Central, Space Trash, Izvinigorod, Legacy Code), WEBDL-720p `.mkv`, copied from the NAS media share through a read-only Docker NFS volume (only these four files; volume removed afterwards) into `~/fukikae-ws/media/The Cyberfarm/Season 2/`.
- Audio: each file has four `rus` tracks (stereo-as-5.1 `ac3` and `eac3`, each with an audio-description twin). `source_language = "ru"` takes the first, stream 1, `Русский 5.1`, the plain mix; checked on E01 and E04. ASR: `faster_whisper`.
- Config: `[align] backend = "qwen3_forced_aligner"`. The default wav2vec2 aligner has no Russian model and the first run failed in the ASR stage with `wav2vec2 has no alignment model for language 'ru'`; the Qwen3 aligner covers ru. No code changed, and the failed run left nothing behind to clean.
- Ground truth: `sdh`, `Русские (для слабослышащих)`, stream 6. Coverage 8% of spans (44 of 563 over four episodes), 21% of speech in the tune pair; alignment 83%, 75%, 83%, 83% for E01 to E04.
- Aliases accepted (`~/fukikae-ws/cyberfarm/labels-aliases.yaml`): `Костя радостно: Костя`, the same character with a delivery note (1 line, 1.8 s). Declined: `Галя голосом Кости` (Galya speaking with Kostya's voice, a different voice from Kostya's own), `отец Коли` (Kolya's father).
- Baseline: no character reaches 5 s in both E01 and E02 (Костя has 28.2 s then 1.6 s; Коля 3.3 s then 1.6 s), so recurring is 0 and the verdict is `insufficient`. The 100% nearest-centroid figure covers two characters on a handful of seconds and means nothing. Reason for the drop: the Russian SDH track tags a name only where a line needs one, which leaves the second tune episode with about 10 labeled seconds. E03 and E04 are diarized and available if a later task wants a different tune pair, but the brief's pair stands and the holdout rule forbids choosing by E04.

## linkclick-en (English dub, donghua): dropped

- Files: Link Click S01E04 to S01E06 (Confess, Farewell, Search of the Child), WEBDL-1080p, `~/fukikae-ws/media/Link Click/Season 1/`.
- Audio: `eng`, stream 2 (checked; stream 1 is Mandarin). ASR: `faster_whisper`, aligner wav2vec2.
- Ground truth: `sdh`, `Dub (SDH)`, stream 4, chosen by auto-label without an override. Coverage 43 of 378 spans (11%), 17% of speech in the tune pair; alignment 67%, 80%, 69% for E04 to E06, passing but the lowest of any set.
- Aliases: none accepted. Auto-label suggests `CHEN'S MOM` with `CHENG'S MOM`; the harness lands both on one character id (12), which is suggestive, but Chen Xiao and Cheng Xiaoshi are different characters, so the names alone don't settle it. If someone who has watched the series confirms, it adds to one character.
- Baseline: no character has 5 s in both E04 and E05 (CHENG XIAOSHI 12.2 s then 1.9 s, CHEN XIAO 1.6 s then 11.9 s, LU GUANG 6.9 s then 3.7 s), so recurring is 0, verdict `insufficient`, nearest-centroid 28%. Reason for the drop: the dub's SDH track tags a minority of lines and the main cast alternates between episodes.

## What the later tasks get

Three usable sets: fairytail and fairytail-en (the real comparison material: 9 and 10 recurring characters, 12 and 13 labeled in both tune episodes) and grid (a thin cross-check on a third language with two recurring characters). The ECAPA row for each is the number to beat. In both Fairy Tail sets TOUKA is at 3% nearest-centroid accuracy, the same character that collapsed in the registry run.

Reproduce one set: `fukikae run ~/fukikae-ws/<set> <ep> --to diarize` per episode, then `fukikae eval auto-label ~/fukikae-ws/<set> --episodes <all> --source <ass|sdh> [--subtitle-track N] --out ~/fukikae-ws/<set>/labels-auto --report`, then the identity command above.
