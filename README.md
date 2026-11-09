# fukikae

fukikae automatically dubs foreign-language TV into English. The pipeline extracts audio, separates speech from the background, transcribes, diarizes and builds a registry of recurring characters, then translates, speaks each line with the character's voice card, fits it to its slot, mixes and muxes a dubbed episode.

## Commands

```bash
uv sync                      # install
uv run pytest                # tests (gpu and media tests are skipped by default)
uv run ruff check && uv run ruff format --check
uv run pyright
mise -C . exec -- ffprobe -version   # ffmpeg comes from mise, not the host
```

## Stages and artifacts

`fukikae run <workspace> <episode>` runs, in order: extract, separate, language, asr, diarize, characters, tag (when `[tag] enabled`), translate, tts, timefit and mix. Each writes under `<workspace>/.fukikae/episodes/<episode>/`:

- `06-translate/cues.<target>.json`: the lines to dub (translated on the ASR path, the subtitle text on the subtitle path).
- `07-tts/<cue>.wav` and `plan.json`: one render per spoken cue, silence trimmed, rate-fitted up to `[timefit] max_speedup`.
- `08-timefit/<cue>.wav` and `report.json`: each render placed in its slot. What the rate fit left is time-stretched (ffmpeg `atempo`, pitch kept; rate and stretch together never exceed `max_speedup`), then up to `borrow_silence_ms` of silence is borrowed after the slot and before it. A line that still doesn't fit plays whole past its slot and is listed under `over`; nothing is cut. `summary` counts stretched, borrowed and over lines.
- `09-mix/dub.<target>.wav`, `dub.<target>.ass` and `report.json`: the dub track and its subtitles (`Speaker: line`). The original vocals are removed over each dubbed line's whole slot and wherever its audio plays, and kept elsewhere; a line that is not dubbed (errored, untranslated, skipped) keeps its voice even where a dubbed slot overlaps it, and over songs and laughter the original vocals stay untouched. Songs come from a theme-song detector on the separation stems (openings and endings: loud music with near-constant, pitched vocals in the first 6 or last 4 minutes; every candidate window's scores are in the report), subtitle cues the filter marked as songs, and the tag stage's singing spans; a translated lyric inside a song is not placed, and `dropped` says which source caught it. The dub's level follows the original dialogue's, and the mix's EBU R128 integrated loudness the original mix's, by one linear gain and a true-peak limiter at -1 dBTP, so the loudness range is unchanged (`loudness` in the report holds integrated loudness, range and true peak of the original, the mix before and after).
- `out/<episode>.fukikae.mkv`: the video copied, the dub as the first and default audio track, every original audio track after it, the dub's ASS subtitles first and default, the original subtitle tracks after them (neither default nor forced), language tags set.

`[timefit]` takes `max_speedup` (1.25), `max_slowdown` (0.9; nothing is slowed today) and `borrow_silence_ms` (400). `[mix]` takes `keep_original_vocals` (true), `duck_db` (unset: removed; e.g. -18 keeps a quiet bleed of the original voice under the dub), `fade_ms` (40), `match_dialogue` (true), `protect_bgm` (false) and `detect_songs` (true); `[mix.songs]` holds the detector's thresholds and window lengths (`fukikae.songs.SongSettings`; `scripts/song_scores.py` scores episodes with them).

Two deliberate departures from the foundation spec (ruling CD11): the tag stage's BGM spans do not keep the original vocals by default, because SenseVoice marks BGM under ordinary dialogue (on Fairy Tail E11, 10 dialogue lines sat in BGM spans outside any song, which would have played the dub over the untouched Japanese line); and the original vocals are cleared over a dubbed line's whole slot, not only where the dub plays, because a dub shorter than its slot would otherwise let the original line resume mid-sentence.

To listen, open `out/<episode>.fukikae.mkv` in any player (mpv switches audio with `#` and subtitles with `j`), or play `09-mix/dub.<target>.wav` against `00-extract/audio.wav`.

## Profiles

A workspace's `fukikae.toml` picks a tested set of identity settings with `profile = "anime" | "live-action"`; without it the `default` profile applies (WeSpeaker embeddings, the centroid matcher at 0.6, Demucs, pyannote). Precedence, highest first: the workspace's `[x.pair.P]`, its `[x.lang.L]`, its `[x]`, then the profile's `[x.pair.P]`, `[x.lang.L]` and `[x]`, then the built-in fallbacks; a layer that names a different backend drops the options of the layers below it. A workspace `[embed]` table that sets options must name its `backend`. Measured results and held-out verdicts are in `docs/results/2026-11-07-identity-profiles.md` for anime and default, and in `docs/results/2026-11-09-live-action-ground-truth.md` for live-action (superseding the 2026-11-07 doc's live-action section).

- `anime`: tuned on Fairy Tail. The best setting is the joint matcher with episode cannot-link on at 0.6. Joint resolves a whole season at once, so it runs as `fukikae registry resolve <workspace> --episodes S01E01,S01E02,...` after the episodes are diarized, reading the profile's `[resolve]` table (`matcher`, `threshold`, matcher options such as `episode_cannot_link`); flags on the command override it, and a workspace's own `[resolve]` overrides the profile's. The per-episode characters stage keeps centroid.
- `live-action`: picked on live-action ground truth in English, German, Spanish, Japanese and Korean (`docs/results/2026-11-09-live-action-ground-truth.md`). It lowers pyannote's `clustering_threshold` from 0.6 to 0.5, which splits the Spanish dub's leads, and keeps the centroid matcher at 0.6 in the characters stage. Its `[resolve]` is joint at 0.6 with `episode_cannot_link = true`, the joint setting that tied with centroid. The German dub's two leads stay on one id at every setting measured.

Anime fails the 90% held-out bar on Fairy Tail E14 (ja 0.55, dub 0.60), and so does default on the same tracks (ja 0.55, dub 0.60). Live-action, held out once after its pick: it passes BBT S01E03 in English and Spanish (4 of 4, no merge) and Bazinga's The Office S01E05, and fails BBT S01E03 in German (0.80, Leonard and Sheldon on one id), Bazinga's TBBT S01E05 (0.67: Howard at 0.52, and Raj, who has no id from the tuning episodes) and Grid E05 (0.80, no merge). Kaze, Kaoru's held-out E11 is pending. Details: `docs/results/2026-11-09-live-action-ground-truth.md`.

The live-action profile's `clustering_threshold = 0.5` is part of the diarize stage's cache key, so an existing live-action workspace re-runs diarize, and the stages after it, on its next run.

The default embedder is now WeSpeaker (it was ECAPA), so a workspace with no `profile` re-runs diarize on its next run.

Changing the embedder (a different `[embed]` backend or model, a profile switch, or the WeSpeaker default above for a workspace with no `profile`) leaves the registry's exemplars and centroids in the old embedding space, and matching refuses them. Run `fukikae registry reembed <workspace>` first (`--dry-run` reports counts per dimension): it embeds every exemplar again from its episode's existing diarize spans with the configured embedder, recomputes the centroids and drops projections of the old dimension, keeping names, locks and voices. Then re-run the pipeline. A locked actor's centroid, and that of an actor with no exemplars left, is never recomputed from exemplars: it is kept as stored when the dimension is unchanged, and otherwise carried into the new space by a ridge linear map fit on the old and new exemplar pairs, an approximation, since two embedders' spaces are not linearly related, so check a mapped actor's next matches or re-lock it after a run (reembed refuses, naming those actors, when there are fewer than 2 x the old dimension pairs; unlock them first and lock again afterwards). Run it before diarize re-runs: the exemplars' cluster labels refer to the existing diarization, and reembed refuses an episode whose diarization has changed.

## Upgrade notes

Stage cache keys and the clip-set and identity embedding caches now include the resolved model of each backend (the model file or name, plus the Hugging Face revision when known), so a changed default model re-runs the stage instead of reusing old output. The first run after this change re-runs separate, language (when it detects), asr and diarize once in each existing workspace, and `eval identity` recomputes its cached embeddings. The Hugging Face revision in a key is read from the local cache, so a fresh cache (or a `refs/main` that moved) changes a stage's key once, and the first episode run afterwards re-runs that stage once. The CAM++ default is now the zh_en 3D-Speaker model; a workspace using `campplus` needs `fukikae registry reembed` as for any embedder change.

## Ground-truth tools

Commands that build or check speaker ground truth for `eval identity` (`docs/results/2026-11-09-live-action-ground-truth.md` shows them in use):

- `fukikae eval subalign`: finds the offset and linear drift from one subtitle file to a reference subtitle file (by text) or to a workspace's detected speech (`--speech`), and writes the file retimed; it refuses a capture with a cut or a jump unless `--force`.
- `fukikae eval transcript-align`: names the cues of a timed subtitle file from an untimed, speaker-named transcript (bigbangtrans pages), written as named ASS for `--gt ass`.
- `fukikae eval colour-map`: maps each caption colour to a character from the named lines of all the given episodes and writes `labels-colours.yaml`, which the `colour` reader uses.
- `fukikae eval bazinga-labels`: turns Bazinga's word-level speaker annotations into named ASS, keeping confident words only, with `--time-scale` for audio slowed to film speed.
- `fukikae eval embedders`: scores embedders by nearest-centroid accuracy on a folder-per-character clip set and writes one ledger row per embedder.

## Results ledger

`fukikae eval asr`, `fukikae eval identity` and `fukikae eval embedders` append one row per run and split (tune or holdout) to a single user-level JSONL file, `$XDG_DATA_HOME/fukikae/results.jsonl` (`~/.local/share/fukikae/results.jsonl` by default); `FUKIKAE_RESULTS` or `--ledger` points them elsewhere, `--no-ledger` skips the write, and `--set` and `--tag` name the test set and the run. Each row records the backend, model and revision, a hash of the options, the language, the genre and title from the workspace's `[series]` table (`title` defaults to the workspace directory's name, `genre` to the profile's name; `tmdb`, `tvdb` and `anidb` are accepted for later use), the episodes, the metrics and fukikae's commit. `fukikae eval report` renders the ledger as a matrix, one row per stage, test set, split and configuration and one column per `language/genre`, with `--stage`, `--backend`, `--set`, `--split` and `--tag` filters, `--metric` for a metric other than the headline one, `--format md`, and `--export PATH` to write the filtered rows as JSONL. A line it can't read (a write cut short, a row from a newer schema) is skipped with a warning naming the line.

An explicit `[series] title` also names the series that `fukikae registry adapt` stores the LDA projection under (the older top-level `series = "Name"` still works); without one the key is `default`, never the directory-name default of the title. When a workspace that has a projection stored under `default` gains a title, the characters stage keeps using that projection and warns, naming both keys, until `registry adapt` fits one under the title.

Heavy backends are optional extras: `fukikae[demucs]`, `[whisper]`, `[qwen]`, `[pyannote]`, `[sherpa]`, `[wavlm]`, `[dialogue]`.

License: PolyForm Shield 1.0.0.
