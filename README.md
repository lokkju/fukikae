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

- `06-translate/cues.<target>.json`: the lines to dub (translated on the ASR path, the subtitle text on the subtitle path). With `[translate] source = "auto"` (every profile) the subtitle path reads target-language subtitles when there are dialogue subtitles: a `[subs] file` or `track`, a sidecar, or the embedded dialogue track; a track titled signs, songs or forced, or too sparse for dialogue (under 3 cues a minute, or 40% of the densest track), is skipped, a forced flag on a track that carries every line is not believed, and a plain track wins over CC/SDH (ruling CD25).
- `07-tts/<cue>.wav` and `plan.json`: one render per spoken cue, silence trimmed, rate-fitted up to `[timefit] max_speedup`.
- `08-timefit/<cue>.wav` and `report.json`: each render placed in its slot. What the rate fit left is time-stretched (ffmpeg `atempo`, pitch kept; rate and stretch together never exceed `max_speedup`), then up to `borrow_silence_ms` of silence is borrowed after the slot and before it. A line that still doesn't fit plays whole past its slot and is listed under `over`; nothing is cut. `summary` counts stretched, borrowed and over lines.
- `09-mix/dub.<target>.wav`, `dub.<target>.ass` and `report.json`: the dub track and its subtitles (`Speaker: line`). The original vocals are removed over each dubbed line's whole slot and wherever its audio plays, and kept elsewhere; a line that is not dubbed (errored, untranslated, skipped) keeps its voice even where a dubbed slot overlaps it, and over songs and laughter the original vocals stay untouched. Songs come from a theme-song detector on the separation stems (openings and endings: loud music with near-constant, pitched vocals in the first 6 or last 4 minutes; every candidate window's scores are in the report), subtitle cues the filter marked as songs, and the tag stage's singing spans; a translated lyric inside a song is not placed, and `dropped` says which source caught it. The dub's level follows the original dialogue's, and the mix's EBU R128 integrated loudness the original mix's, by one linear gain and a true-peak limiter at -1.5 dBTP (headroom for the AAC encode), so the loudness range is unchanged (`loudness` in the report holds integrated loudness, range and true peak of the original, the mix before and after, and the dub track in the mkv after encoding).
- `out/<episode>.fukikae.mkv`: the video copied, the dub as the first and default audio track, every original audio track after it, the dub's ASS subtitles first and default, the original subtitle tracks after them (neither default nor forced), language tags set.

`[timefit]` takes `max_speedup` (1.25), `max_slowdown` (0.9; nothing is slowed today) and `borrow_silence_ms` (400). `[mix]` takes `keep_original_vocals` (true), `duck_db` (unset: removed; e.g. -18 keeps a quiet bleed of the original voice under the dub), `fade_ms` (40), `match_dialogue` (true), `protect_bgm` (false), `detect_songs` (true) and `true_peak_db` (-1.5, the limiter's ceiling); `[mix.songs]` holds the detector's thresholds and window lengths (`fukikae.songs.SongSettings`; `scripts/song_scores.py` scores episodes with them).

Two deliberate departures from the foundation spec (ruling CD11): the tag stage's BGM spans do not keep the original vocals by default, because SenseVoice marks BGM under ordinary dialogue (on Fairy Tail E11, 10 dialogue lines sat in BGM spans outside any song, which would have played the dub over the untouched Japanese line); and the original vocals are cleared over a dubbed line's whole slot, not only where the dub plays, because a dub shorter than its slot would otherwise let the original line resume mid-sentence.

To listen, open `out/<episode>.fukikae.mkv` in any player (mpv switches audio with `#` and subtitles with `j`), or play `09-mix/dub.<target>.wav` against `00-extract/audio.wav`.

## Voices: cards, cloning by default, packs

Every recurring character speaks with a voice card per target language: a small TOML file in the workspace's card store (`.fukikae/cards/<id>.toml`, the id a hash of what the card sounds like), with its reference clips and fingerprint beside it. The registry records which card each character has, with a numbered history.

**Automatic cards.** The first time a character needs a voice in an episode and has no card for the target language, the tts stage gives it one, `auto` and unlocked, and logs a line to `07-tts/assignments.log`:

- With `[tts] clone_backend` set (every profile sets `indextts2`, the operator's pick after a blind listening test, D24), a clone card of that backend. Its reference clips are the character's cleanest turns on the separated vocals across the episodes the workspace has processed: no overlap with another speaker, 3 to 12 s each, an ASR transcript that isn't implausibly sparse for its length, outside theme songs and tagged singing, ranked by signal-to-noise with turns that emotion2vec scores non-neutral last (up to 3 clips, 20 s). When a chosen clip is scored non-neutral, the log and `voices show` say `references may carry emotion: <labels>`.
- A character with under 6 s of such speech gets a stock Kokoro voice from the pool matching its estimated gender instead, and the line says so: `character 7: 4.2 s of clean speech, under 6 s; stock voice am_puck`.
- Extras (unrecurring speakers) always get stock voices, one per speaker label.
- An automatic card never replaces a card that exists. To start over, assign a card by hand.

Set `clone_backend = ""` under `[tts]` in a workspace's `fukikae.toml` to give new characters stock voices only; `[tts] backend` (Kokoro) is the stock and extras backend either way. Qwen3-TTS and CosyVoice 3 also passed the listening test; a character can be given either with `voices assign` or `voices clone --backend`.

**Changing a voice.** `fukikae voices <command> <workspace> <character> ...`, where a character is its registry id or name:

- `list`, `show`, `history`: what each character has, the current card (`references may carry emotion` and the clone mode included), the assignment rows.
- `assign <card>`: a card id from the store, `stock:kokoro:<voice>`, or one character's card from a pack (`<pack>:<name>`); locked unless `--no-lock`.
- `clone --clip <wav> --transcript TEXT [--backend NAME]`: a clone from a clip you cut yourself, 3 to 30 s of the character alone, for a character whose automatic references are wrong (a cluster that merged two voices, say); assigned locked.
- `edit` (a copy of the card in `$EDITOR`, assigned locked), `derive --to <lang>` (the same voice for another target language), `revert [--to N]`, `lock`, `unlock`, `restock` (re-pins automatic stock cards after a Kokoro revision bump).
- `verify [--record]`: renders a fixed test script and compares its speaker embedding with the card's fingerprint.

Locked cards and cards an operator chose are never replaced by automatic assignment or by an import without `--replace`.

**Packs.** `fukikae voices export <workspace> -o <pack>[.tar.zst]` writes the named characters' current cards, clips and fingerprints to a pack (names and external ids, never registry ids); `fukikae voices import <workspace> <pack>` assigns them to another workspace's characters by external id, then by name, as `import:<title>`, unlocked, and verifies each fingerprinted card. A pack is untrusted input: it is read without extraction, every member checked against the layout and size limits, cards re-parsed and their ids recomputed, and a fingerprint can only name an allowlisted embedder that is already on disk, so a pack can never cause a download or load code.

**Models and privacy.** `fukikae backends setup <name>` is the only command that downloads: worker environments (`uv sync --locked`) and every pinned model, IndexTTS2's included (`fukikae backends setup indextts2` before the first run). Runs load models from the local cache only, with pyannote's, Hugging Face's and onnxruntime's telemetry turned off for the whole process and for every worker (ruling CD24). Ten older backends (ASR, alignment, diarization, dialogue separation and four embedders) still fetch an unpinned model on a cache miss; pinning them is open work.

## Cloning backends

Kokoro runs inside fukikae. It, the tag models (SenseVoice, emotion2vec) and the managed translator's GGUF load their pinned files from the local Hugging Face cache only; fetch them once with `fukikae backends setup kokoro` (and `sensevoice`, `emotion2vec`, `openai_compat`; `--workspace <dir>` fetches the model that workspace's config names). A run never downloads them, and fukikae turns off pyannote's, Hugging Face's and onnxruntime's telemetry for the whole process.

The cloning backends run out of process, each in an environment of its own, so their pinned dependencies never meet fukikae's: `workers/<name>/` is a separate uv project (its own Python and lock) and its environment lives in `.venvs/<name>`. fukikae starts the backend as a worker process speaking JSON lines (`fukikae.backends.tts.remote`), on one GPU only (`CUDA_VISIBLE_DEVICES`), and stops and reaps it when the stage ends. `fukikae doctor` lists each worker environment and the command that sets it up. IndexTTS2 (`[tts] backend = "indextts2"`, or a card on it) needs Python 3.11, which uv downloads:

```bash
fukikae backends setup indextts2
```

That runs `uv sync --locked` into `.venvs/indextts2`, records the lock's sha256 there (a worker whose environment came from another lock refuses to start and names this command), then downloads the model and its four auxiliary models at their pinned revisions into the Hugging Face cache. It is the only step that downloads: the worker runs offline, with an allowlisted environment that carries none of the secrets in fukikae's own. The environment takes about 8 GB, the models about 7 GB. IndexTTS2 clones from a card's reference clips and takes the source line's audio as a separate emotion reference, scaled by the card's `emotion_strength`; it has no duration control.

CosyVoice 3 (`cosyvoice3`, Fun-CosyVoice3-0.5B) needs Python 3.10:

```bash
fukikae backends setup cosyvoice3
```

CosyVoice and its Matcha-TTS submodule ship no package, so the setup step also downloads each one's code at a pinned commit into `.venvs/cosyvoice3/fukikae-code`, keeps only the package directory and refuses it unless its files hash to the pinned digest; the worker checks that digest again every time it loads. The environment takes about 5.5 GB, the model about 5 GB, and a run about 4 GB of VRAM. It clones from a card's reference clips: `[tts.cosyvoice3] mode = "auto"` (the default) gives the model the clips' transcripts when they are in the target language (`zero_shot`) and only their sound when they aren't (`cross_lingual`, CosyVoice's own advice for a foreign-language prompt); `mode` can force either. Every line is neutral: it takes no delivery direction, emotion audio, emotion vector or duration target (CosyVoice's instruct mode is not offered until an A/B shows it helps), and the card's `rate` is CosyVoice's speed. CosyVoice's own control tags (`[laughter]`, `[breath]`, `<strong>` and its phoneme tags) are stripped from the text before it reaches the model. It runs without a text normalizer: CosyVoice's wetext fetches its grammars unpinned at every start, so it isn't installed, and CosyVoice still spells out English numbers.

Qwen3-TTS (`qwen3_tts`, Qwen3-TTS-12Hz-1.7B-Base) runs on Python 3.12:

```bash
fukikae backends setup qwen3_tts
```

The `qwen-tts` package is pinned by version (and by hash in `workers/qwen3tts/uv.lock`) and checked again at every load; the model is pinned by revision. The environment takes about 7 GB, the model about 4.3 GB. It clones from a card's reference clips: `[tts.qwen3_tts] mode = "auto"` (the default) gives the model the clips' transcripts and speech codes (`icl`) when every clip has a transcript, and only the speaker embedding (`xvector`) otherwise; `mode` can force either. It takes no delivery input (no direction, emotion audio, emotion vector, speed or duration target), and the card's `seed`, `temperature` and `top_p` are its only voice keys. Only the clone model is integrated; Qwen3-TTS's VoiceDesign model (`source = "design"`) is not.

Each TTS backend takes its options and its device from its own table, `[tts.<backend>]` (for example `[tts.indextts2] fp16 = true`). Kokoro runs on the CPU and the cloning backends on the GPU unless that table's `device` says otherwise; the old stage-wide `[tts] device` applies only to a backend that declares no default. IndexTTS2 refuses the CPU (fp32, about 9 GB of RAM, minutes per line) unless `force_cpu = true` is in `[tts.indextts2]`.

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
