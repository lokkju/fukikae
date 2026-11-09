# fukikae

fukikae automatically dubs foreign-language TV into English. The pipeline extracts audio, separates speech from the background, transcribes, diarizes and builds a registry of recurring characters; translation, synthesis and mixing follow in later stages.

## Commands

```bash
uv sync                      # install
uv run pytest                # tests (gpu and media tests are skipped by default)
uv run ruff check && uv run ruff format --check
uv run pyright
mise -C . exec -- ffprobe -version   # ffmpeg comes from mise, not the host
```

## Profiles

A workspace's `fukikae.toml` picks a tested set of identity settings with `profile = "anime" | "live-action"`; without it the `default` profile applies (WeSpeaker embeddings, the centroid matcher at 0.6, Demucs, pyannote). Precedence, highest first: the workspace's `[x.pair.P]`, its `[x.lang.L]`, its `[x]`, then the profile's `[x.pair.P]`, `[x.lang.L]` and `[x]`, then the built-in fallbacks; a layer that names a different backend drops the options of the layers below it. A workspace `[embed]` table that sets options must name its `backend`. Measured results and held-out verdicts are in `docs/results/2026-11-07-identity-profiles.md`.

- `anime`: tuned on Fairy Tail. The best setting is the joint matcher with episode cannot-link on at 0.6. Joint resolves a whole season at once, so it runs as `fukikae registry resolve <workspace> --episodes S01E01,S01E02,...` after the episodes are diarized, reading the profile's `[resolve]` table (`matcher`, `threshold`, matcher options such as `episode_cannot_link`); flags on the command override it, and a workspace's own `[resolve]` overrides the profile's. The per-episode characters stage keeps centroid.
- `live-action`: picked on live-action ground truth in English, German, Spanish, Japanese and Korean (`docs/results/2026-11-09-live-action-ground-truth.md`). It lowers pyannote's `clustering_threshold` from 0.6 to 0.5, which splits the Spanish dub's leads, and keeps the centroid matcher at 0.6 in the characters stage. Its `[resolve]` is joint at 0.6 with `episode_cannot_link = true`, the joint setting that tied with centroid. The German dub's two leads stay on one id at every setting measured.

None of the three profiles passes the 90% held-out bar: anime fails on Fairy Tail E14 (ja 0.55, dub 0.60), the earlier provisional live-action setting failed on Grid E05 (0.80; the new pick's held-out verdicts are in its results doc), and default fails on the same E14 tracks (ja 0.55, dub 0.60).

The live-action profile's `clustering_threshold = 0.5` is part of the diarize stage's cache key, so an existing live-action workspace re-runs diarize, and the stages after it, on its next run.

The default embedder is now WeSpeaker (it was ECAPA), so a workspace with no `profile` re-runs diarize on its next run.

Changing the embedder (a different `[embed]` backend or model, a profile switch, or the WeSpeaker default above for a workspace with no `profile`) leaves the registry's exemplars and centroids in the old embedding space, and matching refuses them. Run `fukikae registry reembed <workspace>` first (`--dry-run` reports counts per dimension): it embeds every exemplar again from its episode's existing diarize spans with the configured embedder, recomputes the centroids and drops projections of the old dimension, keeping names, locks and voices. Then re-run the pipeline. A locked actor's centroid, and that of an actor with no exemplars left, is never recomputed from exemplars: it is kept as stored when the dimension is unchanged, and otherwise carried into the new space by a ridge linear map fit on the old and new exemplar pairs, an approximation, since two embedders' spaces are not linearly related, so check a mapped actor's next matches or re-lock it after a run (reembed refuses, naming those actors, when there are fewer than 2 x the old dimension pairs; unlock them first and lock again afterwards). Run it before diarize re-runs: the exemplars' cluster labels refer to the existing diarization, and reembed refuses an episode whose diarization has changed.

## Upgrade notes

Stage cache keys and the clip-set and identity embedding caches now include the resolved model of each backend (the model file or name, plus the Hugging Face revision when known), so a changed default model re-runs the stage instead of reusing old output. The first run after this change re-runs separate, language (when it detects), asr and diarize once in each existing workspace, and `eval identity` recomputes its cached embeddings. The CAM++ default is now the zh_en 3D-Speaker model; a workspace using `campplus` needs `fukikae registry reembed` as for any embedder change.

## Results ledger

`fukikae eval asr` and `fukikae eval identity` append one row per run and split (tune or holdout) to a single user-level JSONL file, `$XDG_DATA_HOME/fukikae/results.jsonl` (`~/.local/share/fukikae/results.jsonl` by default); `FUKIKAE_RESULTS` or `--ledger` points them elsewhere, `--no-ledger` skips the write, and `--set` and `--tag` name the test set and the run. Each row records the backend, model and revision, a hash of the options, the language, the genre and title from the workspace's `[series]` table (`title` defaults to the workspace directory's name, `genre` to the profile's name; `tmdb`, `tvdb` and `anidb` are accepted for later use), the episodes, the metrics and fukikae's commit. `fukikae eval report` renders the ledger as a matrix, one row per stage, test set, split and configuration and one column per `language/genre`, with `--stage`, `--backend`, `--set`, `--split` and `--tag` filters, `--metric` for a metric other than the headline one, `--format md`, and `--export PATH` to write the filtered rows as JSONL. A line it can't read (a write cut short, a row from a newer schema) is skipped with a warning naming the line.

An explicit `[series] title` also names the series that `fukikae registry adapt` stores the LDA projection under (the older top-level `series = "Name"` still works); without one the key is `default`, never the directory-name default of the title. When a workspace that has a projection stored under `default` gains a title, the characters stage keeps using that projection and warns, naming both keys, until `registry adapt` fits one under the title.

Heavy backends are optional extras: `fukikae[demucs]`, `[whisper]`, `[qwen]`, `[pyannote]`, `[sherpa]`, `[wavlm]`, `[dialogue]`.

License: PolyForm Shield 1.0.0.
