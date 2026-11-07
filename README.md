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
- `live-action`: provisional, written from one pair of Grid episodes with two recurring characters. Its `[resolve]` is joint at 0.6 with `episode_cannot_link = false`.

None of the three profiles passes the 90% held-out bar: anime fails on Fairy Tail E14 (ja 0.55, dub 0.60), live-action is provisional and fails on Grid E05 (0.80), and default fails on the same E14 tracks (ja 0.55, dub 0.60).

The default embedder is now WeSpeaker (it was ECAPA), so a workspace with no `profile` re-runs diarize on its next run.

Changing the embedder (a different `[embed]` backend or model, a profile switch, or the WeSpeaker default above for a workspace with no `profile`) leaves the registry's exemplars and centroids in the old embedding space, and matching refuses them. Run `fukikae registry reembed <workspace>` first (`--dry-run` reports counts per dimension): it embeds every exemplar again from its episode's existing diarize spans with the configured embedder, recomputes the centroids and drops projections of the old dimension, keeping names, locks and voices. Then re-run the pipeline. A locked actor's centroid, and that of an actor with no exemplars left, is never recomputed from exemplars: it is kept as stored when the dimension is unchanged, and otherwise carried into the new space by a ridge linear map fit on the old and new exemplar pairs, an approximation, since two embedders' spaces are not linearly related, so check a mapped actor's next matches or re-lock it after a run (reembed refuses, naming those actors, when there are fewer than 2 x the old dimension pairs; unlock them first and lock again afterwards). Run it before diarize re-runs: the exemplars' cluster labels refer to the existing diarization, and reembed refuses an episode whose diarization has changed.

Heavy backends are optional extras: `fukikae[demucs]`, `[whisper]`, `[qwen]`, `[pyannote]`, `[sherpa]`, `[wavlm]`, `[dialogue]`.

License: PolyForm Shield 1.0.0.
