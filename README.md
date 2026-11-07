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

A workspace's `fukikae.toml` picks a tested set of identity settings with `profile = "anime" | "live-action"`; without it the `default` profile applies (WeSpeaker embeddings, the centroid matcher at 0.6, Demucs, pyannote). Precedence: the workspace's own tables, then per-language overrides, then the profile, then the built-in fallbacks. Measured results and held-out verdicts are in `docs/results/2026-11-07-identity-profiles.md`.

- `anime`: tuned on Fairy Tail. The best setting is the joint matcher with episode cannot-link on at 0.6, which resolves a whole season at once and so runs as `fukikae registry resolve <workspace> --episodes S01E01,S01E02,...` after the episodes are diarized; the per-episode characters stage keeps centroid.
- `live-action`: provisional, written from one pair of Grid episodes with two recurring characters. Its best setting is joint with `--matcher-option episode_cannot_link=false`, again through `registry resolve`.

Heavy backends are optional extras: `fukikae[demucs]`, `[whisper]`, `[qwen]`, `[pyannote]`.

License: PolyForm Shield 1.0.0.
