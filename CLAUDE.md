# CLAUDE.md

fukikae: Python CLI that dubs foreign-language TV into English. Spec and plan are in `docs/superpowers/`.

## Commands
- `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`. uv only, never pip.
- Markers: `gpu`, `media` (need `FUKIKAE_TEST_MEDIA`; deselected by default), `ffmpeg` (needs only the binary), `slow`.
- ffmpeg/ffprobe come from mise (`.mise/config.toml`): `mise exec -- ffprobe -version`. Call them with argument lists.

## Rules
- Model code lives only in `src/fukikae/backends/<kind>/`; heavy backends are optional extras.
- Source media is read-only; work in the workspace.
- Conventional Commits.
- Identity profiles live in `src/fukikae/profiles/<name>.toml`, chosen by `profile = "anime" | "live-action" | "default"` in the workspace's `fukikae.toml`. Every table in a profile names its `backend` (`matcher` for `[characters]`). Joint is not a stage matcher: `fukikae registry resolve` runs it and reads the profile's `[resolve]` table (`matcher`, `threshold`, matcher options; the selector is `matcher`, so `[resolve]` and `[characters]` are exempt from the `backend` rule); CLI flags override it. A profile's numbers come from tuning episodes only; see `docs/results/2026-11-07-identity-profiles.md` (anime, default) and `docs/results/2026-11-09-live-action-ground-truth.md` (live-action, which supersedes the 2026-11-07 live-action pick).
- Changing the embedder (backend, model, profile, or the WeSpeaker default for a workspace with no profile) means `fukikae registry reembed <workspace>` before diarize re-runs, then re-running the pipeline. reembed never recomputes a locked or exemplar-less centroid (Ruling R20): kept as stored at the same dimension, mapped by a ridge fit on exemplar pairs across a dimension change, refused with too few pairs. Dimension-mismatch errors point there; never tell a user to re-run diarize and rematch, which fails against the old-dimension registry.
- Voice cards (`src/fukikae/cards/`): a card's id hashes what it sounds like, and its defaults are frozen forever (Ruling CD1: a changed default is a new field, never a new value). Automatic assignment (`by = "auto"`, the tts stage via `cards.auto.ensure_assignments`) only fills characters with no card; the registry refuses it, and restock and imports, over a locked or operator-chosen card (CD2, CD3, CD9). New characters get a clone card of `[tts] clone_backend` (IndexTTS2 in every profile, D24) from `cards.select` references, or a stock voice under 6 s of clean speech; `[tts] backend` stays the stock and extras backend.
- Cloning backends run as workers in their own uv projects (`workers/<name>`, `.venvs/<name>`, CD13); every GPU job holds the per-GPU flock (`fukikae.gpulock`, CD15). Nothing downloads at run time: models load from the local cache, `fukikae backends setup <name>` is the only fetch, and telemetry is off process-wide (CD24). A new backend declares a capability only with A/B evidence in the ledger (CD20).
- The subtitle path (`[translate] source = "auto"`) reads only a dialogue subtitle track (`media.dialogue_track`, CD25): never signs/songs/forced by title, never a track too sparse for dialogue; a forced flag on a full track is ignored.
