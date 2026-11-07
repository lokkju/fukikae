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
- Identity profiles live in `src/fukikae/profiles/<name>.toml`, chosen by `profile = "anime" | "live-action" | "default"` in the workspace's `fukikae.toml`. Every table in a profile names its `backend` (`matcher` for `[characters]`). Joint is not a stage matcher: `fukikae registry resolve` runs it and reads the profile's `[resolve]` table (`matcher`, `threshold`, matcher options; the selector is `matcher`, so `[resolve]` and `[characters]` are exempt from the `backend` rule); CLI flags override it. A profile's numbers come from tuning episodes only; see `docs/results/2026-11-07-identity-profiles.md`.
