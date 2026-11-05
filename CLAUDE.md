# CLAUDE.md

doppio: Python CLI that dubs foreign-language TV into English. Spec and plan are in `docs/superpowers/`.

## Commands
- `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`. uv only, never pip.
- Markers: `gpu`, `media` (need `DOPPIO_TEST_MEDIA`; deselected by default), `ffmpeg` (needs only the binary), `slow`.
- ffmpeg/ffprobe come from mise (`.mise/config.toml`): `mise exec -- ffprobe -version`. Call them with argument lists.

## Rules
- Model code lives only in `src/doppio/backends/<kind>/`; heavy backends are optional extras.
- Source media is read-only; work in the workspace.
- Conventional Commits.
