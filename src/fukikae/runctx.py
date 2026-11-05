"""Per-invocation settings that stages read (set by the CLI)."""

# Seconds a GPU stage waits for free VRAM; None waits forever, 0 fails at once (`--no-wait`).
vram_timeout_s: float | None = None
