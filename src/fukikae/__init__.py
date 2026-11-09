"""fukikae: automatic English dubbing for foreign-language TV."""

from fukikae.privacy import disable_telemetry

# First, before any model library loads (ruling CD24): third-party telemetry off.
disable_telemetry()

from importlib.metadata import PackageNotFoundError, version  # noqa: E402 - after the switches

try:
    __version__ = version("fukikae")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
