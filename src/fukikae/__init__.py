"""fukikae: automatic English dubbing for foreign-language TV."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fukikae")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
