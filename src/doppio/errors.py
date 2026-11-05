"""Exceptions the CLI reports as a one-line error instead of a traceback."""


def user_errors() -> tuple[type[Exception], ...]:
    """Errors caused by input, configuration or the machine, not by a bug in doppio."""
    from doppio.backends.loader import BackendUnavailable
    from doppio.config import ConfigError
    from doppio.gpu import VramTimeout
    from doppio.media import MediaError

    return (FileNotFoundError, ValueError, MediaError, ConfigError, BackendUnavailable, VramTimeout)
