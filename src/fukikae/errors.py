"""Exceptions the CLI reports as a one-line error instead of a traceback."""


def user_errors() -> tuple[type[Exception], ...]:
    """Errors caused by input, configuration or the machine, not by a bug in fukikae."""
    from fukikae.backends.loader import BackendUnavailable
    from fukikae.cards.card import CardError
    from fukikae.config import ConfigError
    from fukikae.gpu import VramTimeout
    from fukikae.media import MediaError

    return (
        FileNotFoundError,
        ValueError,
        MediaError,
        ConfigError,
        BackendUnavailable,
        VramTimeout,
        CardError,
    )
