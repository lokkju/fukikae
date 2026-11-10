"""Exceptions the CLI reports as a one-line error instead of a traceback."""


def user_errors() -> tuple[type[Exception], ...]:
    """Errors caused by input, configuration or the machine, not by a bug in fukikae."""
    from fukikae.backends.loader import BackendUnavailable
    from fukikae.backends.tts.remote import BackendError
    from fukikae.cards.card import CardError
    from fukikae.config import ConfigError
    from fukikae.gpu import VramTimeout
    from fukikae.gpulock import GpuBusy
    from fukikae.media import MediaError
    from fukikae.translation import TranslateBackendError

    return (
        FileNotFoundError,
        ValueError,
        MediaError,
        ConfigError,
        BackendUnavailable,
        VramTimeout,
        CardError,
        # the GPU held by another job, an unreachable translator, a TTS worker that failed:
        # the machine's state or an endpoint, not a bug in fukikae (C6)
        GpuBusy,
        TranslateBackendError,
        BackendError,
    )
