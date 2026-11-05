"""Fake backends for loader tests."""

from typing import Any, ClassVar, Literal


class FakeTranscriber:
    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "asr"
    languages: ClassVar[frozenset[str] | Literal["*"]] = frozenset({"ko", "ja"})
    devices: ClassVar[frozenset[str]] = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    supports_lid: ClassVar[bool] = False

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options = options
        self.device = device

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class AnyLanguage(FakeTranscriber):
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
