import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Every test writes ledger rows to a temp file, never the operator's results.jsonl."""
    monkeypatch.setenv("FUKIKAE_RESULTS", str(tmp_path_factory.mktemp("ledger") / "results.jsonl"))


@pytest.fixture(autouse=True)
def _isolated_gpu_locks(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Every test takes GPU locks in a temp dir, never the operator's runtime dir."""
    monkeypatch.setenv("FUKIKAE_LOCK_DIR", str(tmp_path_factory.mktemp("gpulocks")))


@pytest.fixture(autouse=True)
def started_pids(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    """The GPU lock's leftover scan sees only the processes a test started and added here,
    so the suite doesn't fail while a real fukikae job runs on the machine."""
    from fukikae import gpulock

    pids: set[int] = set()
    monkeypatch.setattr(
        gpulock, "_proc_pids", lambda: sorted(p for p in pids if os.path.exists(f"/proc/{p}"))
    )
    return pids


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("ffmpeg") and not (
        shutil.which("ffmpeg") and shutil.which("ffprobe")
    ):
        pytest.skip("ffmpeg/ffprobe not on PATH")


@pytest.fixture
def readonly_copy(tmp_path: Path):
    """5 s mkv, read-only dir: video 0, eng stereo 880 Hz (1, default), jpn 5.1 440 Hz (2)."""
    made = tmp_path / "made.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
            "-filter_complex",
            "[1:a]aformat=channel_layouts=5.1[a0];[2:a]aformat=channel_layouts=stereo[a1]",
            "-map", "0:v", "-map", "[a1]", "-map", "[a0]",
            "-c:v", "mpeg4", "-c:a", "aac",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "language=jpn",
            "-disposition:a:0", "default", "-disposition:a:1", "0",
            str(made),
        ],
        check=True,
    )  # fmt: skip
    folder = tmp_path / "src" / "dir (x)"
    folder.mkdir(parents=True)
    target = folder / "it's a file.mkv"
    shutil.copy(made, target)
    mode = folder.stat().st_mode
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    yield target
    folder.chmod(mode)


@pytest.fixture(scope="session")
def clip_ep(tmp_path_factory: pytest.TempPathFactory):
    """The 90 s Korean clip, extracted and separated once per session (gpu tests only)."""
    import os

    media = os.environ.get("FUKIKAE_TEST_MEDIA")
    if not media:
        pytest.skip("FUKIKAE_TEST_MEDIA not set")
    from fukikae.config import Config
    from fukikae.stages.extract import ExtractStage
    from fukikae.stages.separate import SeparateStage
    from fukikae.workspace import Episode

    root = tmp_path_factory.mktemp("clipws")
    work = root / "episodes" / "e1"
    ep = Episode(
        source=Path(media).expanduser() / "S01E01-clip.mkv",
        id="e1",
        work=work,
        workspace_root=root,
    )
    cfg = Config(source_language="ko")
    ExtractStage().run(ep, cfg)
    SeparateStage().run(ep, cfg)
    return ep
