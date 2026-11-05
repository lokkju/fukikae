import shutil
import stat
import subprocess
from pathlib import Path

import pytest


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
