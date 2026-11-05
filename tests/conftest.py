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
    """A 5 s test-tone mkv (5.1 eng-dub-less 'jpn' track plus stereo 'eng') in a read-only dir."""
    made = tmp_path / "made.mkv"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
            "-filter_complex",
            "[0:a]aformat=channel_layouts=5.1[a0];[1:a]aformat=channel_layouts=stereo[a1]",
            "-map", "[a0]", "-map", "[a1]",
            "-c:a", "aac",
            "-metadata:s:a:0", "language=jpn",
            "-metadata:s:a:1", "language=eng",
            "-disposition:a:1", "default",
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
