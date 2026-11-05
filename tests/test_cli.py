from typer.testing import CliRunner

from doppio import __version__
from doppio.cli import app


def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.stdout
