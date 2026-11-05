from pathlib import Path

from doppio.workspace import Workspace


def test_episode_id_from_awkward_path(tmp_path):
    p = (
        tmp_path
        / "Chimera (2021)"
        / "Season 1"
        / "Chimera - S01E01 - A Strange Shadow WEBDL-1080p.mkv"
    )
    p.parent.mkdir(parents=True)
    p.touch()
    ws = Workspace(tmp_path / "ws")
    ep = ws.find_episode(str(p))
    assert ep.id == "S01E01"
    assert ep.source == p
    assert ep.work == tmp_path / "ws" / ".doppio" / "episodes" / "S01E01"
    assert ep.workspace_root == tmp_path / "ws"


def test_episode_id_slug_when_no_code(tmp_path):
    p = tmp_path / "It's a Movie (2020).mkv"
    p.touch()
    assert Workspace(tmp_path).find_episode(str(p)).id == "it-s-a-movie-2020"


def test_find_by_code_under_source(tmp_path):
    media = tmp_path / "media" / "Show (2021)" / "Season 1"
    media.mkdir(parents=True)
    f = media / "Show - S01E02 - x.mkv"
    f.touch()
    (media / "Show - S01E01 - y.mkv").touch()
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    (ws_root / "doppio.toml").write_text(f'source = "{tmp_path / "media"}"\n')
    ws = Workspace(ws_root)
    assert ws.find_episode("s01e02").source == f
    assert ws.find_episode("S01E02").id == "S01E02"


def test_paths(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.registry_path == tmp_path / ".doppio" / "registry" / "registry.sqlite"
    assert ws.work_dir("S01E01") == Path(tmp_path / ".doppio" / "episodes" / "S01E01")
