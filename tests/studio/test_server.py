"""`fukikae studio <ws>`: the loopback server, its token and host guards, the JSON API over
the season index and decisions, and audio serving (Task 17, server half)."""

from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae.config import load_config
from fukikae.registry.store import Registry
from fukikae.workspace import Workspace
from tests.studio.test_season import ws  # noqa: F401 (the season fixture workspace)

TOKEN = "test-token-0123456789abcdef"  # noqa: S105


class Client:
    def __init__(self, server) -> None:
        self.server = server
        self.port = server.server_address[1]

    def request(
        self,
        method: str,
        path: str,
        body: bytes | str | dict | None = None,
        *,
        token: str | None = TOKEN,
        cookie: str | None = None,
        host: str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = "application/json",
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            if host != "":
                conn.putheader("Host", host or f"127.0.0.1:{self.port}")
            if token is not None:
                conn.putheader("X-Fukikae-Token", token)
            if cookie is not None:
                conn.putheader("Cookie", cookie)
            data = b""
            if body is not None:
                data = (
                    json.dumps(body).encode() if isinstance(body, dict)
                    else body.encode() if isinstance(body, str) else body
                )  # fmt: skip
                if content_type:
                    conn.putheader("Content-Type", content_type)
            if method == "POST":
                conn.putheader("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                conn.putheader(k, v)
            conn.endheaders()
            if data:
                conn.send(data)
            r = conn.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
        finally:
            conn.close()

    def json(self, method: str, path: str, body=None, **kw) -> tuple[int, dict]:
        status, _, raw = self.request(method, path, body, **kw)
        return status, json.loads(raw) if raw else {}


@pytest.fixture
def studio(ws: Workspace) -> Iterator[Client]:  # noqa: F811
    from fukikae.studio.server import StudioServer

    server = StudioServer(ws, load_config(ws.root, user_config=None), port=0, token=TOKEN)
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    try:
        yield Client(server)
    finally:
        server.shutdown()
        server.server_close()
        t.join(5)


def _season(c: Client) -> dict:
    status, doc = c.json("GET", "/api/season")
    assert status == 200, doc
    return doc


def _entry(doc: dict, cid: int) -> dict:
    return next(x for x in doc["characters"] if x["id"] == cid)


def _reg_name(reg: Registry, cid: int) -> str | None:
    return next(c.name for c in reg.characters() if c.id == cid)


def _name(doc: dict, cid: int, name: str) -> dict:
    return {"op": "name", "character": cid, "name": name, "digest": _entry(doc, cid)["digest"]}


# --- binding -------------------------------------------------------------------------------


def test_binds_loopback_only(studio):
    assert studio.server.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.30.201", "example.com", "::"])  # noqa: S104
def test_non_loopback_bind_is_refused_without_the_flag(ws, host):  # noqa: F811
    from fukikae.studio.server import StudioServer, check_bind

    with pytest.raises(ValueError, match="loopback"):
        check_bind(host, allow_non_loopback=False)
    with pytest.raises(ValueError, match="loopback"):
        StudioServer(ws, load_config(ws.root, user_config=None), host=host, port=0, token=TOKEN)


def test_non_loopback_ipv4_needs_the_explicit_flag():
    from fukikae.studio.server import check_bind

    assert check_bind("localhost", allow_non_loopback=False) == "127.0.0.1"
    assert check_bind("127.0.0.1", allow_non_loopback=False) == "127.0.0.1"
    assert check_bind("192.168.30.201", allow_non_loopback=True) == "192.168.30.201"
    with pytest.raises(ValueError, match="IPv4"):
        check_bind("example.com", allow_non_loopback=True)


def test_a_fresh_token_per_server(ws):  # noqa: F811
    from fukikae.studio.server import StudioServer

    cfg = load_config(ws.root, user_config=None)
    a = StudioServer(ws, cfg, port=0)
    b = StudioServer(ws, cfg, port=0)
    try:
        assert a.token != b.token and len(a.token) >= 32
        assert a.url().startswith(f"http://127.0.0.1:{a.server_address[1]}/?t=")
        assert a.url().endswith(a.token)
    finally:
        a.server_close()
        b.server_close()


# --- the token and host guards -------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/api/season", "/api/characters/1", "/static/index.html"])
def test_requests_without_the_token_are_refused(studio, path):
    assert studio.request("GET", path, token=None)[0] == 403
    assert studio.request("GET", path, token="wrong")[0] == 403  # noqa: S106


def test_page_load_with_the_token_sets_a_strict_cookie_and_drops_it_from_the_url(studio):
    status, headers, _ = studio.request("GET", f"/?t={TOKEN}", token=None)
    assert status == 303 and headers["location"] == "/"
    cookie = headers["set-cookie"]
    name = f"fukikae_studio_{studio.port}"
    assert cookie.startswith(f"{name}={TOKEN}")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
    status, headers, page = studio.request("GET", "/", token=None, cookie=f"{name}={TOKEN}")
    assert status == 200 and headers["content-type"].startswith("text/html")
    assert b"/api/season" in page
    assert "default-src 'self'" in headers["content-security-policy"]


def test_cookie_authorizes_reads_but_not_writes(studio):
    cookie = f"fukikae_studio_{studio.port}={TOKEN}"
    assert studio.request("GET", "/api/season", token=None, cookie=cookie)[0] == 200
    doc = _season(studio)
    status, _ = studio.json(
        "POST", "/api/decisions", _name(doc, 3, "Erza"), token=None, cookie=cookie
    )
    assert status == 403


def test_token_in_the_query_is_only_taken_on_the_page_load(studio):
    assert studio.request("GET", f"/api/season?t={TOKEN}", token=None)[0] == 403


def test_another_ports_cookie_is_refused(studio):
    cookie = f"fukikae_studio_{studio.port + 1}={TOKEN}"
    assert studio.request("GET", "/api/season", token=None, cookie=cookie)[0] == 403


@pytest.mark.parametrize("host", ["evil.example:{port}", "127.0.0.1:1", "", "192.168.1.5:{port}"])
def test_foreign_host_header_is_refused(studio, host):
    status, _, _ = studio.request("GET", "/api/season", host=host.format(port=studio.port))
    assert status == 403


def test_localhost_host_header_is_accepted(studio):
    assert studio.request("GET", "/api/season", host=f"localhost:{studio.port}")[0] == 200


def test_foreign_origin_is_refused(studio):
    doc = _season(studio)
    status, _ = studio.json(
        "POST", "/api/decisions", _name(doc, 3, "Erza"),
        headers={"Origin": "https://evil.example"},
    )  # fmt: skip
    assert status == 403
    status, _ = studio.json(
        "POST", "/api/decisions/preview", _name(doc, 3, "Erza"),
        headers={"Origin": f"http://127.0.0.1:{studio.port}"},
    )  # fmt: skip
    assert status == 200


def test_request_log_never_prints_the_token(studio, capsys):
    studio.request("GET", f"/?t={TOKEN}", token=None)
    assert TOKEN not in capsys.readouterr().err


# --- request bodies ------------------------------------------------------------------------


def test_body_over_one_megabyte_is_413(studio):
    status, _, _ = studio.request("POST", "/api/decisions", b"x" * (1024 * 1024 + 1))
    assert status == 413


def test_non_json_post_is_415(studio):
    status, _, _ = studio.request(
        "POST", "/api/decisions", "op=name", content_type="application/x-www-form-urlencoded"
    )
    assert status == 415


def test_malformed_decisions_are_400(studio):
    assert studio.request("POST", "/api/decisions", "{not json")[0] == 400
    status, body = studio.json("POST", "/api/decisions", {"op": "name", "character": "x"})
    assert status == 400 and body["error"] == "invalid"


# --- reading -------------------------------------------------------------------------------


def test_season_endpoint_serves_the_index(studio, ws):  # noqa: F811
    doc = _season(studio)
    assert doc["schema_version"] == 1 and doc["workspace"] == str(ws.root)
    assert sorted(c["id"] for c in doc["characters"]) == [1, 2, 3, 4, 5]
    assert (ws.root / "studio" / "season.json").is_file()


def test_character_endpoint(studio):
    status, entry = studio.json("GET", "/api/characters/1")
    assert status == 200 and entry["name"] == "Lucy" and entry["samples"]
    assert studio.json("GET", "/api/characters/99")[0] == 404
    assert studio.json("GET", "/api/characters/abc")[0] == 404


def test_unknown_routes_are_404(studio):
    assert studio.request("GET", "/api/nothing")[0] == 404
    assert studio.request("POST", "/api/season", {})[0] == 405


# --- decisions -----------------------------------------------------------------------------


def test_preview_says_what_reruns_and_writes_nothing(studio, ws):  # noqa: F811
    doc = _season(studio)
    status, body = studio.json("POST", "/api/decisions/preview", _name(doc, 1, "Lucia"))
    assert status == 200, body
    assert body["lines"] and all(isinstance(x, str) for x in body["lines"])
    assert body["stale"] is None
    assert all(set(i) == {"episode", "stage"} for i in body["invalidations"])
    with Registry(ws.registry_path) as reg:
        assert _reg_name(reg, 1) == "Lucy"
    assert not (ws.root / "studio" / "decisions.jsonl").exists()


def test_preview_flags_a_stale_decision(studio, ws):  # noqa: F811
    doc = _season(studio)
    with Registry(ws.registry_path) as reg:
        reg.rename_character(3, "Someone")
    status, body = studio.json("POST", "/api/decisions/preview", _name(doc, 3, "Erza"))
    assert status == 200 and "character 3" in body["stale"]


def test_decision_post_applies_and_returns_the_rebuilt_character(studio, ws):  # noqa: F811
    doc = _season(studio)
    status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 200, body
    assert body["character"]["id"] == 3 and body["character"]["name"] == "Erza"
    assert body["character"]["digest"] != _entry(doc, 3)["digest"]
    assert body["lines"]
    with Registry(ws.registry_path) as reg:
        assert _reg_name(reg, 3) == "Erza"
    assert (ws.root / "studio" / "decisions.jsonl").is_file()
    assert _entry(_season(studio), 3)["name"] == "Erza"  # the index was rebuilt


def test_stale_decision_is_409_with_the_difference(studio, ws):  # noqa: F811
    doc = _season(studio)
    with Registry(ws.registry_path) as reg:
        reg.rename_character(3, "Someone")
    status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 409 and body["error"] == "stale"
    assert "character 3" in body["message"] and "Someone" in body["message"]
    with Registry(ws.registry_path) as reg:
        assert _reg_name(reg, 3) == "Someone"


def test_decision_without_a_digest_is_422_unchecked(studio):
    status, body = studio.json("POST", "/api/decisions", {"op": "name", "character": 3,
                                                          "name": "Erza"})  # fmt: skip
    assert status == 422 and body["error"] == "unchecked"


def test_registry_refusal_is_422(studio, ws):  # noqa: F811
    from fukikae.studio.apply import stamp
    from fukikae.studio.model import parse_decision

    with Registry(ws.registry_path) as reg:  # a merge of one actor with itself
        d = stamp(reg, parse_decision({"op": "merge", "keep": 1, "drop": 1}))
    d = d.model_dump(mode="json")
    status, body = studio.json("POST", "/api/decisions", d)
    assert status == 422 and body["error"] == "refused"


def test_accept_is_not_taken_by_the_identity_endpoint(studio):
    d = {"op": "accept", "character": 1, "language": "en", "card_id": "x", "draft": "1.en",
         "version": 1, "digest": "0" * 64}  # fmt: skip
    for path in ("/api/decisions", "/api/decisions/preview"):
        status, body = studio.json("POST", path, d)
        assert status == 422 and body["error"] == "not-identity"


def test_busy_studio_lock_is_423(studio, ws):  # noqa: F811
    from fukikae.studio.apply import studio_lock

    doc = _season(studio)
    with studio_lock(ws, "a `fukikae registry rename`"):
        status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 423 and body["error"] == "busy"
    assert "fukikae registry rename" in body["message"]


def test_unfinished_records_are_409_pointing_at_resume(studio, ws, monkeypatch):  # noqa: F811
    from fukikae.studio import server
    from fukikae.studio.apply import UnfinishedRecords

    def unfinished(*_a, **_kw):
        raise UnfinishedRecords("the merge logged at t is incomplete; run `fukikae studio resume`")

    monkeypatch.setattr(server, "apply_decision", unfinished)
    doc = _season(studio)
    status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 409 and body["error"] == "unfinished"
    assert body["resume"] == f"fukikae studio resume {ws.root}"


def test_incomplete_decision_is_409_and_says_the_registry_part_landed(studio, ws, monkeypatch):  # noqa: F811
    from fukikae.studio import server
    from fukikae.studio.apply import IncompleteDecision

    def incomplete(*_a, **_kw):
        raise IncompleteDecision("the name landed; stale marks left")

    monkeypatch.setattr(server, "apply_decision", incomplete)
    doc = _season(studio)
    status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 409 and body["error"] == "incomplete" and body["landed"] is True
    assert body["resume"] == f"fukikae studio resume {ws.root}"


# --- audio ---------------------------------------------------------------------------------


def _a_clip(doc: dict) -> str:
    return next(s["clip"] for c in doc["characters"] for s in c["samples"])


def test_referenced_clip_is_served(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    status, headers, data = studio.request("GET", f"/audio/{clip}")
    assert status == 200 and headers["content-type"] == "audio/wav"
    assert data == (ws.root / "studio" / "audio" / clip).read_bytes()
    cookie = f"fukikae_studio_{studio.port}={TOKEN}"  # what an <audio> element sends
    assert studio.request("GET", f"/audio/{clip}", token=None, cookie=cookie)[0] == 200
    assert studio.request("GET", f"/audio/{clip}", token=None)[0] == 403


def test_unreferenced_or_odd_audio_names_are_404(studio, ws):  # noqa: F811
    _season(studio)
    audio = ws.root / "studio" / "audio"
    stray = "f" * 64 + ".wav"
    (audio / stray).write_bytes(b"RIFF")
    for path in (f"/audio/{stray}", "/audio/../season.json", "/audio/%2e%2e/season.json",
                 "/audio/season.json", "/audio/" + "A" * 64 + ".wav", "/audio/"):  # fmt: skip
        assert studio.request("GET", path)[0] == 404, path


def test_symlinked_clip_is_not_followed(studio, ws, tmp_path):  # noqa: F811
    clip = _a_clip(_season(studio))
    path = ws.root / "studio" / "audio" / clip
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"secret")
    path.unlink()
    os.symlink(secret, path)
    assert studio.request("GET", f"/audio/{clip}")[0] == 404


def test_symlinked_audio_directory_is_not_followed(studio, ws, tmp_path):  # noqa: F811
    clip = _a_clip(_season(studio))
    audio = ws.root / "studio" / "audio"
    moved = tmp_path / "elsewhere"
    audio.rename(moved)
    os.symlink(moved, audio)
    assert studio.request("GET", f"/audio/{clip}")[0] == 404


# --- static files --------------------------------------------------------------------------


def test_static_paths_cannot_escape_the_package(studio):
    assert studio.request("GET", "/static/index.html")[0] == 200
    for path in ("/static/../server.py", "/static/%2e%2e/server.py", "/static/..%2fserver.py",
                 "/static/.hidden", "/static/", "/static//etc/passwd"):  # fmt: skip
        assert studio.request("GET", path)[0] == 404, path


# --- the command ---------------------------------------------------------------------------


def _no_serve(monkeypatch) -> list:
    from fukikae.studio import server

    started: list = []

    def fake(self, *_a, **_kw):
        started.append(self.server_address)
        raise KeyboardInterrupt

    monkeypatch.setattr(server.StudioServer, "serve_forever", fake)
    return started


def test_studio_ws_starts_the_server_and_prints_the_url_and_tunnel(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--port", "0", "--no-open"])
    assert r.exit_code == 0, r.output
    assert started and started[0][0] == "127.0.0.1"
    port = started[0][1]
    assert f"http://127.0.0.1:{port}/?t=" in r.output
    assert f"ssh -L {port}:127.0.0.1:{port} " in r.output
    assert (ws.root / "studio" / "season.json").is_file()


def test_studio_serve_is_the_same_command(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", "serve", str(ws.root), "--port", "0", "--no-open"])
    assert r.exit_code == 0, r.output and started


def test_studio_subcommands_still_resolve(ws):  # noqa: F811
    from fukikae.cli import app

    r = CliRunner().invoke(app, ["studio", "apply", "--help"])
    assert r.exit_code == 0 and "FILE" in r.output
    r = CliRunner().invoke(app, ["studio", "resume", str(ws.root)])
    assert r.exit_code == 0, r.output


def test_studio_refuses_a_non_loopback_host_without_the_flag(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--host", "0.0.0.0", "--no-open"])  # noqa: S104
    assert r.exit_code != 0 and "loopback" in r.output and not started


def test_studio_on_a_missing_workspace_fails_cleanly(tmp_path: Path):
    from fukikae.cli import app

    r = CliRunner().invoke(app, ["studio", str(tmp_path / "nope"), "--no-open"])
    assert r.exit_code != 0 and "Traceback" not in r.output
