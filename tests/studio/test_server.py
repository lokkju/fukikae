"""`fukikae studio <ws>`: the loopback server, its token and host guards, the JSON API over
the season index and decisions, and audio serving (Task 17, server half)."""

from __future__ import annotations

import contextlib
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

    server = StudioServer(
        ws, load_config(ws.root, user_config=None), port=0, token=TOKEN, user_config=None
    )
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


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.30.201", "example.com", "::", "::1", "127.0.0.2", "127.1"],  # noqa: S104
)
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


@pytest.mark.parametrize("host", ["127.0.0.2", "127.255.255.254"])
def test_loopback_means_127_0_0_1_only(host):
    from fukikae.studio.server import check_bind

    for flag in (False, True):
        with pytest.raises(ValueError, match="127.0.0.1"):
            check_bind(host, allow_non_loopback=flag)


@pytest.mark.parametrize("host", ["0.0.0.0", "255.255.255.255", "224.0.0.1", "::"])  # noqa: S104
def test_non_loopback_needs_one_interface_address(host):
    from fukikae.studio.server import check_bind

    with pytest.raises(ValueError, match="interface|IPv4"):
        check_bind(host, allow_non_loopback=True)


def _local_ipv4() -> str | None:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent
            addr = s.getsockname()[0]
    except OSError:
        return None
    return None if addr.startswith("127.") else addr


def test_a_non_loopback_bind_allows_its_own_host_and_prints_its_url(ws):  # noqa: F811
    from fukikae.studio.server import StudioServer

    addr = _local_ipv4()
    if addr is None:
        pytest.skip("no non-loopback IPv4 address here")
    s = StudioServer(ws, load_config(ws.root, user_config=None), host=addr, port=0,
                     token=TOKEN, allow_non_loopback=True)  # fmt: skip
    try:
        assert f"{addr}:{s.port}" in s.allowed_hosts
        assert f"127.0.0.1:{s.port}" not in s.allowed_hosts
        assert s.url() == f"http://{addr}:{s.port}/#t={TOKEN}"
    finally:
        s.server_close()


def test_a_fresh_token_per_server(ws):  # noqa: F811
    from fukikae.studio.server import StudioServer

    cfg = load_config(ws.root, user_config=None)
    a = StudioServer(ws, cfg, port=0)
    b = StudioServer(ws, cfg, port=0)
    try:
        assert a.token != b.token and len(a.token) >= 32
        assert a.url() == f"http://127.0.0.1:{a.server_address[1]}/#t={a.token}"
    finally:
        a.server_close()
        b.server_close()


# --- the token and host guards -------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/season", "/api/characters/1", "/api/nothing"])
def test_api_requests_without_the_token_header_are_refused(studio, path):
    assert studio.request("GET", path, token=None)[0] == 403
    assert studio.request("GET", path, token="wrong")[0] == 403  # noqa: S106  # noqa: S106


def test_audio_without_the_token_header_is_refused(studio):
    clip = _a_clip(_season(studio))
    assert studio.request("GET", f"/audio/{clip}", token=None)[0] == 403
    assert studio.request("GET", "/audio/nothing.wav", token=None)[0] == 403


def test_the_static_shell_needs_no_token_and_holds_no_data(studio, ws):  # noqa: F811
    doc = _season(studio)
    for path in ("/", "/static/index.html", "/static/token.js"):
        status, headers, body = studio.request("GET", path, token=None)
        assert status == 200, path
        assert "set-cookie" not in headers
        assert TOKEN.encode() not in body and str(ws.root).encode() not in body
        for c in doc["characters"]:
            if c["name"]:
                assert c["name"].encode() not in body, (path, c["name"])
    status, headers, page = studio.request("GET", "/", token=None)
    assert headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in headers["content-security-policy"]
    assert b"/api/season" in page and b"/static/token.js" in page
    assert b"<script>" not in page  # no inline script: the CSP allows none


def test_no_response_ever_sets_a_cookie(studio):
    doc = _season(studio)
    clip = _a_clip(doc)
    for method, path, body in [
        ("GET", "/", None), ("GET", f"/?t={TOKEN}", None), ("GET", "/static/index.html", None),
        ("GET", "/api/season", None), ("GET", "/api/characters/1", None),
        ("GET", f"/audio/{clip}", None), ("GET", "/api/nothing", None),
        ("POST", "/api/decisions/preview", _name(doc, 1, "Lucia")),
    ]:  # fmt: skip
        for token in (TOKEN, None):
            _, headers, _ = studio.request(method, path, body, token=token)
            assert "set-cookie" not in headers, (method, path, token)


def test_a_cookie_alone_authorizes_nothing(studio):
    doc = _season(studio)
    clip = _a_clip(doc)
    cookie = f"fukikae_studio_{studio.port}={TOKEN}"
    assert studio.request("GET", "/api/season", token=None, cookie=cookie)[0] == 403
    assert studio.request("GET", f"/audio/{clip}", token=None, cookie=cookie)[0] == 403
    status, _ = studio.json(
        "POST", "/api/decisions", _name(doc, 3, "Erza"), token=None, cookie=cookie
    )
    assert status == 403
    status, _, _ = studio.request("POST", "/api/decisions", "{}", token=None, cookie=cookie,
                                  content_type="text/plain")  # fmt: skip
    assert status == 403


def test_a_token_in_the_query_is_ignored(studio):
    clip = _a_clip(_season(studio))
    assert studio.request("GET", f"/api/season?t={TOKEN}", token=None)[0] == 403
    assert studio.request("GET", f"/audio/{clip}?t={TOKEN}", token=None)[0] == 403
    status, headers, _ = studio.request("GET", f"/?t={TOKEN}", token=None)
    assert status == 200 and "location" not in headers  # the shell, no login redirect


def test_the_token_header_authorizes_api_reads_and_audio(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    assert studio.request("GET", "/api/season")[0] == 200
    assert studio.request("GET", "/api/characters/1")[0] == 200
    status, headers, data = studio.request("GET", f"/audio/{clip}")
    assert status == 200 and data == (ws.root / "studio" / "audio" / clip).read_bytes()


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
    studio.request("GET", f"/api/season?t={TOKEN}", token=None)
    studio.request("GET", "/api/season")
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
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--port", "0"])
    assert r.exit_code == 0, r.output
    assert started and started[0][0] == "127.0.0.1"
    port = started[0][1]
    assert f"http://127.0.0.1:{port}/#t=" in r.output
    assert "?t=" not in r.output
    assert f"ssh -L {port}:127.0.0.1:{port} " in r.output
    assert (ws.root / "studio" / "season.json").is_file()


def test_studio_serve_is_the_same_command(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", "serve", str(ws.root), "--port", "0"])
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
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--host", "0.0.0.0"])  # noqa: S104
    assert r.exit_code != 0 and "loopback" in r.output and not started


def test_studio_on_a_missing_workspace_fails_cleanly(tmp_path: Path):
    from fukikae.cli import app

    r = CliRunner().invoke(app, ["studio", str(tmp_path / "nope")])
    assert r.exit_code != 0 and "Traceback" not in r.output


# --- failures the handler didn't expect -----------------------------------------------------


def test_an_uncaught_exception_is_a_generic_500_with_the_traceback_on_stderr(
    studio, monkeypatch, capsys
):
    import fukikae.studio.server as srv

    def boom(*_a, **_kw):
        raise RuntimeError("boom /secret/path")

    monkeypatch.setattr(srv, "load_or_build", boom)
    status, headers, body = studio.request("GET", "/api/season")
    assert status == 500 and json.loads(body) == {"error": "internal"}
    assert b"secret" not in body
    err = capsys.readouterr().err
    assert "Traceback" in err and "boom /secret/path" in err


@pytest.mark.parametrize(
    ("exc", "status", "error"),
    [
        ("schema", 409, "registry-schema"),
        ("locked", 503, "busy"),
        ("corrupt", 409, "registry-unreadable"),
    ],
)
def test_season_failures_are_clear_errors(studio, monkeypatch, exc, status, error):
    import sqlite3

    import fukikae.studio.server as srv
    from fukikae.registry.store import RegistrySchemaError

    errors = {
        "schema": RegistrySchemaError("registry schema 9 is newer than this fukikae (5)"),
        "locked": sqlite3.OperationalError("database is locked"),
        "corrupt": sqlite3.DatabaseError("file is not a database"),
    }

    def fail(*_a, **_kw):
        raise errors[exc]

    monkeypatch.setattr(srv, "load_or_build", fail)
    for path in ("/api/season", "/api/characters/1"):
        got, body = studio.json("GET", path)
        assert got == status and body["error"] == error, (path, body)
        assert body["message"]


def test_a_decision_that_lands_when_the_index_rebuild_fails_is_200_with_a_warning(
    studio,
    ws,  # noqa: F811
    monkeypatch,
):
    import fukikae.studio.server as srv

    doc = _season(studio)

    def boom(*_a, **_kw):
        raise ValueError(f"can't write {ws.root}/studio/season.json")

    monkeypatch.setattr(srv, "load_or_build", boom)
    status, body = studio.json("POST", "/api/decisions", _name(doc, 3, "Erza"))
    assert status == 200, body
    assert body["character"] is None and body["lines"]
    assert isinstance(body["invalidations"], list)
    assert "landed" in body["warning"] and "index" in body["warning"]
    assert "ValueError" in body["warning"] and str(ws.root) not in body["warning"]
    with Registry(ws.registry_path) as reg:
        assert _reg_name(reg, 3) == "Erza"


def test_a_broken_pipe_while_sending_an_error_is_quiet(studio, monkeypatch, capsys):
    import fukikae.studio.server as srv

    def gone(_self, *_a, **_kw):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(srv._Handler, "_bytes", gone)
    with contextlib.suppress(http.client.HTTPException, OSError):
        studio.request("GET", "/api/nothing")
    import time

    time.sleep(0.2)
    err = capsys.readouterr().err
    assert "Traceback" not in err and "BrokenPipe" not in err


# --- timeouts and limits ------------------------------------------------------------------


def _serve(ws, **kw):  # noqa: F811
    from fukikae.studio.server import StudioServer

    server = StudioServer(ws, load_config(ws.root, user_config=None), port=0, token=TOKEN, **kw)
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    return server, t


def _stop(server, t) -> None:
    server.shutdown()
    server.server_close()
    t.join(5)


def _raw(port: int, data: bytes, timeout: float = 5) -> bytes:
    import socket

    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.sendall(data)
    out = b""
    try:
        while chunk := s.recv(65536):
            out += chunk
    except TimeoutError:
        out += b"<<TIMEOUT>>"
    s.close()
    return out


def test_handlers_time_out_after_30_seconds():
    from fukikae.studio.server import StudioServer, _Handler

    assert _Handler.timeout == 30
    assert StudioServer.request_timeout == 30


def test_a_body_shorter_than_its_content_length_times_out(ws):  # noqa: F811
    import time

    server, t = _serve(ws, request_timeout=0.5)
    try:
        p = server.port
        req = (
            f"POST /api/decisions HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nX-Fukikae-Token: {TOKEN}"
            "\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{}"
        ).encode()
        t0 = time.monotonic()
        out = _raw(p, req, timeout=5)
        assert time.monotonic() - t0 < 4
        assert b"<<TIMEOUT>>" not in out
        assert out.startswith(b"HTTP/1.0 408") or out == b"", out[:200]
    finally:
        _stop(server, t)


def test_handlers_over_the_cap_get_a_prompt_503(ws):  # noqa: F811
    import socket
    import time

    server, t = _serve(ws, max_handlers=4)
    socks = []
    try:
        p = server.port
        for _ in range(4):  # four handlers, each waiting on headers that never finish
            s = socket.create_connection(("127.0.0.1", p))
            s.sendall(b"GET / HTTP/1.1\r\n")
            socks.append(s)
        time.sleep(0.3)
        t0 = time.monotonic()
        out = _raw(p, f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n\r\n".encode(), timeout=3)
        assert time.monotonic() - t0 < 2
        assert out.startswith(b"HTTP/1.1 503"), out[:200]
        assert b'"busy"' in out
        for s in socks:
            s.close()
        socks.clear()
        time.sleep(0.3)  # the slots free as their handlers end
        out = _raw(p, f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n\r\n".encode(), timeout=3)
        assert out.startswith(b"HTTP/1.0 200"), out[:200]
    finally:
        for s in socks:
            s.close()
        _stop(server, t)


def test_the_default_cap_is_16():
    from fukikae.studio.server import MAX_HANDLERS

    assert MAX_HANDLERS == 16


# --- opening files safely -----------------------------------------------------------------


def test_a_fifo_in_place_of_a_clip_is_refused_without_blocking(studio, ws):  # noqa: F811
    import time

    clip = _a_clip(_season(studio))
    f = ws.root / "studio" / "audio" / clip
    f.unlink()
    os.mkfifo(f)
    p = studio.port
    t0 = time.monotonic()
    out = _raw(p, (f"GET /audio/{clip} HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n"
                   f"X-Fukikae-Token: {TOKEN}\r\n\r\n").encode(), timeout=3)  # fmt: skip
    assert time.monotonic() - t0 < 2
    assert out.startswith(b"HTTP/1.0 404"), out[:200]


def test_a_symlinked_studio_directory_is_not_followed(studio, ws, tmp_path):  # noqa: F811
    clip = _a_clip(_season(studio))
    studio_dir = ws.root / "studio"
    moved = tmp_path / "studio-elsewhere"
    studio_dir.rename(moved)
    os.symlink(moved, studio_dir)
    assert studio.request("GET", f"/audio/{clip}")[0] == 404


def test_static_files_are_opened_without_following_links(studio, monkeypatch, tmp_path):
    import fukikae.studio.server as srv

    root = tmp_path / "static"
    (root / "sub").mkdir(parents=True)
    (root / "ok.txt").write_text("ok")
    (root / "sub" / "ok.txt").write_text("deep")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    os.symlink(outside / "secret.txt", root / "link.txt")
    os.symlink(outside, root / "linkdir")
    os.mkfifo(root / "fifo.txt")
    monkeypatch.setattr(srv, "STATIC_DIR", root)
    assert studio.request("GET", "/static/ok.txt", token=None)[2] == b"ok"
    assert studio.request("GET", "/static/sub/ok.txt", token=None)[2] == b"deep"
    for path in ("/static/link.txt", "/static/linkdir/secret.txt", "/static/fifo.txt",
                 "/static/sub", "/static/missing.txt"):  # fmt: skip
        assert studio.request("GET", path, token=None)[0] == 404, path


# --- the preview is a dry run -------------------------------------------------------------


def test_preview_returns_the_stamped_decision_to_post(studio, ws):  # noqa: F811
    doc = _season(studio)
    bare = {"op": "name", "character": 3, "name": "Erza"}
    status, body = studio.json("POST", "/api/decisions/preview", bare)
    assert status == 200, body
    assert body["unchecked"] is True and body["stale"] is None
    d = body["decision"]
    assert d["digest"] == _entry(doc, 3)["digest"] and d["op"] == "name" and d["name"] == "Erza"
    status, applied = studio.json("POST", "/api/decisions", d)  # exactly what it previewed
    assert status == 200, applied
    with Registry(ws.registry_path) as reg:
        assert _reg_name(reg, 3) == "Erza"


def test_preview_keeps_a_stale_digest_as_sent(studio, ws):  # noqa: F811
    doc = _season(studio)
    sent = _name(doc, 3, "Erza")
    with Registry(ws.registry_path) as reg:
        reg.rename_character(3, "Someone")
    status, body = studio.json("POST", "/api/decisions/preview", sent)
    assert status == 200 and body["stale"] and body["decision"]["digest"] == sent["digest"]
    assert studio.json("POST", "/api/decisions", body["decision"])[0] == 409


def test_preview_of_a_merge_stamps_both_characters(studio, ws):  # noqa: F811
    from fukikae.studio.apply import decision_digest
    from fukikae.studio.model import parse_decision

    status, body = studio.json("POST", "/api/decisions/preview",
                               {"op": "merge", "keep": 1, "drop": 2})  # fmt: skip
    assert status == 200, body
    d = body["decision"]
    assert d["op"] == "merge" and d["keep"] == 1 and d["drop"] == 2
    with Registry(ws.registry_path) as reg:
        assert d["digest"] == decision_digest(
            reg, parse_decision({"op": "merge", "keep": 1, "drop": 2})
        )
    assert set(body) >= {"lines", "invalidations", "stale", "unchecked", "decision"}


def test_preview_runs_the_write_time_refusals(studio):
    status, body = studio.json("POST", "/api/decisions/preview",
                               {"op": "merge", "keep": 1, "drop": 1})  # fmt: skip
    assert status == 422 and body["error"] == "refused"


def test_preview_never_migrates_or_creates_a_registry(studio, ws):  # noqa: F811
    doc = _season(studio)
    sent = _name(doc, 3, "Erza")
    ws.registry_path.unlink()
    ws.registry_path.write_bytes(b"")
    status, body = studio.json("POST", "/api/decisions/preview", sent)
    assert status in (409, 422) and body["error"] != "internal", body
    assert ws.registry_path.read_bytes() == b""
    ws.registry_path.unlink()
    status, body = studio.json("POST", "/api/decisions/preview", sent)
    assert status == 409 and not ws.registry_path.exists()


def test_preview_does_not_open_the_registry_directly():
    import inspect

    import fukikae.studio.server as srv

    assert "Registry(" not in inspect.getsource(srv._Handler)


# --- the command, continued -------------------------------------------------------------------


def test_studio_options_before_the_workspace_go_to_serve(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", "--port", "0", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert started and f"http://127.0.0.1:{started[0][1]}/#t=" in r.output


def test_studio_help_and_subcommands_still_work():
    from fukikae.cli import app

    r = CliRunner().invoke(app, ["studio", "--help"])
    assert r.exit_code == 0 and "apply" in r.output and "resume" in r.output
    for sub in ("serve", "apply", "resume"):
        r = CliRunner().invoke(app, ["studio", sub, "--help"])
        assert r.exit_code == 0, (sub, r.output)
    r = CliRunner().invoke(app, ["studio", "serve", "--help"])
    assert "--no-open" not in r.output


def test_studio_never_opens_a_browser(ws, monkeypatch):  # noqa: F811
    import webbrowser

    from fukikae.cli import app

    opened: list = []
    monkeypatch.setattr(webbrowser, "open", lambda *a, **_k: opened.append(a))
    monkeypatch.setenv("DISPLAY", ":0")
    _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--port", "0"])
    assert r.exit_code == 0, r.output and not opened
    assert not opened
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--port", "0", "--no-open"])
    assert r.exit_code != 0  # the option is gone


def test_studio_refuses_another_loopback_address(ws, monkeypatch):  # noqa: F811
    from fukikae.cli import app

    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--host", "127.0.0.2"])
    assert r.exit_code != 0 and "127.0.0.1" in r.output and not started


@pytest.mark.parametrize("which", ["schema", "corrupt"])
def test_studio_startup_reports_index_failures_in_one_line(ws, monkeypatch, which):  # noqa: F811
    import sqlite3

    import fukikae.studio.server as srv
    from fukikae.cli import app
    from fukikae.registry.store import RegistrySchemaError

    exc = (RegistrySchemaError("registry schema 9 is newer than this fukikae") if which == "schema"
           else sqlite3.DatabaseError("file is not a database"))  # fmt: skip

    def fail(*_a, **_kw):
        raise exc

    monkeypatch.setattr(srv, "load_or_build", fail)
    started = _no_serve(monkeypatch)
    r = CliRunner().invoke(app, ["studio", str(ws.root), "--port", "0"])
    assert r.exit_code == 1 and not started
    assert "Traceback" not in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)
    errs = [ln for ln in r.output.splitlines() if ln.startswith("error:")]
    assert len(errs) == 1 and str(exc) in errs[0]


# --- range requests ------------------------------------------------------------------------


def test_audio_answers_a_single_byte_range(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    data = (ws.root / "studio" / "audio" / clip).read_bytes()
    n = len(data)
    status, headers, body = studio.request("GET", f"/audio/{clip}")
    assert status == 200 and headers["accept-ranges"] == "bytes"
    status, headers, body = studio.request("GET", f"/audio/{clip}",
                                           headers={"Range": "bytes=0-99"})  # fmt: skip
    assert status == 206 and body == data[:100]
    assert headers["content-range"] == f"bytes 0-99/{n}" and headers["content-length"] == "100"
    assert headers["content-type"] == "audio/wav"
    status, headers, body = studio.request("GET", f"/audio/{clip}",
                                           headers={"Range": "bytes=100-"})  # fmt: skip
    assert status == 206 and body == data[100:]
    assert headers["content-range"] == f"bytes 100-{n - 1}/{n}"
    status, _, body = studio.request("GET", f"/audio/{clip}", headers={"Range": "bytes=-10"})
    assert status == 206 and body == data[-10:]
    status, headers, body = studio.request("GET", f"/audio/{clip}",
                                           headers={"Range": f"bytes=10-{n + 500}"})  # fmt: skip
    assert status == 206 and body == data[10:]
    assert headers["content-range"] == f"bytes 10-{n - 1}/{n}"


@pytest.mark.parametrize("rng", ["bytes={n}-", "bytes={n}-{m}", "bytes=-0"])
def test_an_unsatisfiable_range_is_416(studio, ws, rng):  # noqa: F811
    clip = _a_clip(_season(studio))
    n = (ws.root / "studio" / "audio" / clip).stat().st_size
    status, headers, _ = studio.request(
        "GET", f"/audio/{clip}", headers={"Range": rng.format(n=n, m=n + 10)}
    )
    assert status == 416 and headers["content-range"] == f"bytes */{n}"


@pytest.mark.parametrize("rng", ["bytes=5-2", "bytes=0-1,4-5", "items=0-5", "bytes=x-", "bytes="])
def test_a_range_the_server_does_not_take_gets_the_whole_clip(studio, ws, rng):  # noqa: F811
    clip = _a_clip(_season(studio))
    data = (ws.root / "studio" / "audio" / clip).read_bytes()
    status, _, body = studio.request("GET", f"/audio/{clip}", headers={"Range": rng})
    assert status == 200 and body == data


def test_a_range_still_needs_the_token_and_a_referenced_clip(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    rng = {"Range": "bytes=0-9"}
    assert studio.request("GET", f"/audio/{clip}", token=None, headers=rng)[0] == 403
    stray = "e" * 64 + ".wav"
    (ws.root / "studio" / "audio" / stray).write_bytes(b"RIFF" * 10)
    assert studio.request("GET", f"/audio/{stray}", headers=rng)[0] == 404


# --- video ---------------------------------------------------------------------------------


@contextlib.contextmanager
def _serve_ws(w: Workspace) -> Iterator[Client]:
    from fukikae.studio.server import StudioServer

    server = StudioServer(w, load_config(w.root, user_config=None), port=0, token=TOKEN)
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    try:
        yield Client(server)
    finally:
        server.shutdown()
        server.server_close()
        t.join(5)


@pytest.fixture
def video_studio(ws: Workspace, tmp_path: Path) -> Iterator[Client]:  # noqa: F811
    from tests.studio.test_season import _make_source, _with_source

    _with_source(ws, _make_source(tmp_path / "made.mkv"), tmp_path / "media")
    with _serve_ws(ws) as c:
        yield c


def _a_video(doc: dict) -> str:
    return next(s["video"] for c in doc["characters"] for s in c["samples"] if s.get("video"))


@pytest.mark.ffmpeg
def test_a_referenced_video_is_cut_on_first_fetch_and_served(video_studio, ws):  # noqa: F811
    name = _a_video(_season(video_studio))
    path = ws.root / "studio" / "video" / name
    assert not path.exists()
    status, headers, data = video_studio.request("GET", f"/video/{name}")
    assert status == 200 and headers["content-type"] == "video/mp4"
    assert headers["accept-ranges"] == "bytes" and headers["cache-control"] == "no-store"
    assert data == path.read_bytes() and data[4:8] == b"ftyp"
    status, headers, part = video_studio.request("GET", f"/video/{name}",
                                                 headers={"Range": "bytes=0-1023"})  # fmt: skip
    assert status == 206 and part == data[:1024]
    assert headers["content-range"] == f"bytes 0-1023/{len(data)}"
    status, headers, _ = video_studio.request("GET", f"/video/{name}",
                                              headers={"Range": f"bytes={len(data)}-"})  # fmt: skip
    assert status == 416 and headers["content-range"] == f"bytes */{len(data)}"


@pytest.mark.ffmpeg
def test_video_needs_the_token_header(video_studio):
    name = _a_video(_season(video_studio))
    assert video_studio.request("GET", f"/video/{name}", token=None)[0] == 403
    assert video_studio.request("GET", f"/video/{name}", token="wrong")[0] == 403  # noqa: S106
    assert video_studio.request("GET", f"/video/{name}", token=None,
                                headers={"Range": "bytes=0-9"})[0] == 403  # fmt: skip
    assert video_studio.request("GET", "/video/nothing.mp4", token=None)[0] == 403


@pytest.mark.ffmpeg
def test_unreferenced_or_odd_video_names_are_404(video_studio, ws):  # noqa: F811
    _season(video_studio)
    videos = ws.root / "studio" / "video"
    videos.mkdir(parents=True, exist_ok=True)
    stray = "f" * 64 + ".mp4"
    (videos / stray).write_bytes(b"\0\0\0\x18ftypisom" * 8)
    odd = ("/video/../season.json", "/video/%2e%2e/season.json", "/video/" + "A" * 64 + ".mp4",
           "/video/" + "a" * 64 + ".wav", "/video/")  # fmt: skip
    for path in (f"/video/{stray}", *odd):
        assert video_studio.request("GET", path)[0] == 404, path
    assert video_studio.request("GET", f"/video/{stray}", headers={"Range": "bytes=0-9"})[0] == 404
    clip = _a_clip(_season(video_studio))  # an audio clip's name is not a video
    assert video_studio.request("GET", f"/video/{clip}")[0] == 404
    assert video_studio.request("POST", f"/video/{_a_video(_season(video_studio))}", {})[0] == 405


@pytest.mark.ffmpeg
def test_a_symlink_or_fifo_in_place_of_a_video_is_refused(video_studio, ws, tmp_path):  # noqa: F811
    import time

    doc = _season(video_studio)
    names = [s["video"] for c in doc["characters"] for s in c["samples"] if s.get("video")]
    videos = ws.root / "studio" / "video"
    videos.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"secret")
    os.symlink(secret, videos / names[0])
    assert video_studio.request("GET", f"/video/{names[0]}")[0] == 404
    assert secret.read_bytes() == b"secret"
    os.mkfifo(videos / names[1])
    p = video_studio.port
    t0 = time.monotonic()
    out = _raw(p, (f"GET /video/{names[1]} HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n"
                   f"X-Fukikae-Token: {TOKEN}\r\n\r\n").encode(), timeout=3)  # fmt: skip
    assert time.monotonic() - t0 < 2
    assert out.startswith(b"HTTP/1.0 404"), out[:200]


@pytest.mark.ffmpeg
def test_a_symlinked_video_directory_is_not_followed(video_studio, ws, tmp_path):  # noqa: F811
    name = _a_video(_season(video_studio))
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    videos = ws.root / "studio" / "video"
    if videos.exists():
        videos.rmdir()
    os.symlink(moved, videos)
    assert video_studio.request("GET", f"/video/{name}")[0] == 404
    assert list(moved.iterdir()) == []  # nothing was cut through the link


# --- range edges, HEAD, and cut liveness ----------------------------------------------------


def test_range_edges_on_audio(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    path = ws.root / "studio" / "audio" / clip
    data = path.read_bytes()
    n = len(data)
    status, headers, body = studio.request("GET", f"/audio/{clip}", headers={"Range": "bytes=0-0"})
    assert status == 206 and body == data[:1] and headers["content-length"] == "1"
    assert headers["content-range"] == f"bytes 0-0/{n}"
    status, headers, body = studio.request("GET", f"/audio/{clip}",
                                           headers={"Range": f"bytes=-{n * 10}"})  # fmt: skip
    assert status == 206 and body == data and headers["content-range"] == f"bytes 0-{n - 1}/{n}"
    status, headers, body = studio.request("HEAD", f"/audio/{clip}")
    assert status == 200 and body == b"" and headers["content-length"] == str(n)
    status, headers, body = studio.request("HEAD", f"/audio/{clip}",
                                           headers={"Range": "bytes=0-9"})  # fmt: skip
    assert status == 206 and body == b"" and headers["content-length"] == "10"


def test_range_requests_against_an_empty_clip(studio, ws):  # noqa: F811
    clip = _a_clip(_season(studio))
    (ws.root / "studio" / "audio" / clip).write_bytes(b"")
    status, headers, body = studio.request("GET", f"/audio/{clip}")
    assert status == 200 and body == b"" and headers["content-length"] == "0"
    for rng in ("bytes=0-0", "bytes=0-", "bytes=-5"):
        status, headers, body = studio.request("GET", f"/audio/{clip}", headers={"Range": rng})
        assert status == 416 and headers["content-range"] == "bytes */0", rng
        assert headers["content-length"] == str(len(body))


FAKE_MP4 = b"\0\0\0\x18ftypisom" + b"\0" * 100


@contextlib.contextmanager
def _serve_kw(w: Workspace, **kw) -> Iterator[Client]:
    from fukikae.studio.server import StudioServer

    server = StudioServer(w, load_config(w.root, user_config=None), port=0, token=TOKEN, **kw)
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    try:
        yield Client(server)
    finally:
        server.shutdown()
        server.server_close()
        t.join(5)


@pytest.fixture
def video_ws(ws: Workspace, tmp_path: Path) -> Workspace:  # noqa: F811
    from tests.studio.test_season import _make_source, _with_source

    _with_source(ws, _make_source(tmp_path / "made.mkv"), tmp_path / "media")
    return ws


class _FakeCut:
    """`media.run_ffmpeg` that writes `FAKE_MP4` (or fails) after a delay or a gate, counting
    calls and the most running at once."""

    def __init__(self, *, delay: float = 0.0, fail: bool = False, gate=None) -> None:
        self.delay, self.fail, self.gate = delay, fail, gate
        self.calls: list[str] = []
        self.running = self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, args: list[str], *, timeout: float | None = None) -> None:  # noqa: ARG002
        import time

        from fukikae.media import MediaError

        with self.lock:
            self.calls.append(args[-1])
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if self.gate is not None:
                self.gate.wait(30)
            time.sleep(self.delay)
            if self.fail:
                raise MediaError("boom")
            Path(args[-1]).write_bytes(FAKE_MP4)
        finally:
            with self.lock:
                self.running -= 1


def _videos_of(doc: dict) -> list[str]:
    return list(dict.fromkeys(s["video"] for c in doc["characters"] for s in c["samples"]
                              if s.get("video")))  # fmt: skip


def _fetch_all(c: Client, paths: list[str]) -> list[tuple[int, dict[str, str], bytes]]:
    out: list = [None] * len(paths)

    def one(i: int) -> None:
        out[i] = c.request("GET", paths[i])

    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(paths))]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return out


@pytest.mark.ffmpeg
def test_head_on_an_uncut_video_does_not_cut_it(video_ws, monkeypatch):
    from fukikae import media

    fake = _FakeCut()
    monkeypatch.setattr(media, "run_ffmpeg", fake)
    with _serve_kw(video_ws) as c:
        name = _a_video(_season(c))
        status, _, body = c.request("HEAD", f"/video/{name}")
        assert status == 404 and not fake.calls
        assert not (video_ws.root / "studio" / "video" / name).exists()
        assert c.request("GET", f"/video/{name}")[0] == 200 and len(fake.calls) == 1
        status, headers, body = c.request("HEAD", f"/video/{name}")
        assert status == 200 and body == b"" and headers["content-length"] == str(len(FAKE_MP4))
        status, headers, body = c.request("HEAD", f"/video/{name}",
                                          headers={"Range": "bytes=0-0"})  # fmt: skip
        assert status == 206 and body == b"" and headers["content-length"] == "1"
        assert len(fake.calls) == 1


@pytest.mark.ffmpeg
@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_fetches_of_one_name_cut_it_once(video_ws, monkeypatch, fail):
    from fukikae import media

    fake = _FakeCut(delay=0.3, fail=fail)
    monkeypatch.setattr(media, "run_ffmpeg", fake)
    with _serve_kw(video_ws) as c:
        name = _a_video(_season(c))
        got = _fetch_all(c, [f"/video/{name}"] * 6)
        assert [r[0] for r in got] == [404 if fail else 200] * 6
        assert len(fake.calls) == 1


@pytest.mark.ffmpeg
def test_a_failed_cut_is_remembered_until_it_expires_or_the_index_is_rebuilt(video_ws, monkeypatch):
    from fukikae import media
    from fukikae.studio import server as srv

    fake = _FakeCut(fail=True)
    monkeypatch.setattr(media, "run_ffmpeg", fake)
    with _serve_kw(video_ws) as c:
        name = _a_video(_season(c))
        got = [c.request("GET", f"/video/{name}") for _ in range(5)]
        assert [r[0] for r in got] == [404] * 5 and len(fake.calls) == 1
        assert json.loads(got[-1][2])["error"] == "cut-failed"
        monkeypatch.setattr(srv, "VIDEO_FAILURE_TTL_S", 0.0)
        assert c.request("GET", f"/video/{name}")[0] == 404 and len(fake.calls) == 2
        monkeypatch.setattr(srv, "VIDEO_FAILURE_TTL_S", 300.0)
        assert c.request("GET", f"/video/{name}")[0] == 404 and len(fake.calls) == 3
        assert c.request("GET", f"/video/{name}")[0] == 404 and len(fake.calls) == 3
        (video_ws.root / "studio" / "season.json").unlink()  # the next index is a rebuild
        assert name in _videos_of(_season(c))
        fake.fail = False
        assert c.request("GET", f"/video/{name}")[0] == 200 and len(fake.calls) == 4


@pytest.mark.ffmpeg
def test_at_most_max_cuts_run_at_once(video_ws, monkeypatch):
    from fukikae import media
    from fukikae.studio.server import MAX_CUTS

    fake = _FakeCut(delay=0.4)
    monkeypatch.setattr(media, "run_ffmpeg", fake)
    with _serve_kw(video_ws) as c:
        names = _videos_of(_season(c))[:4]
        assert len(names) == 4
        got = _fetch_all(c, [f"/video/{n}" for n in names])
        assert [r[0] for r in got] == [200] * 4
        assert len(fake.calls) == 4 and fake.peak == MAX_CUTS


@pytest.mark.ffmpeg
def test_stuck_cuts_answer_busy_and_leave_the_index_answering(video_ws, monkeypatch):
    import time

    from fukikae import media

    gate = threading.Event()
    fake = _FakeCut(gate=gate)
    monkeypatch.setattr(media, "run_ffmpeg", fake)
    try:
        with _serve_kw(video_ws, max_handlers=4, cut_wait=0.5) as c:
            names = _videos_of(_season(c))[:3]
            out: list = []

            def one(path: str) -> None:
                out.append(c.request("GET", path))

            for i in range(5):
                threading.Thread(target=one, args=(f"/video/{names[i % 3]}",),
                                 daemon=True).start()  # fmt: skip
                time.sleep(0.05)
            time.sleep(2.0)
            t0 = time.monotonic()
            assert c.request("GET", "/api/season")[0] == 200
            assert time.monotonic() - t0 < 5
            assert len(fake.calls) == 2  # MAX_CUTS stuck; the rest gave up waiting
            busy = [r for r in out if r[0] == 503]
            assert len(out) == 3 and len(busy) == 3
            assert all(json.loads(r[2])["error"] == "busy" and r[1].get("retry-after")
                       for r in busy)  # fmt: skip
    finally:
        gate.set()
