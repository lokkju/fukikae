#!/usr/bin/env python3
"""A stand-in for ``docker`` in the indextts2_trt tests.

Every call appends its argv as a JSON line to ``$FUKIKAE_FAKE_DOCKER_LOG`` when that is set.
``run`` execs the container's command locally, with only the environment the ``-e`` flags give
(plus the interpreter and PYTHONPATH the test names), so a variable that isn't passed
explicitly is not there. The other commands answer from the environment:

- ``ps``: the names in ``$FUKIKAE_FAKE_DOCKER_PS`` (comma-separated), one per line;
- ``rm``: exits ``$FUKIKAE_FAKE_DOCKER_RM_RC`` (default 0);
- ``inspect``/``container inspect``: succeeds only with ``$FUKIKAE_FAKE_DOCKER_STUCK`` set
  (the container is still there), else "No such container";
- ``image inspect`` and ``info``: ``$FUKIKAE_FAKE_DOCKER_DAEMON`` ``down`` or ``denied`` fail
  as an unreachable daemon or a refused socket would; ``image inspect`` also fails as a
  missing image with ``$FUKIKAE_FAKE_DOCKER_IMAGE`` ``missing``; ``info`` prints
  ``$FUKIKAE_FAKE_DOCKER_INFO`` (default: a docker root of ``/`` and the nvidia runtime).
"""

from __future__ import annotations

import json
import os
import sys

VALUE_FLAGS = {"--name", "--entrypoint", "--gpus", "--network", "--user", "-v", "-e", "-w",
               "--shm-size", "--ipc", "--ulimit", "--pids-limit", "--security-opt",
               "--cap-drop", "--tmpfs"}  # fmt: skip
DOWN = (
    "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
    "Is the docker daemon running?"
)
DENIED = (
    "permission denied while trying to connect to the Docker daemon socket at "
    "unix:///var/run/docker.sock"
)


def _daemon() -> int | None:
    state = os.environ.get("FUKIKAE_FAKE_DOCKER_DAEMON", "")
    if state == "down":
        print(DOWN, file=sys.stderr)
        return 1
    if state == "denied":
        print(DENIED, file=sys.stderr)
        return 1
    return None


def _run(argv: list[str]) -> int:
    env: dict[str, str] = {}
    entrypoint = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in VALUE_FLAGS:
            if a == "-e":
                k, _, v = argv[i + 1].partition("=")
                env[k] = v
            if a == "--entrypoint":
                entrypoint = argv[i + 1]
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            break
    cmd = argv[i + 1 :]  # after the image
    if entrypoint == "python" or (cmd and cmd[0] == "python"):
        cmd = [os.environ["FUKIKAE_FAKE_DOCKER_PYTHON"], *(cmd if entrypoint else cmd[1:])]
    env["PYTHONPATH"] = os.environ["FUKIKAE_FAKE_DOCKER_PYTHONPATH"]
    env["PATH"] = os.environ.get("PATH", "")
    os.execve(cmd[0], cmd, env)  # noqa: S606 - the test's interpreter and worker
    return 1


def main(argv: list[str]) -> int:
    log = os.environ.get("FUKIKAE_FAKE_DOCKER_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + "\n")
    if not argv:
        return 0
    cmd = argv[0]
    if cmd == "run":
        return _run(argv)
    if cmd == "ps":
        for name in filter(None, os.environ.get("FUKIKAE_FAKE_DOCKER_PS", "").split(",")):
            print(name)
        return 0
    if cmd == "rm":
        return int(os.environ.get("FUKIKAE_FAKE_DOCKER_RM_RC", "0"))
    if cmd == "inspect" or argv[:2] == ["container", "inspect"]:
        if os.environ.get("FUKIKAE_FAKE_DOCKER_STUCK"):
            print("running")
            return 0
        print(f"Error: No such container: {argv[-1]}", file=sys.stderr)
        return 1
    if argv[:2] == ["image", "inspect"]:
        rc = _daemon()
        if rc is not None:
            return rc
        if os.environ.get("FUKIKAE_FAKE_DOCKER_IMAGE") == "missing":
            print(f"Error response from daemon: No such image: {argv[-1]}", file=sys.stderr)
            return 1
        print("sha256:" + "0" * 64)
        return 0
    if cmd == "info":
        rc = _daemon()
        if rc is not None:
            return rc
        print(os.environ.get("FUKIKAE_FAKE_DOCKER_INFO", '/ {"nvidia":{},"runc":{}}'))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
