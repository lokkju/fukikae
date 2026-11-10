#!/usr/bin/env python3
"""A stand-in for ``docker`` in the indextts2_trt tests: ``run`` execs the container's command
locally, with only the environment the ``-e`` flags give (plus the interpreter and
PYTHONPATH the test names), so a variable that isn't passed explicitly is not there. ``rm``
and ``image`` succeed and do nothing."""

from __future__ import annotations

import os
import sys

VALUE_FLAGS = {"--name", "--gpus", "--network", "--user", "-v", "-e", "-w", "--shm-size",
               "--ipc", "--ulimit"}  # fmt: skip


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "run":
        return 0
    env: dict[str, str] = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in VALUE_FLAGS:
            if a == "-e":
                k, _, v = argv[i + 1].partition("=")
                env[k] = v
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            break
    cmd = argv[i + 1 :]  # after the image
    if cmd and cmd[0] == "python":
        cmd[0] = os.environ["FUKIKAE_FAKE_DOCKER_PYTHON"]
    env["PYTHONPATH"] = os.environ["FUKIKAE_FAKE_DOCKER_PYTHONPATH"]
    env["PATH"] = os.environ.get("PATH", "")
    os.execve(cmd[0], cmd, env)  # noqa: S606 - the test's interpreter and worker
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
