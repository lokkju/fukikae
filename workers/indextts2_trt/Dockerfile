# syntax=docker/dockerfile:1.7
# The IndexTTS2 TensorRT worker image (D28 lever 2). Built locally by
# `fukikae backends setup indextts2_trt` and never pushed anywhere. It holds what the host
# doesn't: OpenMPI 4 (TensorRT-LLM links libmpi.so.40 and needs orted for its singleton init), a C
# compiler (Triton, which TensorRT-LLM imports, builds a small helper at first use),
# Python 3.12 and this directory's uv.lock synced into /opt/venv, and upstream index-tts at the
# pinned commit in /opt/index-tts (on PYTHONPATH, which gives both `indextts` and
# `backends.trt`). Models, engines and fukikae's own source are mounted at run time; nothing in
# the image downloads anything once it is built.
FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867

# Versions pinned to what noble and noble-updates served on 2026-11-10. A pin that comes from
# -updates or -security leaves the index when Ubuntu supersedes it; the build then fails at this
# step, and the fix is the new version here (a new project digest, so new engines too).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopenmpi3t64=4.1.6-7ubuntu2 openmpi-bin=4.1.6-7ubuntu2 \
      ca-certificates=20260816ubuntu0.24.04.2 gcc=4:13.2.0-7ubuntu1 libc6-dev=2.39-0ubuntu8.9 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

WORKDIR /opt/worker
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && uv sync --locked --no-install-project
# TensorRT-LLM's bindings link libpython3.12.so, which uv's Python keeps in its own lib/.
RUN ln -s "$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')/libpython3.12.so.1.0" \
        /usr/local/lib/libpython3.12.so.1.0 \
 && ldconfig

# Upstream index-tts at impl_indextts2.CODE_REVISION.
ADD --keep-git-dir=false \
    https://github.com/index-tts/index-tts.git#d9e41aac89fd00b3d71497fddb287b7f24613712 \
    /opt/index-tts

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/index-tts \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache

# The text normalizers build their grammars on first use and cache them (the Chinese one beside
# its code, the English one under XDG_CACHE_HOME), where the unprivileged worker can't write:
# build them here, as root, readable by everyone. (Loaded by path: importing the package would
# import TensorRT-LLM, which wants a GPU.)
ENV XDG_CACHE_HOME=/opt/cache
RUN python -c "import importlib.util as u; p = '/opt/index-tts/backends/trt/pipeline/text_processing.py'; s = u.spec_from_file_location('tp', p); m = u.module_from_spec(s); s.loader.exec_module(m); m.TextNormalizer().load()" \
 && chmod -R a+rX /opt/cache /opt/index-tts/backends/trt/pipeline/tagger_cache

WORKDIR /tmp
