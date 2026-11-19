# A container image for STILT projects: PYSTILT with every extra and the
# libraries for s3:// and gs:// buckets, installed from uv.lock. One image for
# a run on your own machine and for the pods of a Kubernetes Job.
#
# It holds no HYSPLIT. Download HYSPLIT from NOAA's Air Resources Laboratory
# (https://www.ready.noaa.gov/HYSPLIT.php) and mount its folder at
# /opt/hysplit, whose exec/ is on PATH:
#
#   docker build --build-arg VERSION=0.1.0 -t pystilt .
#   docker run --rm \
#     -v /path/to/hysplit:/opt/hysplit:ro \
#     -v /path/to/met:/met:ro \
#     -v "$PWD/my_project:/project" \
#     pystilt stilt run /project
#
# docs/guides/containers.rst has more, and how to build an image with your
# HYSPLIT inside it.

FROM python:3.13-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
# runtraj comes from git until its first release ([tool.uv.sources]).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
# setuptools-scm reads the version from git, and the build leaves .git out.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYSTILT=${VERSION}
WORKDIR /src
# The dependencies first, so a change to the code alone reuses this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    --extra complete --extra s3 --extra gcs
COPY . .
RUN uv sync --locked --no-dev --no-editable \
    --extra complete --extra s3 --extra gcs

FROM python:3.13-slim-bookworm
# tini runs as process 1 and passes a stop signal (docker stop, or
# Kubernetes' SIGTERM) on to stilt, which ends its runs and exits. The base
# image's pip goes: nothing here uses it, and the old copies of urllib3,
# msgpack and setuptools it carries inside it are what a scan finds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y pip
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:/opt/hysplit/exec:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=UTC
RUN useradd --create-home --uid 10001 stilt
USER 10001
WORKDIR /home/stilt
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["stilt", "--help"]
