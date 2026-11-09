#!/bin/sh
set -eu
if [ ! -f CMakePresets.json ]; then
    echo 'Run scripts/bootstrap.sh from the repository root.' >&2
    exit 1
fi
# Preserve project-local HOME, TMPDIR and cache settings.
export UV_PYTHON_DOWNLOADS=never
if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Install uv 0.9.8 under .tools/ and put its directory on PATH.' >&2
    exit 1
fi
if [ ! -d .venv ]; then
    uv venv --python python3 .venv
fi
uv pip install --python .venv/bin/python -e '.[dev]'
