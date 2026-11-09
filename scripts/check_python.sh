#!/bin/sh
set -eu
case "$*" in
    ''|'--cpu-only') ;;
    *) echo 'Usage: sh scripts/check_python.sh [--cpu-only]' >&2; exit 2 ;;
esac
if [ ! -f pyproject.toml ]; then
    echo 'Run scripts/check_python.sh from the repository root.' >&2
    exit 1
fi
export PATH="$PWD/.venv/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never
export UV_CONCURRENT_BUILDS=2
export UV_CONCURRENT_INSTALLS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export PYTHONDONTWRITEBYTECODE=1
metal=OFF
required=0
if [ "$(uname -s)" = Darwin ] && [ "${1-}" != --cpu-only ]; then
    metal=ON
    required=1
fi
uv pip install --python .venv/bin/python --no-build-isolation --reinstall-package batchfit -e . \
    -Cbuild-dir=build/python-check -Ccmake.define.BATCHFIT_METAL="$metal"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
    BATCHFIT_REQUIRE_METAL="$required" python -m pytest -q
python tools/run_tidy.py build/python-check python/bindings.cpp
