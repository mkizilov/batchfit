#!/bin/sh
set -eu
if [ ! -f CMakePresets.json ]; then
    echo 'Run scripts/check.sh from the repository root.' >&2
    exit 1
fi
export PATH="$PWD/.venv/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CTEST_PARALLEL_LEVEL=2
for tool in python3 cmake ninja clang-format clang-tidy; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool missing: $tool. Run scripts/bootstrap.sh first." >&2
        exit 1
    fi
done
python3 tools/check_format.py
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
python3 tools/run_tidy.py build/cpu
build/cpu/batchfit_bench --benchmark_min_time=0.01s
if [ "$(uname -s)" = Darwin ]; then
    cmake --preset metal
    cmake --build --preset metal
    ctest --preset metal
    python3 tools/run_tidy.py build/metal
fi
echo 'All checks pass: formatting, CPU tests, clang-tidy, benchmark smoke, and Metal tests on macOS.'
