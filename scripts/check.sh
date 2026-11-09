#!/bin/sh
set -eu
case "$*" in
    ''|'--cpu-only') ;;
    *) echo 'Usage: sh scripts/check.sh [--cpu-only]' >&2; exit 2 ;;
esac
if [ ! -f CMakePresets.json ]; then
    echo 'Run scripts/check.sh from the repository root.' >&2
    exit 1
fi
export PATH="$PWD/.venv/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CTEST_PARALLEL_LEVEL=2
export PYTHONDONTWRITEBYTECODE=1
metal=0
if [ "$(uname -s)" = Darwin ] && [ "${1-}" != --cpu-only ]; then
    metal=1
fi
for tool in python3 uv cmake ninja clang-format clang-tidy; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool missing: $tool. Run scripts/bootstrap.sh first." >&2
        exit 1
    fi
done
mkdir -p .agent-local
sanitizer_root=$(mktemp -d "$PWD/.agent-local/check-sanitizers-XXXXXX")
cleanup() { cmake -E rm -rf "$sanitizer_root"; }
trap cleanup 0
trap 'exit 130' INT
trap 'exit 143' TERM HUP
sanitized_cpp() {
    preset=$1
    enabled=$2
    directory="$sanitizer_root/$preset"
    flags='-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all'
    set --
    if [ "$enabled" = ON ]; then
        flags="$flags -fsanitize-ignorelist=$PWD/tools/sanitizer_ignorelist.txt"
        set -- "-DCMAKE_OBJCXX_FLAGS=${OBJCXXFLAGS:-} $flags" \
            "-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$PWD/third_party/metal-cpp"
    fi
    cmake -S . -B "$directory" -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DBATCHFIT_METAL="$enabled" -DBATCHFIT_BUILD_TESTS=ON -DBATCHFIT_BUILD_BENCHMARKS=OFF \
        -DBATCHFIT_BUILD_PYTHON=OFF \
        "-DCMAKE_CXX_FLAGS=${CXXFLAGS:-} $flags" \
        "-DCMAKE_EXE_LINKER_FLAGS=${LDFLAGS:-} -fsanitize=address,undefined" \
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$PWD/build/$preset/_deps/googletest-src" "$@"
    cmake --build "$directory" --parallel 2
    if [ "$enabled" = ON ]; then
        ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
            MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
            ctest --test-dir "$directory" --output-on-failure --parallel 2 --no-tests=error
    else
        ASAN_OPTIONS=halt_on_error=1:detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
            ctest --test-dir "$directory" --output-on-failure --parallel 2 --no-tests=error
    fi
}
python3 tools/check_format.py
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
sanitized_cpp cpu OFF
python3 tools/check_generator.py build/cpu/batchfit_generate
python3 tools/run_tidy.py build/cpu
build/cpu/batchfit_bench --benchmark_filter='cpu_residuals|cpu_batch_smoke' --benchmark_min_time=0.01s
if [ "$metal" = 1 ]; then
    cmake --preset metal
    cmake --build --preset metal
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ctest --preset metal
    sanitized_cpp metal ON
    python3 tools/run_tidy.py build/metal
    build/metal/batchfit_bench --benchmark_filter=metal_lm_smoke
    build/metal/batchfit_metal_profile --smoke
    build/metal/batchfit_cooperative_baseline --smoke
    build/metal/batchfit_pipeline_baseline --smoke
    build/metal/batchfit_pipeline_baseline --smoke --float32
    python3 tools/check_pipeline.py build/metal/batchfit_fit_npy
    python3 tools/check_numerics.py build/metal/batchfit_numerics_study
fi
if [ "$metal" = 1 ]; then
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 sh scripts/check_python.sh
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 python3 tools/check_demo.py
    python3 tools/check_benchmarks.py
    python3 tools/run_tidy.py build/benchmark-suite bench/bindings.cpp
else
    sh scripts/check_python.sh --cpu-only
    python3 tools/check_demo.py --device cpu
fi
echo 'All checks pass: formatting, clang-tidy, normal/ASan/UBSan C++ suites, applicable Metal API/shader validation, Python, notebook and benchmark smokes.'
