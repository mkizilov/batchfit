# batchfit

C++20 infrastructure for fitting batches of small independent nonlinear models
on CPU and Apple Metal. See [DESIGN.md](DESIGN.md) for the fitting API and solver
contract. **This first scaffold implements residual compute smoke tests, not a
nonlinear fitter.**

## Local setup

Requires a C++20 compiler and Python 3.9+. On macOS, Command Line Tools provide
Clang and framework headers; Metal executes on the default GPU. No offline shader
compiler is needed. Do not install or alter system tools from this repository.

Install uv **0.9.8** locally if it is not already supplied. For Apple Silicon,
with shell network access:

```sh
mkdir -p .tools .agent-local
curl -fL https://github.com/astral-sh/uv/releases/download/0.9.8/uv-aarch64-apple-darwin.tar.gz -o .agent-local/uv.tar.gz
tar -xzf .agent-local/uv.tar.gz -C .tools
export PATH="$PWD/.tools/uv-aarch64-apple-darwin:$PATH"
rm .agent-local/uv.tar.gz
sh scripts/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
```

Use the corresponding pinned uv archive on other platforms. Bootstrap creates
`.venv` using uv and installs exact development versions from `pyproject.toml`;
it does not download a Python interpreter. Preserve supplied project-local
HOME/TMPDIR/cache settings. Outside orchestration, set them under `.agent-local/`
before installing anything. CMake downloads GoogleTest/Benchmark in `build/` and
Apple's metal-cpp in `third_party/`; the CPU backend does not depend on metal-cpp.

```sh
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
# macOS: equivalently configure with -DBATCHFIT_METAL=ON (the macOS default)
cmake --preset metal
cmake --build --preset metal
ctest --preset metal
sh scripts/check.sh
```

Both presets cap build and test concurrency at two. For a minimal CPU-only
library build without downloading test/benchmark dependencies:

```sh
cmake -S . -B build/minimal -G Ninja -DBATCHFIT_METAL=OFF -DBATCHFIT_BUILD_TESTS=OFF -DBATCHFIT_BUILD_BENCHMARKS=OFF
cmake --build build/minimal --parallel 2
```

Formatting: `python3 tools/check_format.py`; apply fixes with
`clang-format -i <changed-source-files>`. Static analysis:
`python3 tools/run_tidy.py build/cpu` (also `build/metal` on macOS). Never format
or analyze downloaded third-party code as project source.

## Offline diagnostic check

When installers cannot reach their hosts, the existing system Clang can still
exercise the CPU library and public header without CMake or external frameworks:

```sh
mkdir -p .agent-local/check
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -DBATCHFIT_HAS_METAL=0 -Iinclude -Isrc src/cpu/reference.cpp tests/cpu_smoke.cpp -o .agent-local/check/cpu_smoke
.agent-local/check/cpu_smoke
rm .agent-local/check/cpu_smoke
rmdir .agent-local/check
```

This is **not** a full test pass. Initial implementation environment: shell DNS
failed for GitHub, PyPI and developer.apple.com; uv/CMake/Ninja/clang-tidy were
absent. Consequently dependency installation, uv environment creation, CMake,
GoogleTest/Benchmark and Metal execution could not be verified. Re-run the full
gate after connectivity and prerequisites are available; do not bypass isolation.

The offline CPU smoke did pass with Clang 21 after explicitly supplying
`-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; the bare compiler
did not discover its SDK headers in this environment. This flag reads the
existing system SDK and does not install or modify it. Formatting was checked
with the available Apple clang-format 21, not the pinned development version 18.
The CPU smoke also passed with AddressSanitizer and UndefinedBehaviorSanitizer.
Indexed diagnostics remained partial: no CMake compile database could be generated
and the Python diagnostic server was unavailable. Direct compiler and script
syntax checks, not that partial diagnostic snapshot, establish the subset results.
