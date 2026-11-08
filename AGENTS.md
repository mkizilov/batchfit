# batchfit: project boundaries and working agreement

- Work only in `/Users/mk-rune/test_projects/13-batchfit`. This is one of fourteen independent repositories.
- Do not read, modify, clean up, or run commands in sibling repositories or the user's configuration.
- Work autonomously: choose sensible defaults and document assumptions; do not ask questions or delegate.
- Never install anything globally or use sudo, Homebrew, launchd, host credentials, or host Docker sockets.
- Dependencies, toolchains, caches, browser binaries, SDKs, temporary files and test data stay in this repository. Preserve the supplied project-local environment. Python packages go in `.venv`.
- `.agent-local/` is ignored, project-local build/runtime state. `.opencode/` is managed orchestration configuration; do not change or remove it or bypass the sandbox.
- Use existing system tools read-only, or install pinned tools under this repository. Do not change shell profiles, user settings, Keychain, or global Git settings.
- Bind test servers to loopback; use ports 22200–22299. Limit build/test workers to two unless a required benchmark explicitly needs more.
- Docker-dependent integration tests must remain reproducible but be reported as unverified here if they require access outside this repository. Do not bypass this boundary.
- Do not touch attached hardware, enroll services, flash firmware, install agents, or alter OS settings. Supply manual checklists for such steps.
- Implement only the current numbered prompt, preserving prior behavior. Run appropriate checks, fix failures, and report genuine limitations without fabricating test or benchmark results.
- Before finishing every numbered turn: stop its servers/watchers, remove scratch outputs, update relevant documentation, and make a local conventional-commit Git commit. Do not push.
- End each turn with a short summary of changes, checks actually run, remaining limitations, and commit hash. The orchestrator will provide the next prompt only after this turn ends.

Extend this file with the project's language-specific architecture and verification commands while retaining these boundaries.

## Architecture

- C++20 library target `batchfit::batchfit`; public headers in `include/batchfit/`.
- `src/cpu/` owns the numerical reference; `src/metal/` owns metal-cpp host code
  and `.metal` shaders. Internal headers stay in `src/`. Python bindings are later.
- **The CPU reference is the source of truth. Every GPU result is tested against
  it.** Never loosen numerical tests merely to make a backend pass.
- Check and report every Metal error with context: shader compilation, kernel
  lookup, pipeline creation, resource/encoder allocation and command-buffer
  completion status. An explicit Metal request must not silently fall back.
- The initial residual smoke primitive is not the LM fitting implementation.
  Follow `DESIGN.md` for the eventual public fitting contract and precision rules.
- The single-fit CPU reference is `include/batchfit/cpu/solver.hpp` (header-only
  for model/scalar template instantiation). Keep all workspace fixed-size and on
  the stack. Preserve checked Cholesky, exact diagonal damping, best accepted
  state, projected box constraints and the documented stopping/status semantics.
- Model/noise/scalar recovery, all failure statuses and allocation instrumentation
  must pass for solver changes. GPU/batch fitting remains separate future work.
- Automatic guesses live in `include/batchfit/cpu/initial_guess.hpp`, separately
  from LM. Preserve edge-median baselines, centroid/FWHM width conventions and
  documented positive-emission/grid requirements. Never silently subtract a
  baseline from solver inputs or change existing model parameter counts.
- The fixed noisy convergence study must exceed 99% in every single-peak
  model/scalar/noise/SNR group. Keep failed guesses in denominators, report paired
  convergence and signal-recovery rates honestly, and never use truth to seed fits.
- Model structs in `include/batchfit/models.hpp` must compile as both C++ and
  MSL: no C/C++ library includes or dynamic storage, thread-local parameter/output
  arrays in shaders, and no shader double types or string metadata. Keep formulas,
  names, parameter order and width conventions documented in `DESIGN.md`.
- Every analytic model Jacobian must pass central finite differences in float
  and double. The standalone `tests/models_check.cpp` and GoogleTest wrappers
  share the same full checks. The Metal preset also runs the independent native
  SDK oracle `tests/models_metal.mm` (test-only Objective-C++, not a backend).
- CMake 3.24+ presets isolate CPU and Metal builds. `BATCHFIT_METAL=ON` is the
  macOS default; `OFF` must never fetch metal-cpp or link Apple frameworks.
- GoogleTest and Google Benchmark use pinned FetchContent archives; metal-cpp
  downloads into `third_party/`. Keep third-party code out of formatting/tidy.
- Synthetic generation and loading use bounded fit-major chunks, not full-N
  buffers. Preserve fixed-seed output across chunk sizes and parameter/noise RNG
  independence. Follow `DESIGN.md` for linear RMS SNR and scaled Poisson semantics.
- NPY I/O is our narrow v1.0 float32 C-order implementation. Validate file headers,
  shape/byte counts and integer overflow; do not add implicit dtype conversions.
  The CLI must refuse preexisting output directories and roll back its own failures.

## Setup and verification

Keep uv 0.9.8 local (see README); `sh scripts/bootstrap.sh` creates `.venv` using
uv and exact tool pins in `pyproject.toml`. Preserve supplied environment paths.
Use `export PATH="$PWD/.venv/bin:$PATH"` before manual build commands.

```sh
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
cmake --preset metal       # macOS only
cmake --build --preset metal
ctest --preset metal
python3 tools/check_format.py
python3 tools/run_tidy.py build/cpu
python3 tools/run_tidy.py build/metal  # macOS only
python3 tools/check_generator.py build/cpu/batchfit_generate
sh scripts/check.sh
```

Use `clang-format -i <files>` for maintained C++/Metal files; `.clang-format` is
authoritative. `clang-tidy` uses `.clang-tidy` plus each build's compile database.
`tools/run_tidy.py` serially checks library code, CPU model/solver/initial-guess
and synthetic test instantiations, and the generator CLI (including Metal host code).
All ordinary builds/tests have at most two workers. README includes standalone
CPU/MSL model and synthetic commands for offline verification; these do not
replace the full gate.

**All checks pass** means `scripts/check.sh` exits zero: formatting is clean,
both required presets configure/build, every CTest/GoogleTest passes, clang-tidy
has no selected warnings, and the bounded Google Benchmark smoke completes.
The gate also runs independent Python stdlib CLI/NPY checks. Standalone synthetic
tests use the exact same suite as GoogleTest; README gives offline commands.
Metal is required on macOS, not skipped when the device is missing; elsewhere
the CPU-only gate applies. Missing dependencies or unavailable device access are
explicit limitations, never a green gate. The README offline smoke command is a
diagnostic subset only. No Docker or server process is needed for this scaffold.

## Definition of done for every future task

1. Implement only the current numbered task and preserve earlier behavior.
2. `scripts/check.sh` passes; if an environment constraint prevents this, state
   exactly which commands failed or were not run, with actual subset results.
3. Update relevant documentation and remove temporary outputs/stop owned processes.
4. Give a short summary of changes, real checks and remaining limitations, and
   make a local conventional-commit Git commit containing only intended changes.
   Include its hash at handoff. Do not push.
