# Verification and support status

This guide distinguishes **implemented** behavior (present in source),
**verified** behavior (actually checked on a stated workload/environment),
**partial** behavior/evidence (only part of a broader claim is established), and
**unverified** behavior (no successful relevant execution). A verified fixture is
not proof for arbitrary data, hardware or concurrency patterns. Future work is
prioritized in [ROADMAP.md](ROADMAP.md), not silently counted as complete.

## Local gate

From the repository root, preserve supplied HOME/TMPDIR/cache/toolchain settings:

```sh
export PATH="$PWD/.venv/bin:$PATH"
sh scripts/check.sh
sh scripts/check.sh --cpu-only      # explicit subset, not a macOS GPU pass
uv lock --check
git diff --check
```

Bootstrap instructions and pinned tool versions are in [README.md](README.md).
Both commands use at most two ordinary build/test workers. The complete gate
requires Metal on macOS and fails rather than silently skipping a missing device.
Elsewhere it runs the CPU-only path; that path was verified locally on macOS,
**not** presented as a Linux or Windows execution.

The gate currently performs:

1. `tools/check_format.py`, CPU configure/build/CTest, then all CPU CTests again
   in a fresh combined **AddressSanitizer/UndefinedBehaviorSanitizer** build.
2. CLI/NPY generator checks, CPU clang-tidy and bounded Google Benchmark smoke.
3. On macOS, Metal configure/build and **all Metal-preset CTests normally and
   again under ASan/UBSan**, both with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
4. Metal host clang-tidy, fitting/profiling/cooperative/pipeline/numerics smokes,
   typed-output CLI tests and the safe normalized numerical recipe gate.
5. Rebuilt editable Python bindings, pytest including SciPy agreement and full
   uncertainty coverage, binding clang-tidy, and actual IPC-kernel notebook
   execution with real-NPY/tail/error checks. The full Metal path enables both
   validation layers for Python tests and the notebook too.
6. On the full Metal path, 56 all-model/backend benchmark smoke cases, plot
   replay/error detection/censoring checks and benchmark-binding clang-tidy.

ASan/UBSan findings halt CTest; both C++ and Objective-C++ host sources are
instrumented. The sole UBSan ignorelist entry is Apple's intentional Objective-C
nil-message implementation in `third_party/metal-cpp/Foundation/NSObject.hpp`;
**no batchfit code is excluded**. LeakSanitizer is disabled, and a normally tested
Python extension is not an ASan-instrumented Python interpreter. TSan remains a
separate manual check; do not combine it with ASan. See
[the callback/race qualification](PROFILING.md#sanitizer-commands-and-qualification).

Temporary sanitizer builds under `.agent-local/check-sanitizers-*` are removed
on normal success/failure and handled signals; untrappable SIGKILL can leave a
directory requiring inspection after its owner has ended. Successful notebook
checks stop their IPC kernels and remove runtime resources. Preexisting local
tools/caches and requested profiling captures are preserved, not treated as junk.

## Executed local evidence

The complete gate was rerun and passed in the **2026-11-08 pre-publication pass**
(source base `2764a59` plus the licensing/integrity/privacy changes) on
**Mac16,10 / Apple M4, macOS arm64, CPython 3.9.6**, using Apple Clang 21 and
project-local pinned tools. Executed coverage:

- **25 CPU + 35 Metal CTests**, each normally and again under combined
  ASan/UBSan: **120 CTest executions**, all passed. Metal API/shader validation
  applies to both Metal executions; no reported validation findings were hidden.
- **74 Python tests** passed with both Metal validation layers, including the
  10,000-realization/model/backend coverage tests, 1,000/model SciPy comparisons
  and installed-license metadata/text regression check.
- Full **512×512×129** notebook: 262,144 submitted pixels, all **262,132 healthy**
  pixels converged and **12 defects** flagged; real read-only NPY cases also pass.
- The **56-case** bounded benchmark smoke, generator/pipeline/numerics checks,
  formatting and native/binding clang-tidy passed. Lock and Git whitespace checks
  passed separately. Debug smoke timing is not new performance calibration.

This pass also rebuilt the CPU-only Python package: **60 passes / 14 expected
Metal-only skips**, with binding clang-tidy. The full gate restored the
Metal-enabled editable install. Earlier validation separately passed the full
CPU notebook, deliberate invalid-compiler-flag failure/cleanup check and README
Python/C++ quickstarts; those historical checks are not claimed as extra reruns
in the latest gate.

## Additional pre-publication checks

- Fresh project-local PyPI installation of pinned pybind11 3.0.1 and HTTPS
  downloads from GitHub and Apple succeeded. Earlier DNS/registry restrictions
  are historical, not a current reason to skip a build. Existing uv 0.9.8,
  CMake 3.31.6 and clang-tidy 21.1.6 remain local and pinned.
- GoogleTest/Benchmark archive hashes matched their existing pins. The new
  metal-cpp SHA-256 pin matches a fresh archive and all 98 installed files byte
  for byte; CMake accepted the correct digest and rejected a deliberately wrong
  one. This verifies reviewed-byte reproducibility, not a vendor-signed digest.
- Built the source distribution, then a wheel **from that distribution** using
  the normal backend. Both contain the exact project/upstream notice files,
  MIT metadata and no protected configuration, local environments or build/cache
  payloads. An isolated install, without editable redirection, passed CPU64 and
  Metal32 fitting/covariance with API/shader validation. The inspected wheel is
  `cp39-cp39-macosx_27_0_arm64`; it is **not** proof of macOS 15 binary compatibility
  or a cross-version/platform distribution matrix.
- Reachable-history review examined **329 Git blobs across all existing refs**
  (about 8.9 MB) with private-key, common provider-token, credential-URL and
  literal-secret patterns; no such findings or tracked dependency caches/private
  key files were found. This bounded pattern review is not a formal security
  audit or a proof that every possible secret format is absent. The final staged
  review also covered all **160 tracked files**, with no secret-pattern,
  personal-working-path, explicit creation-attribution or cache/private-file
  findings; standard copyrights and upstream notices remain intact.
- Current benchmark metadata and notebook setup output now avoid personal
  absolute paths; smoke checks guard new outputs. Earlier versions remain in
  Git history: `925d148:AGENTS.md`,
  `30161ed:bench/results/metadata.json`, and
  `aef25d0:examples/hyperspectral_demo.ipynb`. These contain local working paths,
  not detected credentials. History and author/committer metadata were not
  rewritten; publication must account for that historical privacy caveat.

## Capability and evidence matrix

| Area | State and established evidence | Boundary / remaining gap |
|---|---|---|
| Shared four models/Jacobians | **Implemented, verified** in C++ float/double and actual MSL; central differences and independent SDK oracle | No arbitrary Python model/plugin mechanism; fixed GPU tables |
| CPU LM, boxes, failure statuses and pool | **Implemented, verified** recovery, accepted-state handling, bitwise per-row batch equivalence, allocation and lifecycle/error tests | Not a global-minimum or arbitrary ill-conditioning guarantee |
| Automatic guesses | **Implemented, verified** >99% in tested single-peak groups; paired study measured | Positive resolved emission/grid domain; paired convergence was 96.733%, not >99% |
| Metal v1/v2/automatic | **Implemented, verified** production metal-cpp, tails, bounds, strict fixtures, ownership, caches and validation layers | **Partial generality:** model/M dispatch calibrated on M4; historical original-seed failures and stricter v1 diagnostic exceptions remain labeled in BENCHMARKS.md |
| Float32 remedies | **Implemented, verified** explicit pre-conversion coordinates, Kahan/scaled solve options, safe/fast/pathology studies | Measured recipe, not lost-bit recovery or universal fast-math/precision guarantee; paired physical boxes couple under centering |
| Parameter uncertainties | **Implemented, verified** independent covariance algebra/SciPy tests and 120,000 Gaussian Monte Carlo fits with all denominators | **Partial statistical domain:** interior/resolved/homoscedastic data; no general low-SNR/blended/constrained coverage claim |
| Python bindings | **Implemented, verified** buffers/copies, GIL release, result lifetime, bounds, concurrency and CPU/Metal dispatch | Fitting calls serialize native work; Python does not expose native coordinate/remedy/streaming APIs; full-batch memory is not the out-of-core pipeline |
| NPY/CLI/streaming | **Implemented, verified** malformed inputs, overflow, typed conversion, tails, drains/rollback and fully written 10.24 GB files | Immutable inputs required; no concurrent file-mutation/security-fuzz proof or atomic crash-durability guarantee |
| Three-slot pipeline overlap | **Implemented, partially verified broader goal:** small tests reach three pending buffers; retained application/GPU timelines show CPU preparation overlap | Large-file runs reached only one pending command; bounded RSS does not bound OS cache or prove cold-cache/outside-physical-RAM behavior |
| Metal capture/profiling | **Implemented** opt-in capture; **verified** timestamps and a retained local `.gputrace` | **Unverified original requirements:** Metal System Trace, profiler occupancy/register/stall/line costs; recorded xctrace attempts failed, public counters supplied timestamps only |
| Performance sweep | **Verified** actual full-N native runs through 10 million fits, GPU/wall/preparation phases and saved plots | Main Lorentzian sweep repeats 1,024 patterns; 12 large SciPy cells are **partial**, not extrapolated full-N results; no new large sweep in this review |
| Notebook/real data | **Implemented, verified** synthetic full cube and read-only NPY input/error/tail branch | No experimental dataset or browser click-through; external Jupyter TLS/plugin downloads unverified |
| Distribution/platform support | **Verified** local editable builds, wheel built from sdist, exact license payloads and isolated CPU/Metal wheel execution | Linux/Windows, other chips, Python 3.10–3.12/minimum-macOS wheels and downstream C++ install/export package remain **unverified/incomplete** |
| Supply chain/security | **Verified** fresh downloads, all FetchContent archive hashes, metal-cpp header equivalence and manual digest rejection; bounded reachable-history secret-pattern review | No formal security audit, maintained tamper-test gate, fuzz campaign or general mutable-input safety claim; historical local paths remain in Git |

## Reproduce broader studies deliberately

- [BENCHMARKS.md](BENCHMARKS.md): Release CPU/cooperative/large-N/dense-file
  commands and historical provenance. Do not rerun a multi-GB study implicitly or
  overwrite retained evidence; generation/fitting/cache qualifications differ.
- [NUMERICS.md](NUMERICS.md): full CPU64 accuracy study, independent covariance
  checks, Monte Carlo tables, assumptions and retained failed experiments.
- [PROFILING.md](PROFILING.md): capture/TSan recipes and the exact unsuccessful
  xctrace commands. Application/Mach JSON is **not Metal System Trace**. Obtain an
  authorized suitable Xcode environment rather than changing host settings or
  bypassing sandbox restrictions.

Do not call indexed diagnostics a compiler pass: clangd currently lacks a root
compile-database marker, while CMake build databases exist and native compilation
and selected clang-tidy checks pass. Markdown has no registered diagnostic
producer; local link/heading checks are separate. Host process enumeration is
sandbox-restricted; verify owned command completion, kernel shutdown and exact
scratch/runtime cleanup instead of asserting an unrestricted host-process audit.
No CI runner, external service, LeakSanitizer or cross-platform result is implied
by the local gate. See [ROADMAP.md](ROADMAP.md) for prioritized closure criteria.
