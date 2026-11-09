# Roadmap

Prioritized from the implementation and retained evidence reviewed on 2026-11-08.
This is **future work**, not a claim that the steps are implemented or verified.
The current capability/verification matrix and rerun commands are in
[VERIFICATION.md](VERIFICATION.md). Preserve existing tolerances, failed-row
denominators and historical measurements when extending coverage.

## Baseline and estimation assumptions

CPU/Metal fitting, all four shared models, initialization, Python bindings,
optional uncertainties, bounded NPY streaming and the local sanitizer/validation
gate are implemented and verified for the recorded local workloads. Do not
rebuild those features as roadmap tasks. Required **Metal System Trace** evidence
and shader-profiler metrics remain unavailable; device/platform generality and
several adversarial cases remain unverified. Performance capacity is not sustained
utilization: large-file runs observed one pending command, not three.

The pre-publication pass rechecked fresh PyPI/GitHub/Apple downloads, preserved
upstream notices, added MIT package metadata and closed the missing metal-cpp
archive-checksum gap. Header bytes matched the fresh archive and CMake checksum
accept/reject checks passed. These are no longer blocked by registry/DNS access.
Remaining tool-version enforcement and wider distribution checks stay below.
The publication pass also verified a wheel built from its source distribution,
exact notice payloads, isolated CPU64/Metal32 fitting/covariance and CPU-only
Python tests (60 passes, 14 expected Metal skips). These are local CPython 3.9.6
macOS arm64 results, not completion of the platform/minimum-OS matrix below.
The final full gate also passes 120 CTest executions, 74 Python tests, the full
notebook and 56 benchmark smoke cases with the required sanitizers/Metal layers.

Estimates are engineering hours for one developer familiar with C++20, Metal and
NumPy, including implementation, tests, review and documentation. They assume the
existing local environment, unchanged four-model scope unless specified, at most
two ordinary build/test workers, and access to authorized test machines for steps
that need them. Hardware/Xcode acquisition, external-service setup, unattended
multi-hour benchmark runtime and external review waiting time are excluded.
No host configuration changes, services or public publishing are authorized here.

The additive total is **236–414 hours** (midpoint 325 h), approximately **6–10.5
40-hour engineering weeks**. Budget **295–518 hours** with 25% contingency. Steps
1–8 account for **112–194 hours** and prioritize correctness, security,
verification and incomplete original profiling requirements before optimization
or new API features. Dependencies constrain ordering; effort is counted once in
each step.

## 1. Close adversarial CPU/GPU trajectory and identifiability gaps

- **Scope:** Replay frozen difficult initial guesses, paired peak blends,
  pseudo-Voigt mixing extremes, weak secondary peaks and low-SNR cases on CPU64,
  CPU32 and all forced/automatic GPU variants. Retain the original inputs, not
  only corrected initializer outputs.
- **Rationale:** Current corrected-seed matrices pass, but archived original-seed
  failures and nine stricter pseudo-Voigt diagnostics in the standalone v1
  baseline remain documented in [BENCHMARKS.md](BENCHMARKS.md). A successful
  corrected workload is not proof of unconditional parameter-identical fitting.
- **Dependencies:** Existing solver/model/initializer and benchmark fixtures;
  no other roadmap step.
- **Acceptance criteria:** Publish a deterministic adversarial corpus with every
  row retained; preserve the >99% gate in each existing single-peak group. All
  identifiable corpus cases meet the unchanged status/cost/parameter gates on
  every variant. Non-identifiable component-label/equal-cost ambiguities have
  explicit diagnostics and a documented domain, not silently widened tolerances,
  CPU fallback or truth-based seeds. Rejected trials preserve accepted state.
- **Estimated effort:** **24–40 hours**.

## 2. Harden and fuzz file-input and output transaction boundaries

- **Scope:** Fuzz the narrow NPY header/parser, dimensions, truncation and mmap
  paths; review same-descriptor header/size validation, path replacement,
  symlinks, disk-full/write/close errors and CLI rollback. Test concurrent mutation
  in isolated child processes only.
- **Rationale:** `MappedReader` currently inspects a pathname before opening its
  mapping descriptor, and size checks cannot guarantee immutable contents or
  prevent post-check truncation. The documented immutable-input contract is
  necessary; this is a hardening gap, not a demonstrated supported-input exploit.
- **Dependencies:** Existing malformed-file/rollback suites; no other step.
- **Acceptance criteria:** A bounded reproducible fuzz corpus runs under
  ASan/UBSan with no memory errors/hangs. Header and payload checks refer to the
  opened object; unsupported mutation has an explicit safe contract/alternative.
  Injected failures never overwrite preexisting outputs or delete unowned paths;
  atomicity and crash-durability guarantees are stated separately.
- **Estimated effort:** **16–28 hours**.

## 3. Enforce tool pins and maintain dependency integrity checks

- **Scope:** Enforce bootstrap tool versions against supported pins without
  global installation and make archive tamper-rejection tests a maintained check.
- **Rationale:** All FetchContent archives now have hashes, including metal-cpp;
  fresh-download, installed-header equivalence and CMake rejection checks passed
  during publication verification. Bootstrap still accepts any uv on PATH, and
  the manual tamper test does not yet run in the ordinary gate.
- **Dependencies:** Existing pinned archives and local FetchContent/bootstrap
  paths; no other step.
- **Acceptance criteria:** Automated intact/corrupted archive tests preserve
  rejection before use; dependency updates review new bytes/notices and hashes.
  CPU-only configuration still performs no Metal download/link. Tool-version
  mismatches are actionable and cached/local installations preserve
  HOME/TMPDIR/toolchain settings.
- **Estimated effort:** **2–4 hours**.

## 4. Establish a real OS, compiler and Python verification matrix

- **Scope:** Run the same CPU-only gate on Linux and another supported compiler,
  and installed-package tests on declared Python 3.9–3.12 versions; add a CI
  configuration only when the target runners are available and authorized.
- **Rationale:** Local CPU-only success on macOS is not Linux/Windows coverage.
  Only CPython 3.9.6/macOS arm64 is verified; POSIX mmap and shader/framework
  requirements limit what can honestly be advertised.
- **Dependencies:** Step 3 for authenticated clean builds; existing
  `scripts/check.sh --cpu-only` and `scripts/check_python.sh --cpu-only`.
- **Acceptance criteria:** Publish actual pass/skip/failure results for each
  claimed platform/version with <=2 workers. Linux CPU-only runs never fetch/link
  Metal; unavailable GPU jobs cannot turn green by skipping required tests.
  Fix failures or narrow support metadata rather than claiming an unrun matrix.
- **Estimated effort:** **20–36 hours**.

## 5. Make race and lifetime verification reproducible in the regular workflow

- **Scope:** Add an explicit, separate TSan check path for the pool, immutable
  Objective-C++ completion bridge, error draining, buffer leases, cache creation
  and Python concurrent-call ownership; qualify leak testing on a supported host.
- **Rationale:** Instrumented TSan checks passed historically, but are manual and
  not in `check.sh`; combined ASan/UBSan cannot detect data races. LeakSanitizer
  and an ASan-instrumented Python interpreter are not part of the current gate.
- **Dependencies:** Existing concurrency fixtures; step 4 for additional runtime
  platforms. Do not combine TSan with ASan or suppress batchfit races.
- **Acceptance criteria:** Repeatable stress tests pass or emit actionable
  failures, instrumenting both C++ and Objective-C++ callbacks; exception and
  cancellation paths drain without use-after-free. Leak checks distinguish driver
  caches from owned allocations, state exclusions and leave no helper processes.
- **Estimated effort:** **12–20 hours**.

## 6. Complete the required Metal System Trace overlap evidence

- **Scope:** Record the existing large-file pipeline with Instruments' Metal
  System Trace in an authorized full-Xcode environment, including CPU preparation,
  submission, completion and GPU command intervals.
- **Rationale:** Both original profiling and out-of-core requirements requested
  this trace. Existing Mach/application timelines prove measured overlap but are
  **not System Traces**; `xctrace` was unavailable and exited 72 here.
- **Dependencies:** User-provided working Xcode/xctrace with repository-local
  output/cache permissions; existing `PROFILING.md` reproduction commands.
- **Acceptance criteria:** Retain a valid trace outside Git's binary history,
  document tool/device/workload provenance and inspect actual preparation/GPU
  overlap. Report observed queue depth and idle intervals, not just three-slot
  allocation. If access remains unavailable, keep this step explicitly blocked;
  no sandbox bypass or replacement trace label counts as completion.
- **Estimated effort:** **4–8 hours**, excluding environment provision/waiting.

## 7. Measure shader occupancy and limiter hypotheses with real profiler data

- **Scope:** Profile representative v1/SIMD/group kernels using supported Xcode
  shader tooling: achieved occupancy, registers/spills, ALU/memory stalls and
  line-level costs where exposed.
- **Rationale:** Current public counters expose timestamps only. The ranked
  private-state/compute bottlenecks and iteration proxy are hypotheses/evidence
  of divergence, not measured register pressure or occupancy.
- **Dependencies:** Step 6's authorized profiling environment and a device/tool
  combination exposing the requested metrics.
- **Acceptance criteria:** Each claimed metric has its actual profiler source,
  capture and units; unavailable metrics stay unavailable. Reconcile a measured
  bottleneck ranking with phase timestamps, validate outputs against CPU, and
  make no performance change solely on an unmeasured occupancy explanation.
- **Estimated effort:** **10–18 hours**.

## 8. Extend numerical and uncertainty validity domains

- **Scope:** Sweep coordinate offsets/dynamic ranges, nonuniform grids,
  near-singular peaks, fixed/active constraints, normalized covariance transforms
  and low SNR; separate input quantization, optimizer error and estimation error.
- **Rationale:** The safe normalized recipe's 2e-4 measured gate and the
  10,000-realization coverage results are strong but restricted evidence.
  Coverage is presently resolved, interior, homoscedastic Gaussian data, not
  blends, active-bound intervals or arbitrary noise.
- **Dependencies:** Step 1's adversarial corpus and existing `NUMERICS.md` study.
- **Acceptance criteria:** Predeclare error/coverage gates and retain every
  attempted fit/unavailable estimate. Test physical/unit-coordinate Jacobian and
  full covariance transforms, including paired baseline coupling. Singular or
  boundary-invalid Gaussian estimates remain explicitly unavailable; no universal
  coverage or fast-math precision promise is inferred from passing interior cases.
- **Estimated effort:** **24–40 hours**.

## 9. Calibrate dispatch beyond the single M4 large-batch policy

- **Scope:** Measure N, M, model, device and cold/warm setup effects on forced
  variants; evaluate conservative CPU/GPU and kernel selection on another GPU.
- **Rationale:** Current kernel policy uses model/M only, calibrated on M4 large
  float32 batches. Python auto chooses Metal for float32 even when small N or cold
  startup makes CPU faster; existing crossover results are not universal rules.
- **Dependencies:** Steps 1, 4 and 7; authorized additional device measurements.
- **Acceptance criteria:** Identical stored inputs and randomized/order-balanced
  repetitions yield zero parity errors and selection within the existing 5%
  noise budget for the declared domain. Include cold setup and preparation in
  device-choice accounting; retain forced overrides and a documented policy for
  uncalibrated devices rather than assuming M4 thresholds transfer.
- **Estimated effort:** **20–36 hours**.

## 10. Test real pipeline utilization and memory budgets under varied I/O

- **Scope:** Compare one/three slots on unique noisy data, slow/fast consumers,
  supplied/data-derived guesses and inputs exceeding physical RAM where feasible;
  investigate preparation/layout changes only after profiling.
- **Rationale:** Fully written 10.24 GB inputs passed with bounded process RSS,
  but those files fit in the M4's 16 GiB RAM. Global filesystem-cache pressure and
  cold-cache behavior were not controlled; three slots were slower for float64.
- **Dependencies:** Steps 2, 5–7 and sufficient repository-local disk capacity.
- **Acceptance criteria:** Validate every output or an independently justified
  bounded reference scheme; record resident/shared/mapped/cache qualifications,
  actual pending depth and phase overlap. Show no N-scaled retained data/events
  and no universal speedup claim. Any budget/layout change passes tails, failure
  draining, validation layers and the reference tolerances first.
- **Estimated effort:** **16–28 hours**.

## 11. Broaden end-to-end benchmarks without extrapolating censored loops

- **Scope:** Run all-model unique-spectrum workloads including generation,
  per-fit feature guesses and realistic iteration variation; schedule complete
  SciPy loops on selected large cells when their wall-time cost is justified.
- **Rationale:** All native N cells were actually computed, but the main
  Lorentzian sweep cycles 1,024 patterns and excludes unique-spectrum generation
  and feature extraction. Twelve large SciPy cells are measured prefixes only.
- **Dependencies:** Steps 1, 8–10; explicit benchmark time/disk budget.
- **Acceptance criteria:** Save raw repetitions, completed counts, failure
  denominators, peak memory, CPU/GPU/wall phases and setup costs. Completed and
  censored SciPy rows stay distinct; publish no synthetic full-N time/speedup.
  Update crossover plots only from actual data, keeping previous figures as
  workload-specific historical evidence rather than rewriting their provenance.
- **Estimated effort:** **12–24 hours**, excluding unattended loop runtime.

## 12. Verify standalone distribution and downstream C++ integration

- **Scope:** Add/test native library header/export installation and downstream
  `find_package` consumption; extend wheel/sdist and installed-package checks
  across the supported Python/platform/minimum-OS matrix.
- **Rationale:** The current CMake install rule covers the Python extension,
  not a complete standalone C++ package. Fresh wheel-from-sdist, exact notices and
  isolated CPU/Metal execution now pass locally; the `macosx_27_0_arm64` wheel does
  not establish compatibility with every declared source-build OS or interpreter.
- **Dependencies:** Steps 3–4 and a stable documented support policy; public
  publishing and credentials remain separate authorized work.
- **Acceptance criteria:** A clean project-local prefix and separate consumer
  compile/link without source-tree paths; CPU package never requires Metal.
  Installed wheels contain the extension, stubs and typing marker, pass applicable
  API/ownership tests and have reproducible source provenance. No release is
  called portable based only on an editable macOS import.
- **Estimated effort:** **20–36 hours**.

## 13. Expose explicit normalization and bounded streaming in Python

- **Scope:** Design a small Python entry to existing C++ coordinate preparation
  and chunk-consumer fitting, with explicit precision/copy/bounds semantics.
- **Rationale:** These capabilities exist natively, but Python `fit` currently
  materializes full batch outputs/shared observations and exposes neither remedy
  flags nor coordinate preparation. This is new API integration, not a missing
  implementation of the C++ pipeline.
- **Dependencies:** Steps 2, 5, 8 and 12; maintain the default four-array result.
- **Acceptance criteria:** Original double inputs are normalized before float
  conversion; physical constraints/covariances transform correctly or reject
  unsupported coupled boxes explicitly. Streaming memory stays bounded as N
  grows, callback exceptions drain, GIL/ownership behavior is tested, and explicit
  Metal never becomes silent CPU fallback.
- **Estimated effort:** **24–40 hours**.

## 14. Add noise-aware fitting only behind an explicit statistical contract

- **Scope:** Consider independent-observation weights and then an explicitly
  chosen Poisson objective, reusing the CPU reference before GPU implementation.
  Do not silently reinterpret current chi2 or uncertainty output.
- **Rationale:** Synthetic Poisson data are supported, but fitting currently
  minimizes **unweighted squared residuals**. Low-count/heteroscedastic data can
  need a different objective and uncertainty model; this is a new feature, not
  evidence that existing Gaussian least-squares coverage is wrong.
- **Dependencies:** Steps 1, 8 and 13 plus reviewed objective/DOF semantics.
- **Acceptance criteria:** CPU analytic derivatives/algebra match an independent
  oracle; every GPU variant meets the unchanged precision/failure gates. Zero,
  invalid and extreme weights/counts have explicit behavior, all denominators
  remain visible in coverage tests, and old default results/statuses stay intact.
- **Estimated effort:** **32–56 hours**.
