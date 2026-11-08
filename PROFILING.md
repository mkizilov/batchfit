# Metal profiling and out-of-core pipeline

## Out-of-core pipeline (turn 10)

The production metal-cpp pipeline emits streamed application phase events to
`profiling/pipeline_events.json` (Chrome/Perfetto JSON). This is **not a Metal
System Trace, shader-profiler output, occupancy measurement or hardware limiter
counter**. CPU timestamps use `mach_absolute_time` with the native timebase;
command-buffer GPUStartTime/GPUEndTime are documented in the same system Mach
domain. Events label next-chunk read-ahead requests, preparation (with map/convert,
initial-copy/guess and transpose children), submit, GPU command intervals and
readback/consumer work. Producer is thread lane 1, readback worker lane 2, GPU
intervals lane 3. These are logical lanes, not invented OS thread identities.

Compare CPU preparation for chunk k+1 against the GPU interval of chunk k. The
shared-buffer ring keeps up to three committed buffers pending, while readback
for k-1 uses another slot. A slow disk/producer can leave fewer than three pending;
configured depth and **observed peak pending** are reported separately. Overlap
does not demonstrate occupancy, three simultaneous kernels, or complete GPU
saturation. GPU command intervals may overlap; use their union for coverage,
not the sum as an occupancy estimate.

Metal System Trace was attempted again, including a `--no-cache` retry:

```sh
xcrun --no-cache xctrace record --template 'Metal System Trace' --time-limit 15s \
  --output profiling/pipeline-system.trace --launch -- \
  "$PWD/build/metal-release/batchfit_pipeline_baseline" --smoke
```

Both attempts exited **72**: `unable to find utility "xctrace"`. Even `--no-cache`
attempted a denied `/var/folders/.../xcrun_db-*` cache write, despite the supplied
project-local TMPDIR. No host file write succeeded, developer-tool selection or
settings were changed, or sandbox boundary was bypassed. Consequently **the
requested System Trace overlap check remains unverified here**. The retained
application timestamps are independent available evidence, not a relabeled
substitute. With a user-provided full Xcode environment, rerun that command
without `--smoke` (allow a sufficiently long recording), then inspect CPU prepare
and command-buffer intervals in Instruments; no service/hardware setup is required
by the library itself.

The Release runner writes a real dense ~10 GB file under its owned
`.agent-local/pipeline-benchmark-*` directory, validates every output, and removes
that directory on normal completion/failure. It refuses an existing event file:
archive an earlier `profiling/pipeline_events.json` deliberately before rerunning.
`BATCHFIT_GPU_CAPTURE` must be disabled for throughput work; first-dispatch capture
intentionally waits for that dispatch to preserve the earlier capture contract.
See BENCHMARKS.md for exact workload, measured throughput, memory and limitations.

### Final application timestamp evidence

The retained final-source JSON covers every chunk and fit:

| Trace | Chunks / fits | GPU interval union s | CPU prepare(k+1) overlapping GPU(k), s | Prepare + read-ahead request overlapping GPU, s |
|---|---:|---:|---:|---:|
| `profiling/pipeline_events.json` (float64) | 123 / 312,500 | 2.824693 | 0.779917 | 2.822912 |
| `profiling/pipeline_float32_events.json` | 198 / 5,000,000 | 5.967427 | 4.575618 | 5.958495 |

Intervals were intersected in the common Mach domain, not inferred from separate
duration totals. Open the JSON in a local Chrome-trace/Perfetto viewer and inspect
CPU lane 1 preparing the next chunk while lane 3 contains the previous GPU
interval; lane 2 consumes completed outputs. Every result was checked against CPU.
There are 984 / 1,584 complete phase events, streamed rather than held by the
pipeline. The JSON provenance explicitly says **NOT Metal System Trace**.

This proves application preparation/GPU-command overlap for those workloads.
It does **not** prove continuous GPU saturation: final large-file runs observed
only one pending command at a time, despite a three-slot capacity and small tests
reaching three. Disk/page-in time dominated; three slots cannot fix an input
bandwidth bottleneck. The float64 run was slower with three slots, while float32
improved modestly; BENCHMARKS.md retains both. No occupancy, ALU/memory limiter or
shader line costs are inferred, and no successful System Trace is claimed.

### Sanitizer commands and qualification

The pipeline/mapping suite passes ASan/UBSan plus Metal API/shader validation,
and TSan with both C++ and the Objective-C++ callback bridge instrumented. Apple's
`NSObject.hpp` nil-message implementation triggers UBSan's C++ null-receiver
check even for empty `NS::SharedPtr`; the initial unsuppressed run reported it.
`tools/sanitizer_ignorelist.txt` excludes **only that upstream inline header**
from UBSan, not batchfit code, numerical tests, address checks or thread checks.
No TSan suppression is used. LeakSanitizer was not run.

```sh
flags="-fsanitize=address,undefined -fno-omit-frame-pointer -fsanitize-ignorelist=$PWD/tools/sanitizer_ignorelist.txt"
cmake -S . -B .agent-local/check/asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DBATCHFIT_METAL=ON -DBATCHFIT_BUILD_BENCHMARKS=OFF -DBATCHFIT_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="$flags" -DCMAKE_OBJCXX_FLAGS="$flags" \
  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build/metal/_deps/googletest-src"
cmake --build .agent-local/check/asan --target batchfit_pipeline_check batchfit_mmap_check --parallel 2
UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=0 MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  .agent-local/check/asan/batchfit_pipeline_check
```

Use a separate build with `-fsanitize=thread -fno-omit-frame-pointer` for both
CXX/OBJCXX flags, `-fsanitize=thread` for executable linking and
`TSAN_OPTIONS=halt_on_error=1`; do not combine TSan with ASan. Scratch sanitizer
builds are removed at handoff. The ordinary `scripts/check.sh` gate is independent
and includes both presets, all CTests, selected clang-tidy checks and CLI checks.
Indexed diagnostics remain partial without a root compile-database marker;
actual builds and clang-tidy against the build databases are authoritative here.

## Status: measured experiments, unavailable Xcode-only metrics

- **Captured:** `profiling/kernel-v1.gputrace`, a real 14-file, 8,892,530-byte
  GPU trace bundle. It contains a production v1 TwoLorentzian dispatch with
  N=4,096, M=512 and 256-thread groups. It is ignored by Git but retained for
  inspection in Xcode. Delete/archive it only after that inspection is no longer
  needed; capture deliberately refuses to overwrite it.
- **Measured:** 100 profiling cases on Apple M4: 96 group/layout/storage cases
  plus four oracle-ordering diagnostics. Every result is recorded in
  [`profiling/results.csv`](profiling/results.csv), including timing, correctness,
  iteration distribution, quantization and hashes. All cases pass CPU comparisons
  for the same stored input values.
- **Not recorded:** Metal System Trace. The exact `xcrun xctrace record` attempt
  below failed with exit 72: selected Command Line Tools cannot find `xctrace`.
  `xcrun` also reported a denied external cache-file write. No workaround,
  developer-directory change, service installation or sandbox bypass was attempted.
- **Not observed:** Xcode shader-profiler occupancy, register allocation/spill
  counts, ALU/memory-stall counters or per-line execution cost. Xcode profiling
  is unavailable here. **Do not interpret the hypotheses below as those metrics.**
- Public Metal counter enumeration returned **only `timestamp / GPUTimestamp`**.
  Stage-boundary sampling is supported; dispatch-boundary sampling is not.
  The timestamp counter validates duration, not occupancy or an ALU/register
  diagnosis. No missing counter values are replaced with estimates.

At the time of this profiling run, metal-cpp compilation was blocked by DNS and
missing local tools. Turn 9 subsequently verifies production metal-cpp builds,
CTest/GoogleTest and static analysis. The historical GPU measurements and capture
use the installed-SDK Objective-C++ profiling driver, with the production shader
algorithm and the same checked shared-buffer/capture-path policies. Production
uses `MTL::CaptureManager`; the SDK driver uses the equivalent MTLCaptureManager.

## Method and boundaries

These frozen profiling inputs/results were recorded before turn 9's data-only
paired shoulder-separation fix. Current reruns may use different paired initial
arrays; do not silently relabel this archived CSV/trace. The new fair v1-v2
comparison and its corrected seed rule are in DESIGN.md and
`profiling/cooperative_results.csv`.

Hardware matches `BENCHMARKS.md`: Mac model identifier **Mac16,10**, **Apple M4**,
10 CPU cores, 16 GiB RAM, macOS 27.2 arm64, Apple Clang 21.0.0. Compiler flags are
`-std=c++20 -O3 -DNDEBUG -pthread`, with safe/precise float32 Metal math and no
fast-math. No affinity, clock, OS or developer settings were changed.

N=100,000, M=512 for the full sweep, using the exact CPU-v1 workload configuration
and seeds from `bench/cpu_workload.hpp`. Three identical-input calls per case;
tables show median **GPU milliseconds**. There is no separate warm-up call: the
median suppresses a single cold launch. Fixed sweep order is point/float,
fit/float, point/half, fit/half, then oracle order. Results are observations, not
confidence intervals; small differences can include frequency/cache/order noise.
No API/shader validation or capture was enabled for the timed sweep.

Six legal threadgroup sizes were tried: 32, 64, 128, 256, 512 and 1024. Every
pipeline reported execution width **32**, maximum legal group **1024** and static
threadgroup memory **0 bytes**. These limits do **not** establish achieved
occupancy or private-register consumption. Production remains at its original
pipeline-derived group size (256 on this Mac); no tuning result was silently
made a new default.

The experiment changes **observation y storage only**. x, initial/final parameters,
normal matrices, factorization and all LM arithmetic remain float32. Half inputs
are IEEE binary16 `_Float16`/MSL `half`, converted to float on load. Point-major
uses `y[m*N+i]`; fit-major uses `y[i*M+m]`. Profiling-only template instantiations
share the exact solver body. The original kernel names still select float32,
point-major storage. No solver tolerances or CPU algorithms changed.

For reference against the required all-core CPU baseline at this exact N/M,
original-order point/float group-256 profiling wall rates are:

| Model | CPU v1 10-worker fits/s | Profiling wall fits/s | Contextual ratio |
| --- | ---: | ---: | ---: |
| Gaussian | 649,667 | 4,308,217 | 6.63x |
| Lorentzian | 1,142,094 | 3,712,911 | 3.25x |
| PseudoVoigt | 404,679 | 2,793,735 | 6.90x |
| TwoLorentzian | 270,610 | 786,075 | 2.90x |

Timing boundaries differ: profiling reuses allocated shared outputs and adds
timestamp-counter attachments; CPU v1 includes output allocation. Neither includes
generation/initial guesses, and profiling excludes packing/transposition. These
ratios provide baseline context, **not raw-image end-to-end speedup claims**.

Data creation, CPU reference fits, layout packing, half conversion and pipeline/
buffer creation are outside timing. GPU time uses command-buffer timestamps;
wall time includes encoding/commit/wait but not counter resolution or result
checking. Two stage-boundary timestamp samples independently measure the encoder;
resolved counter duration agrees with command duration within **41.3 ns** over
all 100 cases. This confirms timing, not shader utilization.

Peak process RSS for the complete sweep was **515,358,720 bytes (491.48 MiB)**.
One original dataset, one current shared observation variant, reusable outputs
and small reference results are retained; no full duplicate half-rounded float
dataset is built. Reference preparation uses two CPU workers. Each case retains
all N fits; the profiler exits nonzero on status, cost or reference-floor errors.

## Every group/layout/storage result

Columns are median GPU milliseconds (lower is better). Exact GPU/wall seconds,
fits/s, counter durations and diagnostics are preserved in the CSV. All 96 cells
have zero status/cost/parameter errors against their matching CPU input reference.

### Gaussian

| Threads/group | Point float32 | Fit float32 | Point half | Fit half |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 22.970 | 22.479 | 21.993 | 21.258 |
| 64 | 22.922 | 22.674 | 21.461 | 20.959 |
| 128 | 23.045 | 22.534 | 21.436 | 21.141 |
| 256 | 22.916 | 23.183 | 21.556 | 21.218 |
| 512 | 23.241 | 22.670 | 21.767 | 21.231 |
| 1024 | 23.367 | 23.108 | 21.872 | 21.431 |

### Lorentzian

| Threads/group | Point float32 | Fit float32 | Point half | Fit half |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 26.524 | 25.808 | 26.246 | 25.394 |
| 64 | 26.578 | 25.583 | 25.716 | 25.522 |
| 128 | 26.450 | 25.833 | 25.802 | 25.718 |
| 256 | 26.652 | 25.898 | 26.015 | 25.687 |
| 512 | 26.802 | 26.284 | 26.088 | 25.781 |
| 1024 | 26.908 | 26.946 | 26.662 | 26.210 |

### Pseudo-Voigt

| Threads/group | Point float32 | Fit float32 | Point half | Fit half |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 35.074 | 33.456 | 34.505 | 34.028 |
| 64 | 34.995 | 33.556 | 34.055 | 33.739 |
| 128 | 35.576 | 33.514 | 34.695 | 33.630 |
| 256 | 35.449 | 33.747 | 34.583 | 33.726 |
| 512 | 35.194 | 33.920 | 34.994 | 34.311 |
| 1024 | 34.957 | 34.404 | 35.117 | 34.353 |

### Two Lorentzians + shared linear baseline

| Threads/group | Point float32 | Fit float32 | Point half | Fit half |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 129.213 | 128.233 | 130.772 | 128.160 |
| 64 | 127.611 | 127.313 | 127.704 | 128.608 |
| 128 | 127.346 | 127.090 | 127.995 | 126.130 |
| 256 | 126.908 | 125.854 | 127.645 | 126.255 |
| 512 | 128.126 | 128.154 | 128.306 | 127.516 |
| 1024 | 127.280 | 129.321 | 126.884 | 127.397 |

For original-order point/float, group-size spread is only about **1.7–2.0%**.
At group 256, fit-major is 1.2% slower for Gaussian but 2.8%, 4.8% and 0.8%
faster for the other models. There is **no consistent measured point-major win**
for this serial-per-fit kernel on M4. This does not invalidate coalescing theory;
cache reuse, compute work and measurement order can dominate the expected benefit.
The default is unchanged, especially because a future cooperative kernel has a
different access pattern.

At group 256, half observation storage speeds up Gaussian/Lorentzian/pseudo-Voigt
by **6.3% / 2.4% / 2.5%** and slows the paired model by **0.6%**. Halving y bytes
does not halve time. Input-y bandwidth is therefore unlikely to be the dominant
limiter in these point-major cases; private-state traffic is a separate question.

## Half precision: correctness and quantization, not a free speedup

Half cases are checked against CPU fitting of **the same half-rounded values**,
not original float values. Status, final cost and parameters pass the existing
reference budgets (cost `1e-6+2e-5*|CPU|`, converged parameters
`5e-4*(1+|CPU|)`) in every cell. Float original-order results have identical
parameter hashes across all groups/layouts; half results likewise have identical
hashes across their groups/layouts, but intentionally differ from float inputs.
The stricter original-float comparison is also recorded, never hidden:

| Model | Max input rounding error | RMS input rounding error | Strict parameter rows vs original float CPU: float / half |
| --- | ---: | ---: | ---: |
| Gaussian | 0.0009765625 | 0.000166947 | 0 / 0 |
| Lorentzian | 0.0009765625 | 0.000156129 | 0 / 0 |
| PseudoVoigt | 0.0009765625 | 0.000159088 | 2 / 60 |
| TwoLorentzian | 0.0009765625 | 0.000276598 | 0 / 6 |

Strict rows use the unchanged `2e-4+2e-4*|CPU|` diagnostic at group 256. Half changes
the optimization input and sometimes the iteration path; its extra differences
from original float are quantization effects, not equivalence claims. These
M=512 experiments do not close the previously documented v1 M=2048 parity limits.

## Divergence from different stopping iterations

The CSV records actual GPU iteration counts. Define an **iteration-lane efficiency
proxy**, for contiguous execution-width groups, as
`sum(iterations) / sum(32*max_iterations_in_SIMD_group)`. This measures hypothetical
lockstep loop work, **not achieved GPU occupancy, active-lane counters, or precise
branch efficiency**: a final gradient check and rejected/accepted updates have
different costs. N=100,000 is divisible by 32, so this study has no initial SIMD
tail waste.

| Model | min / p50 / p90 / p99 / max attempts | Original proxy | Oracle-sorted proxy | Original / sorted GPU ms | Observed speedup |
| --- | --- | ---: | ---: | --- | ---: |
| Gaussian | 2 / 3 / 4 / 4 / 8 | 78.39% | 89.88% | 22.916 / 22.240 | 1.030x |
| Lorentzian | 2 / 3 / 4 / 4 / 9 | 72.84% | 95.95% | 26.652 / 24.211 | 1.101x |
| PseudoVoigt | 2 / 3 / 4 / 5 / 11 | 72.39% | 84.83% | 35.449 / 32.587 | 1.088x |
| TwoLorentzian | 3 / 4 / 4 / 5 / 10 | 78.13% | 94.52% | 126.908 / 114.227 | 1.111x |

Oracle ordering stably sorts the same fits by **already computed CPU iteration
count**, permutes inputs/seeds, dispatches and maps outputs back for validation.
It isolates a work-homogeneity opportunity. It is **not a deployable free speedup**:
learning iterations requires fitting, and sorting/packing costs are excluded.
GPU iterations can differ from CPU predictions, explaining imperfect sorted
efficiency. Oracle hashes are order-dependent; equality is checked per original
fit, not by comparing reordered hashes. The measured 3–11% gain supports divergence
as a real secondary cost, but the proxy's 22–28% idle-lane work is not a hardware
stall percentage.

## What the shader profiler and counters do—and do not—show

| Requested metric | Observation in this environment |
| --- | --- |
| Occupancy | **Unavailable**. Legal group maximum 1024 and zero static threadgroup bytes are measured limits, not resident-thread occupancy. |
| Main limiter (ALU/memory/registers) | **Not counter-confirmed**. Timing sensitivity favors arithmetic/serial LM work over input-y bandwidth; ALU vs private-register/spill traffic cannot be resolved here. |
| Per-line shader cost | **Unavailable**. No Xcode shader replay/profiler sampling was run. |
| Register count/spills | **Unavailable**. No public set exposes them on this device. |
| GPU duration | **Measured twice**: command-buffer timestamps and stage-boundary `GPUTimestamp`; maximum disagreement 41.3 ns. |
| Iteration divergence | GPU iteration distribution + explicit proxy and oracle experiment above; not a branch-counter measurement. |

Source work sites worth inspecting in Xcode (static reasoning only):
`solve` point loop computes residual/Jacobian and lower-triangular J^T J updates;
there are 6/6/10/36 such updates per point for P=3/3/4/8, plus P gradient updates.
At M=512 the paired matrix alone requires 18,432 accumulation updates per
linearization. `cost` makes additional full point passes, `cholesky` has serial
data dependencies, and private matrix/damped/Jacobian/parameter arrays increase
live state with P. **These operation counts are not per-line time percentages or
allocated-register counts.** P=8's GPU time is 5.54x Gaussian at the default
group, consistent with much more work but not proof of spilling.

## Exact commands and Xcode handoff

Native SDK path actually used (CMake/metal-cpp downloads remain blocked):

```sh
mkdir -p .agent-local/check/generated
python3 tools/embed_metal.py .agent-local/check/generated/profile_source.hpp --profile
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O3 -DNDEBUG -pthread -fobjc-arc -Wall -Wextra -Wpedantic -Werror -Iinclude -Isrc/metal -I.agent-local/check/generated -x objective-c++ src/cpu/thread_pool.cpp src/cpu/npy.cpp src/cpu/synthetic.cpp bench/metal_profile.mm -framework Foundation -framework Metal -o .agent-local/check/batchfit_metal_profile
.agent-local/check/batchfit_metal_profile > profiling/results.csv
# Small validation sweep: N=257, M=32, groups 32/256, all layouts/storage.
.agent-local/check/batchfit_metal_profile --smoke
```

With project-local dependencies available, the equivalent target is
`batchfit_metal_profile` in a Release BATCHFIT_METAL=ON build. No separate Metal
compiler is needed. Profiling kernels are included only in the `--profile`
embedded source, not ordinary production compilation.

Programmatic capture is opt-in for both the metal-cpp runtime and native drivers:

```sh
# Existing profiling/kernel-v1.gputrace must first be archived or removed AFTER inspection.
BATCHFIT_GPU_CAPTURE=1 MTL_CAPTURE_ENABLED=1 .agent-local/check/batchfit_metal_profile --single
```

The program sets process-local MTL_CAPTURE_ENABLED when capture is requested and
it was unset; an explicit conflicting value is rejected. Only the first requested
dispatch per process is captured. Existing files/external captures are never
overwritten/stopped. Stop/cleanup is RAII-safe on errors; incomplete owned traces
are removed and failures reported. Capture timings include instrumentation/file
flush and must **not** be compared to the sweep. The captured run here measured
25.899 ms command GPU time and 224.710 ms wall time, not production throughput.
Overwrite refusal was tested by hashing all existing trace files before/after;
the bytes were unchanged. Invalid environment values are also rejected.
An additional public `fit_batch(..., Device::Metal)` native-driver probe with
only BATCHFIT_GPU_CAPTURE set (MTL_CAPTURE_ENABLED initially unset) created a
502,115-byte trace successfully. That temporary probe trace was removed and the
requested 8.89 MB profiling trace restored byte-for-byte.

The requested System Trace command was attempted exactly as follows:

```sh
xcrun xctrace record --template 'Metal System Trace' --time-limit 15s --output profiling/metal-system.trace --launch -- .agent-local/check/batchfit_metal_profile --single
```

It failed with `unable to find utility "xctrace", not a developer tool or in PATH`
(exit 72). **No metal-system.trace was produced.** On an approved session with
full Xcode selected and the same project-local environment, rerun that command;
`xcrun xctrace list templates` verifies the template name. Do not change global
developer settings or bypass this sandbox to make the command work.

Manual Xcode checklist (not performed here):
`open -a Xcode profiling/kernel-v1.gputrace` opens the retained bundle on a machine
with full Xcode; this command was not run in the current harness.

1. Open `profiling/kernel-v1.gputrace` in Xcode and select the labeled
   `fit_two_lorentzian` compute command.
2. Run Xcode's shader/GPU profiling replay; collect achieved occupancy,
   register/spill and ALU/memory stall/limiter views where supported. Distinguish
   replay instrumentation from the uninstrumented timestamp sweep.
3. Inspect source-attributed costs in `src/metal/lm.metal` and shared model code,
   especially point accumulation, model eval/Jacobian, trial cost and Cholesky.
4. Inspect SIMD activity/stopping-loop divergence; compare with, but do not equate
   it to, the iteration proxy. Export actual metrics before replacing any
   unavailable entry in this report.

After inspection, remove/archive the ignored `.gputrace`/`.trace` bundles. Native
build binaries/generated headers are temporary and were removed at handoff;
the requested trace and committed results CSV remain as deliverables.

## Ranked bottlenecks and next evidence to collect

1. **Per-fit LM arithmetic and serial point passes — strongest current candidate.**
   Dense J^T J work grows with P, P=8 takes 5.54x Gaussian, and cutting input bytes
   in half yields only -0.6–6.3% speed change. Tentative main limiter is compute/
   serialized arithmetic, not confirmed ALU saturation. Obtain shader per-line
   costs to separate accumulation, model math and trial-cost passes.
2. **Stopping-iteration divergence — measured secondary opportunity.** Original
   iteration-work proxy is 72–78%; oracle grouping improves GPU time by 3–11%.
   Collect actual SIMD-active/branch data before designing a practical scheduler;
   do not count oracle sorting as a free production optimization.
3. **Private-state/register pressure or spills — plausible, unconfirmed.** Two
   P-by-P arrays plus vectors give P=8 much more logical live state, but group-size
   sensitivity is under 2% and no register/occupancy counter was available. Do not
   assert spilling or an occupancy percentage without Xcode evidence.
4. **Global observation bandwidth/layout — lower priority for v1 on this dataset.**
   Fit/point differences are small and inconsistent; half storage rarely helps
   much and introduces quantization differences. Revisit for cooperative kernels,
   larger/smaller M and other GPUs rather than changing the production default now.
