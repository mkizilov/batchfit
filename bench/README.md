# Unified benchmark

```sh
export PATH="$PWD/.venv/bin:$PATH"
uv pip install -e '.[dev,benchmark]'
python bench/run.py --build --workers 10 --output bench/results/rerun
```

Use the detected core count for an explicit all-core study; ordinary smoke uses
two workers. Native code is built Release-only in `build/benchmark-suite`; the
benchmark adapter is not installed into the public Python package. Existing
result directories containing `raw.csv` are never overwritten. No server is used.

The default Lorentzian sweep covers N=1,000, 10,000, 100,000, 1 million and
10 million, M=32,128,512,2048, plus small-N crossover probes. All four models are
selectable with `--models`. Backends are the SciPy `curve_fit` Python loop,
single-worker and multithreaded C++ CPU, forced Metal v1, forced v2 SIMD/fit,
forced v2 threadgroup/fit, and automatic dispatch. Native fitting uses float32;
SciPy fits the same stored observations/initials in float64, with the same box
bounds. SciPy uses its bounded TRF algorithm, batchfit uses bounded analytic LM.

## Dataset and bounds

Each M/model uses 1,024 deterministic noisy patterns and data-derived guesses
from the existing CPU benchmark workload. Large N cycles these patterns, but
**every requested fit is actually executed**; no fitting or timing is replaced
by cached results. Cached CPU pattern references only bound correctness-check
memory. This repeated-pattern workload is not 10 million unique random spectra;
its cache/divergence behavior must not be generalized without another study.

At most `--chunk-fits` rows (default 16,384) are resident. Results are consumed and
checked per chunk; no full-N output/dataset is retained. All output rows are
validated against the CPU reference. Failures are retained in `raw.csv` and make
the script fail, never disappear from counts or graphs. Uncertainties are off.

## Timing definitions

- `dataset_setup_seconds`: generation, feature guesses, CPU pattern references
  and export of the small input pattern set; recorded separately, not extrapolated
  into an N-dependent cost.
- `setup_seconds`: backend creation (including the first cold GPU compile) and
  one untimed warm batch. The primary study measures warmed repeated calls.
- `allocation_seconds`: bounded input/shared-buffer allocation, included in wall.
- `prepare_seconds`: packing repeated input rows/initials; Metal directly packs
  point-major observations, SciPy converts its bounded inputs to float64. This
  is **not** a claim to measure synthetic generation/feature extraction for N
  unique fits: supplied p0 is already available to every fitting backend.
- `fit_wall_seconds`: synchronous fitting API, including CPU result allocation
  or GPU encoding/submission/waiting.
- `gpu_seconds`: summed command-buffer GPU timestamps, not a CPU stopwatch.
- `readback_seconds`: Metal output copies into CPU-owned result vectors.
- `wall_seconds`: allocation plus measured elapsed streaming phases, excluding
  validation and setup/warmup. It includes loop/destruction overhead; it is not
  merely device kernel time. `validation_seconds` is retained separately.

Three repetitions are used below N=1 million, one long repetition at larger N;
backend order rotates. Summary rows select an **actual median-wall repetition**
so phase timings stay additive, and retain repetition count and min/max wall.

## SciPy limits and plots

SciPy's default per-cell budget is 30 seconds; `--scipy-budget-seconds 0` requests
the complete, potentially multi-hour Python-loop matrix. The soft budget is
checked between individual fits. A limited row reports `state=time_limit`, its
requested N, actual completed count, measured time and measured prefix rate.
It is **never** presented as a complete-N measurement. Prefix markers use actual
completed N on the plots; native full-N and complete SciPy points use requested N.

`raw.csv` retains every repetition. `summary.csv`, scientific `metadata.json`,
and PNG/SVG figures are written under the requested output directory. Throughput
plots have log-log axes; phase figures show warm automatic-Metal wall fractions.
`--plot-only` regenerates figures from a saved summary without running fits.

```sh
python tools/check_benchmarks.py   # bounded all-model smoke, tails/parity/plots
python bench/run.py --plot-only --output bench/results
```

BENCHMARKS.md records the executed range, crossover and limitations. Plotting is
headless (`Agg`) and Matplotlib's cache stays under the project-local cache.
