# Hyperspectral demo

`hyperspectral_demo.ipynb` contains an executed, self-contained 512×512×129
float32 Lorentzian example. It generates smooth center/HWHM/amplitude truth,
Gaussian noise and twelve nonfinite/dead pixels; fits every row with data-only
initial guesses; and displays fitted center, width, amplitude, SSE, categorical
status, defect, ground-truth and signed-error maps. Failed pixels are retained
in status counts and masked only in parameter/error displays.

## Launch from the project environment

From the repository root, preserve the supplied project-local HOME/TMPDIR/cache
settings and use the existing `.venv` (Python 3.9–3.12 dependency pins):

```sh
export PATH="$PWD/.venv/bin:$PATH"
uv pip install -e '.[demo]'
jupyter notebook --no-browser --ip=127.0.0.1 --port=22215 \
  --ServerApp.port_retries=0 --ServerApp.root_dir="$PWD" \
  --KernelManager.transport=ipc \
  --KernelManager.ip="$PWD/.agent-local/tmp/jupyter-demo-kernel"
```

Open the authenticated URL printed by Jupyter, select
`examples/hyperspectral_demo.ipynb`, use the project Python kernel, then
**Restart Kernel → Run All**. The notebook checks that its interpreter is in
this repository's `.venv`. No global kernelspec or user configuration is needed.
The HTTP server binds only loopback port 22215; kernel channels use project-local
Unix sockets rather than unrelated TCP ports. Shut down from Jupyter, Ctrl-C,
or `jupyter server stop 22215`. Do not leave the test server running.

`DEVICE='auto'` uses Metal for the synthetic float32 cube when available; choose
`'cpu'` or `'metal'` explicitly in the parameters cell. Every pixel is submitted,
including defects. Fitting proceeds in 8,192-pixel blocks, retaining only output
maps in addition to the 129 MiB cube; no second full clean cube is allocated.
The width is Lorentzian **HWHM**, not Gaussian sigma or FWHM. Uncertainties are
off because this example needs parameter/error maps, not full covariance storage.

## Real input format

Set `REAL_DATA_DIR = ROOT / 'data/my_scan'` to a directory containing:

- `cube.npy`: C-contiguous native float32/float64 `(H,W,M)` intensities, spectral
  samples last; all pixels share the same x-axis.
- `x.npy`: native float32/float64 `(M,)`, finite and strictly increasing, with
  calibrated physical units. At least four samples are required; a resolved,
  adequately sampled peak is needed for meaningful fitting/automatic guesses.

The real-data cell uses `np.load(...,allow_pickle=False,mmap_mode='r')` for the
cube and streams pixel blocks through the same fitting function. Pixel failures
are not silently dropped. Output maps retain spatial order `(row,column)`.
No physical experimental dataset is bundled or claimed as verified here. The
loader was actually tested with a small read-only NPY cube, an invalid pixel,
uneven chunk tails, and shape/dtype/order/axis errors.

This model assumes one positive emission Lorentzian **without a baseline**;
real spectra may need a different model or explicit calibration/preprocessing.
Do not interpret a converged status as proof of model suitability. float64 with
auto stays on CPU; explicit Metal converts to float32. See NUMERICS.md before
converting large-offset/narrow-range axes, and inspect residuals/status maps.

## Verification

```sh
python tools/check_demo.py                 # full synthetic + real-NPY branch
python tools/check_demo.py --device cpu    # same full image on CPU
python tools/check_demo.py --write         # store actual default outputs
sh scripts/check.sh
```

The checker uses fresh Jupyter kernels over temporary project-local IPC sockets,
fails on cell errors, verifies the full 512×512 workload and embedded map images,
tests the real-data branch, then shuts down kernels and removes connection/fixture
files. Ordinary checks do not rewrite the committed notebook. The dev extra
contains headless execution dependencies; the demo extra also installs the UI.

Recorded Metal execution: **262,132/262,132 healthy converged**, **12/12 defects
flagged**, center RMSE **0.00256871**, HWHM RMSE **0.00365907**, amplitude RMSE
**0.00790029**. The CPU rerun also recovered every healthy pixel and flagged every
defect with equivalent errors. The stored 1.839 s Metal fit includes cold context/
shader startup, guesses and packing; the 0.433 s CPU rerun is not a controlled
warm/cold performance comparison. BENCHMARKS.md is the performance study.

Verified on the supplied CPython 3.9.6/macOS arm64 environment. Jupyter UI startup
and its authenticated loopback route were checked and the server stopped; browser
click-through was not tested. This interpreter's LibreSSL 2.8.3 emits urllib3's
OpenSSL compatibility warning on server startup. Local HTTP and IPC execution
work; external TLS/plugin downloads were not verified or worked around with host
settings or an older SSL dependency.

The final `scripts/check.sh` passes: 25 CPU / 35 Metal CTests, 73 Python tests,
the 56-case benchmark smoke, formatting/tidy, lock consistency and this full
notebook/real-data execution check. Python document diagnostics are clean.
