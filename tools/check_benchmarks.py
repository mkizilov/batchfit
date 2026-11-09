"""Bounded benchmark smoke: real timings, parity, tails, censoring and figures."""
import csv
import importlib.util
import math
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="benchmark-check-", dir=root / ".agent-local") as temporary:
    output = pathlib.Path(temporary) / "results"
    subprocess.run([sys.executable, str(root / "bench/run.py"), "--build", "--smoke", "--output", str(output)], check=True, cwd=root)
    with (output / "raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert b"\r\n" not in (output/"raw.csv").read_bytes()
    assert len(rows) == 56
    for row in rows:
        assert row["state"] == "complete" and row["completed"] == row["N"]
        assert all(int(row[key]) == 0 for key in ("status_errors", "parameter_errors", "cost_errors"))
        wall, fit, gpu = (float(row[key]) for key in ("wall_seconds", "fit_wall_seconds", "gpu_seconds"))
        assert math.isfinite(wall) and wall > 0 and fit > 0
        assert sum(float(row[key]) for key in ("allocation_seconds", "prepare_seconds", "fit_wall_seconds", "readback_seconds")) <= wall + 1e-6
        assert (gpu > 0) == row["backend"].startswith("metal_")
    for model in ("gaussian", "lorentzian", "pseudo_voigt", "two_lorentzian"):
        for suffix in ("throughput", "phases"):
            assert (output / f"{model}_{suffix}.png").read_bytes().startswith(b"\x89PNG")
            assert "<svg" in (output / f"{model}_{suffix}.svg").read_text()
            assert all(line==line.rstrip() for line in (output/f"{model}_{suffix}.svg").read_text().splitlines())
    spec = importlib.util.spec_from_file_location("bench_run", root / "bench/run.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    native = bench.load_native(root / "build/benchmark-suite")
    data = native.inputs("lorentzian", 32, 32)
    limited = bench.scipy_run("lorentzian", data, 1000000, 31, 1e-9)
    assert 0 < limited["completed"] < 1000000
    data["parameters"][0, 0] += 10
    failed = native.run("lorentzian", data, 17, "cpu_single", 1, 31)
    assert failed["parameter_errors"] > 0
    try:
        native.run("lorentzian", data, 17, "missing", 1, 31)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown native backend was accepted")
print("Benchmark smoke passed: 56 complete cases, bounded tails, parity, real GPU/wall phases, figures and explicit SciPy censoring")
