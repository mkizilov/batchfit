"""Validate reproducible numerical-study coverage; never gate intentionally raw/fast errors."""

import csv
import math
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]


def validate(path, smoke):
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == (192 if smoke else 7680), len(rows)
    assert {r["model"] for r in rows} == {"Gaussian", "Lorentzian", "PseudoVoigt", "TwoLorentzian"}
    keys = [(r["model"], r["regime"], r["snr"], r["points"], r["normalized"], r["flags"], r["fast"], r["kernel"]) for r in rows]
    assert len(set(keys)) == len(keys)
    for r in rows:
        for name in ("guess_failures", "reference_failures", "quantization_reference_failures", "nonfinite_rows", "status_mismatches"):
            assert int(r[name]) == 0, (name, r)
        assert int(r["converged"]) == int(r["fits"]) == int(r["accuracy_rows"]), r
        assert float(r["gpu_seconds"]) > 0, r
        assert math.isfinite(float(r["max_unit_error"])), r
        if r["normalized"] == "1" and r["flags"] == "3" and r["fast"] == "0":
            assert float(r["max_unit_error"]) <= 2e-4, r
    print(f"Numerics CSV validated: {len(rows)} configurations, all rows retained, safe normalized recipe <=2e-4 maximum unit error")


if sys.argv[1] == "--csv":
    validate((root / sys.argv[2]).resolve(), False)
else:
    binary = (root / sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="numerics-", dir=root / ".agent-local") as directory:
        output = pathlib.Path(directory) / "results.csv"
        probe = pathlib.Path(directory) / "probe.csv"
        subprocess.run([str(binary), "--smoke", "--output", str(output), "--probe", str(probe)], check=True, cwd=root)
        validate(output, True)
        probes = list(csv.DictReader(probe.open()))
        assert len(probes) == 10
        for row in probes:
            assert int(row["bit_finite"]) == [1, 0, 0, 0, 1][int(row["input"])], row
        print("Native safe/fast nonfinite probe validated; builtin results are recorded, not assumed")
