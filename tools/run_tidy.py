"""Analyze library TUs and model instantiations using the actual compile database."""

import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
build = (root / sys.argv[1]).resolve()
entries = json.loads((build / "compile_commands.json").read_text())
for filename in sorted({entry["file"] for entry in entries}):
    source = pathlib.Path(filename).resolve()
    if root / "src" in source.parents or source in (
        root / "tests/models_check.cpp",
        root / "tests/solver_check.cpp",
        root / "tests/initial_guess_check.cpp",
        root / "tests/batch_check.cpp",
        root / "tests/batch_failure_check.cpp",
        root / "bench/cpu_baseline.cpp",
        root / "bench/cpu_batch.cpp",
        root / "tests/metal_lm_check.cpp",
        root / "tests/metal_disabled_check.cpp",
        root / "bench/metal_baseline.cpp",
        root / "bench/metal_batch.cpp",
        root / "bench/metal_profile.mm",
        root / "tests/synthetic_check.cpp",
        root / "tools/generate.cpp",
    ):
        subprocess.run(["clang-tidy", str(source), "-p", str(build)], check=True)
