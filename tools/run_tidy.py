"""Analyze library TUs and model instantiations using the actual compile database."""

import json
import pathlib
import shlex
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
build = (root / sys.argv[1]).resolve()
entries = json.loads((build / "compile_commands.json").read_text())
resources = {}
for filename in sorted({entry["file"] for entry in entries}):
    source = pathlib.Path(filename).resolve()
    if len(sys.argv)>2 and source!=(root/sys.argv[2]).resolve():
        continue
    if root / "src" in source.parents or source in (
        root / "tests/models_check.cpp",
        root / "tests/solver_check.cpp",
        root / "tests/initial_guess_check.cpp",
        root / "tests/batch_check.cpp",
        root / "tests/batch_failure_check.cpp",
        root / "bench/cpu_baseline.cpp",
        root / "bench/cpu_batch.cpp",
        root / "tests/metal_lm_check.cpp",
        root / "tests/cooperative_check.cpp",
        root / "tests/mmap_check.cpp",
        root / "tests/pipeline_check.cpp",
        root / "tests/numerics_check.cpp",
        root / "tests/uncertainty_check.cpp",
        root / "tests/metal_disabled_check.cpp",
        root / "bench/metal_baseline.cpp",
        root / "bench/cooperative_baseline.cpp",
        root / "bench/pipeline_baseline.cpp",
        root / "bench/numerics_study.mm",
        root / "bench/metal_batch.cpp",
        root / "bench/metal_profile.mm",
        root / "tests/synthetic_check.cpp",
        root / "tools/generate.cpp",
        root / "tools/fit_npy.cpp",
        root / "python/bindings.cpp",
    ):
        arguments = ["clang-tidy", str(source), "-p", str(build), "--quiet"]
        if sys.platform == "darwin":
            entry = next(entry for entry in entries if pathlib.Path(entry["file"]).resolve() == source)
            compiler = (entry.get("arguments") or shlex.split(entry["command"]))[0]
            # Apple's SDK wrappers require the build compiler's resource headers.
            if compiler not in resources:
                resources[compiler] = subprocess.check_output([compiler, "-print-resource-dir"], text=True).strip()
            arguments.append("--extra-arg=-resource-dir=" + resources[compiler])
        if source in (root / "src/metal/backend.cpp", root / "src/metal/runtime.cpp"):
            # metal-cpp SharedPtr intentionally sends retain/release to nil.
            arguments.append("--config={InheritParentConfig: true, CheckOptions: {clang-analyzer-core.CallAndMessage:CXXThisMethodCall: false}}")
        subprocess.run(arguments, check=True)
