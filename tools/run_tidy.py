"""Run clang-tidy serially on our library TUs using the actual compile database."""

import json
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
build = (root / sys.argv[1]).resolve()
entries = json.loads((build / "compile_commands.json").read_text())
for filename in sorted({entry["file"] for entry in entries}):
    source = pathlib.Path(filename).resolve()
    if root / "src" in source.parents:
        subprocess.run(["clang-tidy", str(source), "-p", str(build)], check=True)
