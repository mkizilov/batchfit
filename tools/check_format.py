"""Check only maintained C++ and Metal sources; never traverse downloaded code."""

import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
files = sorted(
    p
    for folder in ("include", "src", "tests", "bench")
    for p in (root / folder).rglob("*")
    if p.suffix in (".cpp", ".hpp", ".metal")
)
subprocess.run(["clang-format", "--dry-run", "--Werror", *map(str, files)], check=True)
