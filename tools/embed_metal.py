"""Embed production MSL and shared headers, preserving diagnostics with #line."""

import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
seen = set()


def expand(path):
    if path in seen:
        return ""
    seen.add(path)
    text = f'#line 1 "{path.relative_to(root).as_posix()}"\n'
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line == "#pragma once":
            text += "\n"
        elif line.startswith("#include <batchfit/"):
            include = line[len("#include <"):-1]
            text += expand(root / "include" / include)
            text += f'#line {number + 1} "{path.relative_to(root).as_posix()}"\n'
        else:
            text += line + "\n"
    return text


source = expand(root / "src/metal/lm.metal")
assert ')batchfit_msl"' not in source
output = pathlib.Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('#pragma once\nnamespace batchfit::metal::detail {\n'
                  'inline constexpr const char* shader_source = R"batchfit_msl(' + source +
                  ')batchfit_msl";\n}\n')
