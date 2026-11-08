#!/usr/bin/env python3
"""!
Doxygen input filter for .dox files.

Doxygen 1.9 does not understand the raise option of the include command with the doc and raise options. This filter
therefore expands such lines itself: it inlines the referenced Markdown file, drops its
YAML front matter, and shifts all Markdown headings down by N levels. Paths are
relative to the directory Doxygen is run from (the repository root).

@param sys.argv[1]: str
  The .dox file to filter. The result is written to stdout.
"""

import re
import sys

INCLUDE = re.compile(r"^@include\{doc,raise=(\d+)\}\s+(\S+)\s*$")


def expand(path, raise_by):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if lines and lines[0].strip() == "---":
        lines = lines[lines[1:].index("---") + 2 :]
    result = []
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            m = re.match(r"^(#+)\s*(\S.*)$", line)
            if m:
                line = "#" * min(len(m.group(1)) + raise_by, 6) + " " + m.group(2)
        result.append(line)
    return "\n".join(result)


with open(sys.argv[1], encoding="utf-8") as f:
    for line in f.read().splitlines():
        m = INCLUDE.match(line)
        print(expand(m.group(2), int(m.group(1))) if m else line)
