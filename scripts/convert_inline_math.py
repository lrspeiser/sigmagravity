#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import os
import re
import sys

# Usage: python scripts/convert_inline_math.py README.md
# Converts all inline LaTeX math delimiters \( ... \) to $...$ in non-code blocks.

path = sys.argv[1] if len(sys.argv) > 1 else "README.md"
with io.open(path, 'r', encoding='utf-8') as f:
    text = f.read()

out_lines = []
code = False
pattern = re.compile(r"\\\((.+?)\\\)")
for line in text.splitlines(True):  # keep line endings
    if line.strip().startswith("```"):
        code = not code
        out_lines.append(line)
        continue
    if not code:
        line = pattern.sub(lambda m: f"${m.group(1)}$", line)
    out_lines.append(line)

new_text = ''.join(out_lines)

if new_text != text:
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
