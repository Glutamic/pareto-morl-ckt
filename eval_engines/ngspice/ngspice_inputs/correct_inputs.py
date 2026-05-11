#!/usr/bin/env python3
"""
Fix .include/.lib paths in .cir netlist files.

Default: convert absolute paths to relative (portable across machines).
With --to-abs: convert relative paths to absolute.

The ngspice wrapper auto-resolves relative paths at runtime, so after
running this once (default mode), no manual path fixing is needed on
any machine.
"""

import os
import re
import argparse

DIRECTIVE_PATTERN = re.compile(
    r'^(?P<dir>\.include|\.lib)\s+'
    r'"?(?P<path>[^\s"]+)"?'
    r'(?P<suffix>.*)$'
)


def process_file(file_path, mode):
    file_dir = os.path.dirname(os.path.abspath(file_path))

    with open(file_path, 'r') as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for line in lines:
        m = DIRECTIVE_PATTERN.match(line.strip())
        if m:
            p = m.group('path')
            if mode == 'to_rel':
                if os.path.isabs(p):
                    rel = os.path.relpath(p, file_dir)
                    new_p = rel
                    changed = True
                else:
                    new_p = p
            else:  # to_abs
                if not os.path.isabs(p):
                    abs_p = os.path.normpath(os.path.join(file_dir, p))
                    new_p = abs_p
                    changed = True
                else:
                    new_p = p

            if '"' in line:
                new_line = f'{m.group("dir")} "{new_p}"{m.group("suffix")}\n'
            else:
                new_line = f'{m.group("dir")} {new_p}{m.group("suffix")}\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    if changed:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Fix .include/.lib paths in .cir netlist files'
    )
    parser.add_argument(
        '--to-abs', action='store_true',
        help='Convert relative paths to absolute (default: abs→rel)'
    )
    args = parser.parse_args()

    mode = 'to_abs' if args.to_abs else 'to_rel'

    root = os.path.dirname(os.path.abspath(__file__))
    netlist_dir = os.path.join(root, 'netlist')

    for dirpath, _, files in os.walk(netlist_dir):
        for fn in sorted(files):
            if fn.endswith('.cir'):
                full = os.path.join(dirpath, fn)
                changed = process_file(full, mode)
                print(f'{"[FIXED]" if changed else "  [OK] "} {full}')


if __name__ == '__main__':
    main()
