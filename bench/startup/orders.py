"""Probe import orders in fresh processes, optionally after a ctypes preload, and print a table.

Usage: orders.py PYTHON LABEL PRELOAD|none
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ORDERS = [
    ["torch", "sklearn", "lightgbm", "xgboost"],
    ["lightgbm", "xgboost", "sklearn", "torch"],
    ["torch", "lightgbm"],
    ["sklearn", "lightgbm"],
    ["torch", "xgboost"],
    ["sklearn", "xgboost"],
    ["lightgbm", "xgboost"],
    ["sklearn", "torch"],
    ["lightgbm", "torch"],
]
TIMEOUT_SECONDS = 60
CHILD = Path(__file__).with_name("child.py")


def probe(python: str, preload: str, order: list[str]) -> tuple[str, str]:
    """Return the result label and the process output for one order."""
    try:
        done = subprocess.run(
            [python, str(CHILD), preload, *order],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        return "HANG", err.stdout.decode() if isinstance(err.stdout, bytes) else (err.stdout or "")
    label = "ok" if done.returncode == 0 else f"exit {done.returncode}"
    return label, done.stdout + (done.stderr[-400:] if done.returncode else "")


def main(python: str, label: str, preload: str) -> int:
    """Probe every order and return the number of failures."""
    rows = []
    for order in ORDERS:
        result, output = probe(python, preload, order)
        runtimes = output.count("libomp") + output.count("libiomp")
        print(f"::group::{label} {'>'.join(order)}: {result}\n{output}\n::endgroup::")
        rows.append((">".join(order), result, runtimes))
    print(f"\n| {label}: order | result | libomp images |\n|---|---|---|")
    for order, result, runtimes in rows:
        print(f"| {label}: {order} | {result} | {runtimes} |")
    return sum(result != "ok" for _, result, _ in rows)


if __name__ == "__main__":
    sys.exit(min(main(*sys.argv[1:4]), 125))
