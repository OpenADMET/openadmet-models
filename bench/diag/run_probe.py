"""Run omp_probe.py for several import orders, each in a fresh process with a timeout, and print a table."""

from __future__ import annotations

import os
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
]
TIMEOUT_SECONDS = int(os.environ.get("PROBE_TIMEOUT", "120"))

# PROBE_ORDERS="torch,lightgbm;lightgbm,torch" replaces the default list of import orders
if os.environ.get("PROBE_ORDERS"):
    ORDERS = [order.split(",") for order in os.environ["PROBE_ORDERS"].split(";")]


def probe(python: str, script: Path, order: list[str]) -> tuple[str, str]:
    """Return the outcome label and the process output for one import order."""
    try:
        done = subprocess.run(
            [python, str(script), *order],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        partial = err.stdout.decode() if isinstance(err.stdout, bytes) else (err.stdout or "")
        return "HANG", partial
    label = "ok" if done.returncode == 0 else f"exit {done.returncode}"
    return label, done.stdout + done.stderr[-400:]


def main(python: str) -> int:
    """Probe every order and return the number of failing orders."""
    script = Path(__file__).with_name("omp_probe.py")
    failures = 0
    rows = []
    for order in ORDERS:
        label, output = probe(python, script, order)
        failures += label != "ok"
        print(f"::group::{'>'.join(order)}: {label}")
        print(output)
        print("::endgroup::")
        rows.append((">".join(order), label))
    print("\n| import order | result |\n|---|---|")
    for order, label in rows:
        print(f"| {order} | {label} |")
    return failures


if __name__ == "__main__":
    sys.exit(min(main(sys.argv[1]), 125))
