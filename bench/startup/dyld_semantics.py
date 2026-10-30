"""Probe how dyld resolves @rpath/libomp.dylib after a preload, using libomp copies with different install names.

Usage: dyld_semantics.py PYTHON
Prints one table row per scenario; the exit code is the number of failing scenarios (informational).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 90
CHILD = Path(__file__).with_name("child.py")


def run(cmd: list[str]) -> str:
    """Run a command and return its combined output."""
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return done.stdout + done.stderr


def make_stub(source: Path, target_dir: Path, install_name: str) -> Path:
    """Copy a libomp, rewrite its install name, and ad-hoc sign the copy."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stub = target_dir / "libomp.dylib"
    shutil.copy2(source, stub)
    stub.chmod(0o755)
    subprocess.run(["install_name_tool", "-id", install_name, str(stub)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(stub)], check=True)
    return stub


def scenario(python: str, label: str, preload: str, modules: list[str]) -> tuple[str, str]:
    """Run one scenario in a fresh process and return the label and result."""
    try:
        done = subprocess.run(
            [python, str(CHILD), preload, *modules],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        out = err.stdout.decode() if isinstance(err.stdout, bytes) else (err.stdout or "")
        print(f"::group::{label}: HANG\n{out}\n::endgroup::")
        return label, "HANG"
    result = "ok" if done.returncode == 0 else f"exit {done.returncode}"
    tail = done.stderr[-500:] if done.returncode else ""
    print(f"::group::{label}: {result}\n{done.stdout}{tail}\n::endgroup::")
    return label, result


def main(python: str) -> int:
    """Run every scenario and print the table."""
    site = Path(run([python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]).strip())
    torch_omp = site / "torch/lib/libomp.dylib"
    sklearn_omp = site / "sklearn/.dylibs/libomp.dylib"
    for lib in (torch_omp, sklearn_omp):
        print(f"{lib}:\n{run(['otool', '-D', str(lib)])}{run(['otool', '-L', str(lib)])}")

    work = Path(tempfile.mkdtemp(prefix="stubs-"))
    stubs = {
        "torch-path": torch_omp,
        "sklearn-path": sklearn_omp,
        "id-rpath": make_stub(torch_omp, work / "a", "@rpath/libomp.dylib"),
        "id-brew": make_stub(torch_omp, work / "b", "/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        "id-macports": make_stub(torch_omp, work / "c", "/opt/local/lib/libomp/libomp.dylib"),
    }
    rows = [
        scenario(python, "baseline torch>lightgbm", "none", ["torch", "lightgbm"]),
        scenario(python, "baseline torch>xgboost", "none", ["torch", "xgboost"]),
        scenario(python, "baseline lightgbm alone", "none", ["lightgbm"]),
    ]
    for name, stub in stubs.items():
        for modules in (["lightgbm"], ["xgboost"], ["lightgbm", "xgboost", "torch", "sklearn"]):
            rows.append(scenario(python, f"preload {name} then {'>'.join(modules)}", str(stub), modules))

    # Check the dyld fallback directory for @rpath libraries (needs sudo, removed afterwards)
    fallback = Path("/usr/local/lib/libomp.dylib")
    subprocess.run(["sudo", "mkdir", "-p", str(fallback.parent)], check=False)
    subprocess.run(["sudo", "cp", str(torch_omp), str(fallback)], check=False)
    for modules in (["lightgbm"], ["xgboost"]):
        rows.append(scenario(python, f"usr-local-lib fallback then {'>'.join(modules)}", "none", modules))
    subprocess.run(["sudo", "rm", "-f", str(fallback)], check=False)

    print("\n| scenario | result |\n|---|---|")
    for label, result in rows:
        print(f"| {label} | {result} |")
    return sum(result != "ok" for _, result in rows)


if __name__ == "__main__":
    sys.exit(min(main(sys.argv[1]), 125))
