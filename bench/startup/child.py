"""Run one startup-preload scenario in this process and report the OpenMP runtimes loaded.

Usage: child.py PRELOAD_PATH|none MODULE [MODULE ...]
PRELOAD_PATH is opened with ctypes (RTLD_GLOBAL) before the modules are imported in the order given.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diag"))
from omp_probe import run_workload  # noqa: E402


def runtimes() -> list[str]:
    """Return the paths of OpenMP runtime images mapped into this process."""
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    found = []
    for i in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(i)
        if name is not None and name.rsplit(b"/", 1)[-1].startswith((b"libomp", b"libiomp", b"libgomp")):
            found.append(name.decode())
    return found


def main(argv: list[str]) -> int:
    """Preload, import, report, and run the workloads."""
    preload, modules = argv[0], argv[1:]
    if preload != "none":
        ctypes.CDLL(preload, mode=ctypes.RTLD_GLOBAL)
    for name in modules:
        importlib.import_module(name)
    images = runtimes()
    print(f"runtimes={len(images)}", flush=True)
    for image in images:
        print(f"  {image}", flush=True)
    for name in modules:
        run_workload(name)
        print(f"  workload {name} ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
