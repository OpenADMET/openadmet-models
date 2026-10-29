"""Import OpenMP-using libraries in a given order inside one process and report the runtimes loaded.

Usage: omp_probe.py MODULE [MODULE ...]
Modules are imported in the order given, then each runs a small multi-threaded workload.
Exit code 0 means every workload finished; a crash shows as a signal or a timeout in the parent.
"""

from __future__ import annotations

import ctypes
import importlib
import sys


def loaded_openmp_images() -> list[str]:
    """Return the paths of OpenMP runtimes mapped into this process (macOS dyld image table)."""
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = (dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count()))
    return sorted(n.decode() for n in names if n is not None and b"omp" in n.rsplit(b"/", 1)[-1])


def run_workload(name: str) -> None:
    """Run a short workload that starts OpenMP worker threads."""
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 20)).astype("float32")
    y = x[:, 0] + rng.normal(size=2000).astype("float32")
    if name == "torch":
        torch = sys.modules["torch"]
        t = torch.from_numpy(x)
        (t @ t.T).sum().item()
    elif name == "sklearn":
        from sklearn.ensemble import RandomForestRegressor

        RandomForestRegressor(n_estimators=8, n_jobs=2, random_state=0).fit(x, y)
    elif name == "lightgbm":
        lgb = sys.modules["lightgbm"]
        lgb.train({"verbose": -1}, lgb.Dataset(x, y), num_boost_round=5)
    elif name == "xgboost":
        xgb = sys.modules["xgboost"]
        xgb.train({"nthread": 2}, xgb.DMatrix(x, y), num_boost_round=5)


def main(modules: list[str]) -> int:
    """Import the modules in order, print the OpenMP images, then run each workload."""
    for name in modules:
        importlib.import_module(name)
    images = loaded_openmp_images()
    print(f"order={'>'.join(modules)} openmp_images={len(images)}", flush=True)
    for image in images:
        print(f"  {image}", flush=True)
    for name in modules:
        run_workload(name)
        print(f"  workload {name} ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
