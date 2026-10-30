"""Open the libomp shim at interpreter start so dyld binds @rpath/libomp.dylib to the torch runtime."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def preload() -> None:
    """Open the shim library globally when it and torch's libomp are both installed."""
    if sys.platform != "darwin":
        return
    shim = Path(__file__).with_name("libomp.dylib")
    if not shim.is_file() or not (shim.parent.parent / "torch/lib/libomp.dylib").is_file():
        return
    ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)


preload()
