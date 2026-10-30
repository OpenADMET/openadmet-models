"""Add uv settings to a pyproject.toml in a throwaway checkout.

Usage: edit_pyproject.py wheels PYPROJECT LIGHTGBM_WHEEL XGBOOST_WHEEL
       edit_pyproject.py source PYPROJECT TORCH_OMP_LIBRARY OMP_INCLUDE_DIR
       edit_pyproject.py source-brew PYPROJECT x x
"""

from __future__ import annotations

import sys
from pathlib import Path

DARWIN = "sys_platform == 'darwin'"


def insert_after(text: str, header: str, lines: str) -> str:
    """Insert lines after a table header that must occur exactly once."""
    assert text.count(f"\n{header}\n") == 1, header
    return text.replace(f"\n{header}\n", f"\n{header}\n{lines}", 1)


def main(mode: str, pyproject: str, a: str, b: str) -> None:
    """Edit the file in place."""
    path = Path(pyproject)
    text = path.read_text()
    if mode == "wheels":
        # Pin lightgbm and xgboost to repaired wheels on macOS, leave Linux on PyPI
        text = insert_after(
            text,
            "[tool.uv.sources]",
            f'lightgbm = {{ path = "{a}", marker = "{DARWIN}" }}\n'
            f'xgboost = {{ path = "{b}", marker = "{DARWIN}" }}\n',
        )
    elif mode == "source-brew":
        # Build both from source against the default libomp and add a relative rpath by linker flag
        body = '"cmake.define.CMAKE_SHARED_LINKER_FLAGS" = "-Wl,-rpath,@loader_path/../../torch/lib"'
        text = insert_after(
            text,
            "[tool.uv]",
            'no-binary-package = ["lightgbm", "xgboost"]\n'
            f"config-settings-package = {{ lightgbm = {{ {body} }}, xgboost = {{ {body} }} }}\n",
        )
    else:
        # Build both from source against the libomp inside torch and add a relative rpath
        flags = f"-Xpreprocessor -fopenmp -I{b}"
        cm = {
            "cmake.define.OpenMP_C_FLAGS": flags,
            "cmake.define.OpenMP_CXX_FLAGS": flags,
            "cmake.define.OpenMP_C_LIB_NAMES": "omp",
            "cmake.define.OpenMP_CXX_LIB_NAMES": "omp",
            "cmake.define.OpenMP_omp_LIBRARY": a,
            "cmake.define.CMAKE_SHARED_LINKER_FLAGS": "-Wl,-rpath,@loader_path/../../torch/lib",
        }
        body = ", ".join(f'"{k}" = "{v}"' for k, v in cm.items())
        text = insert_after(
            text,
            "[tool.uv]",
            'no-binary-package = ["lightgbm", "xgboost"]\n'
            f"config-settings-package = {{ lightgbm = {{ {body} }}, xgboost = {{ {body} }} }}\n",
        )
    path.write_text(text)


if __name__ == "__main__":
    main(*sys.argv[1:5])
