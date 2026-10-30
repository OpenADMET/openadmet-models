"""Edit the pyproject.toml of an env/uv checkout to apply one package-choice variant.

Usage: apply_pkgs_variant.py VARIANT PYPROJECT_PATH
Variants:
  src-rpath  build LightGBM and XGBoost from the PyPI sdist on macOS with an rpath to the torch libomp
  torch210   cap torch below 2.11, the last series whose bundled libomp is shared by name with the wheels
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

LIGHTGBM_SDIST = (
    "https://files.pythonhosted.org/packages/63/8e/4db5e29290d7e619c307fdb8dab0a0514090af2ce3ec483050e024ec6126/"
    "lightgbm-4.7.0.tar.gz"
)
XGBOOST_SDIST = (
    "https://files.pythonhosted.org/packages/38/a9/295320f741c5be4be996c73ee65a2a11852028c50daa7229adb0d61c330b/"
    "xgboost-3.4.1.tar.gz"
)

SRC_RPATH_TABLES = f"""
# Build LightGBM and XGBoost from source on macOS and add an rpath to the libomp bundled with torch
[tool.uv.sources.lightgbm]
url = "{LIGHTGBM_SDIST}"
marker = "sys_platform == 'darwin'"

[tool.uv.sources.xgboost]
url = "{XGBOOST_SDIST}"
marker = "sys_platform == 'darwin'"

[tool.uv.config-settings-package.lightgbm]
"cmake.define.CMAKE_SHARED_LINKER_FLAGS" = "-Wl,-rpath,@loader_path/../../torch/lib"

[tool.uv.config-settings-package.xgboost]
"cmake.define.CMAKE_SHARED_LINKER_FLAGS" = "-Wl,-rpath,@loader_path/../../torch/lib"
"""


def main(variant: str, path: Path) -> None:
    """Rewrite the pyproject file for the variant and check that the result still parses."""
    text = path.read_text()
    if variant == "src-rpath":
        text += SRC_RPATH_TABLES
    elif variant == "torch210":
        old = '  "setuptools<82.0.0",\n  "torch",\n]'
        if old not in text:
            raise SystemExit("torch dependency line not found")
        text = text.replace(old, '  "setuptools<82.0.0",\n  "torch<2.11",\n]')
    else:
        raise SystemExit(f"unknown variant {variant}")
    tomllib.loads(text)
    path.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
