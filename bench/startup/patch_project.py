"""Edit a copy of the env/uv pyproject.toml to install the OpenMP fix through uv.

Usage: patch_project.py VARIANT PROJECT_DIR
VARIANT is shim (local package with a .pth preload) or wheels (repaired LightGBM and XGBoost wheels).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(variant: str, project: str) -> int:
    """Rewrite pyproject.toml in the project directory."""
    path = Path(project) / "pyproject.toml"
    text = path.read_text()
    header = "[tool.uv.sources]\n"
    if variant == "shim":
        text = text.replace('  "torch",\n]\n', '  "torch",\n  "omp-shim ; sys_platform == \'darwin\'",\n]\n', 1)
        text = text.replace(header, header + 'omp-shim = { path = "devtools/omp-shim" }\n', 1)
    elif variant in {"shim-group", "shim-prebuilt"}:
        # Keep the shim out of the published project metadata by using a dependency group installed by default
        text = text.replace("dev = [\n", 'macos = [\n  "omp-shim ; sys_platform == \'darwin\'",\n]\ndev = [\n', 1)
        text = text.replace("[tool.uv]\n", '[tool.uv]\ndefault-groups = ["dev", "macos"]\n', 1)
        text = text.replace(header, header + 'omp-shim = { path = "devtools/omp-shim" }\n', 1)
    elif variant == "wheels":
        lines = []
        for wheel in sorted((Path(project) / "devtools/omp-wheels").glob("*.whl")):
            name = wheel.name.split("-")[0]
            lines.append(f'{name} = {{ path = "devtools/omp-wheels/{wheel.name}", marker = "sys_platform == \'darwin\'" }}\n')
        text = text.replace(header, header + "".join(lines), 1)
    else:
        print(f"unknown variant {variant}", file=sys.stderr)
        return 2
    path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
