"""Add a torch-relative rpath to the shared library inside a LightGBM or XGBoost macOS wheel.

Usage: repair_wheel.py WHEEL OUTPUT_DIR
The repaired wheel keeps the original file name and version; only the library and RECORD change.
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

RPATH = "@loader_path/../../torch/lib"
LIBRARIES = {"lightgbm/lib/lib_lightgbm.dylib", "xgboost/lib/libxgboost.dylib"}


def record_row(path: str, data: bytes) -> str:
    """Return the RECORD row for a file."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def rpaths(lib: Path) -> list[str]:
    """Return the LC_RPATH entries of a library in load order."""
    lines = subprocess.run(["otool", "-l", str(lib)], capture_output=True, text=True, check=True).stdout.splitlines()
    return [lines[i + 2].split()[1] for i, line in enumerate(lines) if line.strip() == "cmd LC_RPATH"]


def patch_library(data: bytes) -> bytes:
    """Return the library with the first libomp search path replaced by the torch directory and re-signed."""
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "lib.dylib"
        lib.write_bytes(data)
        has_new = False
        for rpath in rpaths(lib):
            if rpath == RPATH:
                has_new = True
            elif "libomp" in rpath and not has_new:
                subprocess.run(["install_name_tool", "-rpath", rpath, RPATH, str(lib)], check=True)
                has_new = True
            elif "libomp" in rpath:
                subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(lib)], check=True)
        if not has_new:
            subprocess.run(["install_name_tool", "-add_rpath", RPATH, str(lib)], check=True)
        print(f"rpaths after repair: {rpaths(lib)}")
        subprocess.run(["codesign", "--force", "--sign", "-", str(lib)], check=True)
        return lib.read_bytes()


def main(wheel: str, out_dir: str) -> int:
    """Write the repaired wheel into the output directory."""
    source = Path(wheel)
    target = Path(out_dir) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    patched: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
        record_name = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
        for info in zin.infolist():
            if info.filename == record_name:
                continue
            data = zin.read(info)
            if info.filename in LIBRARIES:
                data = patch_library(data)
            patched[info.filename] = data
            zout.writestr(info, data)
        rows = [record_row(name, data) for name, data in patched.items()]
        rows.append(f"{record_name},,")
        zout.writestr(record_name, "\n".join(rows) + "\n")
    if not any(name in patched for name in LIBRARIES):
        target.unlink()
        print(f"No library to repair in {source.name}", file=sys.stderr)
        return 1
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
