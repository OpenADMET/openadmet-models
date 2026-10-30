"""Rewrite the OpenMP lookup of a lightgbm or xgboost macOS wheel so it resolves to the libomp bundled with torch.

Usage: repair_wheel.py MODE WHEEL OUTDIR
MODE is "rpath" (put the torch directory first in LC_RPATH, keep the Homebrew and MacPorts entries as fallbacks)
or "change" (rewrite the libomp load command to a path inside torch, so no search happens).
The output wheel gets build tag 1, updated RECORD hashes, and an ad-hoc signature on the edited library.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TORCH_RPATH = "@loader_path/../../torch/lib"
TORCH_OMP = f"{TORCH_RPATH}/libomp.dylib"
LIBRARIES = ("lightgbm/lib/lib_lightgbm.dylib", "xgboost/lib/libxgboost.dylib")


def run(*cmd: str) -> str:
    """Run a command and return its stdout, raising on failure."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def rpaths(lib: Path) -> list[str]:
    """Return the LC_RPATH entries of a Mach-O file in load order."""
    out = run("otool", "-l", str(lib)).splitlines()
    found = []
    for i, line in enumerate(out):
        if line.strip() == "cmd LC_RPATH":
            found.append(re.match(r"\s*path (.+) \(offset", out[i + 2]).group(1))
    return found


def omp_dependency(lib: Path) -> str:
    """Return the load command that names libomp."""
    deps = [ln.split()[0] for ln in run("otool", "-L", str(lib)).splitlines()[1:]]
    matches = [d for d in deps if d.endswith("/libomp.dylib")]
    if len(matches) != 1:
        raise SystemExit(f"Expected one libomp dependency in {lib}, found {matches}")
    return matches[0]


def edit_library(lib: Path, mode: str) -> None:
    """Edit the OpenMP lookup of one library and sign it again."""
    if mode == "rpath":
        omp = [r for r in rpaths(lib) if "libomp" in r]
        for r in omp:
            run("install_name_tool", "-delete_rpath", r, str(lib))
        run("install_name_tool", "-add_rpath", TORCH_RPATH, str(lib))
        for r in omp:
            run("install_name_tool", "-add_rpath", r, str(lib))
    elif mode == "change":
        run("install_name_tool", "-change", omp_dependency(lib), TORCH_OMP, str(lib))
    else:
        raise SystemExit(f"Unknown mode {mode}")
    run("codesign", "--force", "--sign", "-", str(lib))


def record_line(name: str, data: bytes) -> str:
    """Return a RECORD row for a file."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}\n"


def repair(wheel: Path, outdir: Path, mode: str) -> Path:
    """Write the repaired wheel and return its path."""
    parts = wheel.name[: -len(".whl")].split("-")
    if len(parts) != 5:
        raise SystemExit(f"Unexpected wheel name {wheel.name}")
    out = outdir / ("-".join([*parts[:2], "1", *parts[2:]]) + ".whl")
    outdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheel) as src, tempfile.TemporaryDirectory() as tmp:
        members = [(info, src.read(info)) for info in src.infolist() if not info.is_dir()]
        rows = []
        record_name = next(i.filename for i, _ in members if i.filename.endswith(".dist-info/RECORD"))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
            for info, data in members:
                if info.filename == record_name:
                    continue
                if info.filename in LIBRARIES:
                    path = Path(tmp) / Path(info.filename).name
                    path.write_bytes(data)
                    edit_library(path, mode)
                    data = path.read_bytes()
                elif info.filename.endswith(".dist-info/WHEEL"):
                    data = data.rstrip(b"\n") + b"\nBuild: 1\n"
                new = zipfile.ZipInfo(info.filename, info.date_time)
                new.external_attr = info.external_attr
                new.compress_type = zipfile.ZIP_DEFLATED
                dst.writestr(new, data)
                rows.append(record_line(info.filename, data))
            rows.append(f"{record_name},,\n")
            dst.writestr(record_name, "".join(rows))
    out.write_bytes(buf.getvalue())
    return out


if __name__ == "__main__":
    mode_arg, wheel_arg, outdir_arg = sys.argv[1:4]
    print(repair(Path(wheel_arg), Path(outdir_arg), mode_arg))
