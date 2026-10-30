"""Build hook that compiles the libomp shim into the wheel on macOS."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Location of torch's libomp relative to the shim: both live directly under site-packages
TORCH_OMP = "@loader_path/../torch/lib/libomp.dylib"


class ShimBuildHook(BuildHookInterface):
    """Compile a dylib named @rpath/libomp.dylib that re-exports the torch runtime."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Compile the shim and add it to the wheel as a platform-specific file."""
        if sys.platform != "darwin":
            return

        # Link a placeholder with torch's relative path as its install name, then re-export it from the shim
        out = Path(tempfile.mkdtemp(prefix="omp-shim-"))
        source = out / "empty.c"
        source.write_text("\n")
        flags = ["clang", "-dynamiclib", "-mmacosx-version-min=11.0"]
        placeholder = out / "placeholder.dylib"
        subprocess.run([*flags, "-o", str(placeholder), str(source), "-install_name", TORCH_OMP], check=True)
        shim = out / "libomp.dylib"
        subprocess.run(
            [*flags, "-o", str(shim), str(source), "-install_name", "@rpath/libomp.dylib",
             f"-Wl,-reexport_library,{placeholder}"],
            check=True,
        )
        subprocess.run(["codesign", "--force", "--sign", "-", str(shim)], check=True)

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data["force_include"][str(shim)] = "omp_shim/libomp.dylib"
