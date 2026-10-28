"""
Create a conda environment for openadmet-models from a committed lock file.

The script installs the lock categories of one dependency group with micromamba, then installs the
group's pip layer (git dependencies and PyPI-only packages) without dependencies and runs
``pip check``. See docs/installation.rst for the dependency groups and hardware flavors.

Examples
--------
Inference-only environment on a CPU::

    python devtools/scripts/install_env.py --python 3.12 --hardware cpu --group infer --name openadmet-infer

Development environment with the package installed in editable mode on an NVIDIA GPU host::

    python devtools/scripts/install_env.py --python 3.12 --hardware cuda --group dev --package editable

"""

import argparse
import logging
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("install_env")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "devtools" / "conda-envs"
LOCK_DIR = REPO_ROOT / "devtools" / "locks"

# Lock categories and pip layer files installed for each dependency group
GROUP_CATEGORIES = {
    "infer": ("main",),
    "full": ("main", "full"),
    "dev": ("main", "full", "dev"),
}
GROUP_PIP_FILES = {
    "infer": ("pip-infer.txt",),
    "full": ("pip-infer.txt", "pip-full.txt"),
    "dev": ("pip-infer.txt", "pip-full.txt"),
}

# Operating system and machine accepted by each hardware flavor
HARDWARE_HOSTS = {
    "cpu": (("Darwin", "arm64"), ("Linux", "x86_64")),
    "mps": (("Darwin", "arm64"),),
    "cuda": (("Linux", "x86_64"),),
}


def find_installer(requested: str | None) -> str:
    """
    Return a micromamba-compatible installer executable.

    Parameters
    ----------
    requested : str or None
        Executable name or path requested by the user, or None to search for micromamba and mamba.

    Returns
    -------
    str
        Executable that can install conda-lock lock files.

    Raises
    ------
    RuntimeError
        If no executable is found or the executable cannot read lock files (mamba before 2.0).

    """
    candidates = [requested] if requested else ["micromamba", "mamba"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path is None:
            continue
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True, check=False
        ).stdout
        match = re.search(r"(\d+)\.\d+", version)
        if match is None or int(match.group(1)) < 2:
            raise RuntimeError(
                f"{path} reports version {version.strip()!r}; lock files need micromamba or mamba 2 or newer"
            )
        return path
    raise RuntimeError(
        "No micromamba or mamba 2 executable found; install micromamba from https://mamba.readthedocs.io"
    )


def run(command: Sequence[str]) -> None:
    """Run a command and raise when it fails."""
    logger.info("+ %s", " ".join(command))
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Create the environment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", choices=("3.12", "3.13"), default="3.12")
    parser.add_argument("--hardware", choices=tuple(HARDWARE_HOSTS), default="cpu")
    parser.add_argument("--group", choices=tuple(GROUP_CATEGORIES), default="full")
    parser.add_argument("--name", default="openadmet-models", help="Environment name")
    parser.add_argument(
        "--installer",
        help="micromamba or mamba 2 executable (default: search the PATH)",
    )
    parser.add_argument(
        "--package",
        choices=("none", "regular", "editable"),
        default="none",
        help="Also install this repository into the environment without dependencies",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Reject hardware flavors the current machine cannot use
    host = (platform.system(), platform.machine())
    if host not in HARDWARE_HOSTS[args.hardware]:
        logger.error("Hardware flavor %s does not support %s %s", args.hardware, *host)
        return 2

    lockfile = LOCK_DIR / f"py{args.python.replace('.', '')}-{args.hardware}-lock.yml"
    if not lockfile.exists():
        logger.error("Missing %s", lockfile)
        return 2

    try:
        installer = find_installer(args.installer)
    except RuntimeError as error:
        logger.error("%s", error)
        return 2

    # Install the conda packages of the selected lock categories
    command = [
        installer,
        "create",
        "--yes",
        "--name",
        args.name,
        "--file",
        str(lockfile),
    ]
    for category in GROUP_CATEGORIES[args.group]:
        command += ["--category", category]
    run(command)

    # Install the pip layer without dependencies, then verify that conda provided every requirement
    pip = [installer, "run", "--name", args.name, "python", "-m", "pip"]
    install = [*pip, "install", "--no-deps"]
    for name in GROUP_PIP_FILES[args.group]:
        install += ["--requirement", str(SPEC_DIR / name)]
    run(install)
    if args.package != "none":
        editable = ["--editable"] if args.package == "editable" else []
        run([*pip, "install", "--no-deps", *editable, str(REPO_ROOT)])
    run([*pip, "check"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
