"""
Install the pip layer of a dependency group into the running Python environment.

The pip layer is the set of packages that the lock files cannot provide: git dependencies, packages that
conda-forge does not build for every supported Python, and PyPI-only packages. Each file in
devtools/conda-envs/pip-*.txt is installed with ``pip install --no-deps`` because the lock files contain
every requirement of these packages. Run the script with the Python of the environment, for example
``micromamba run -n openadmet-models python devtools/scripts/install_pip_layer.py --group full``.

The script uses only the standard library.
"""

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("install_pip_layer")

SPEC_DIR = Path(__file__).resolve().parents[1] / "conda-envs"

# Dependency groups in order of increasing scope; dev uses the pip layer of full
GROUP_RANK = {"infer": 0, "full": 1, "dev": 1}


@dataclass(frozen=True)
class Step:
    """
    One pip command of the pip layer.

    Parameters
    ----------
    file : str
        Requirements file name in devtools/conda-envs.
    group : str
        Smallest dependency group that installs the file.
    python : str or None
        Python version (major.minor) the file applies to, or None for every version.
    ignore_requires_python : bool
        Whether to pass ``--ignore-requires-python`` to pip.

    """

    file: str
    group: str
    python: str | None = None
    ignore_requires_python: bool = False


# Order matters only in that each file is installed without dependencies
STEPS = (
    # The molfeat fork declares requires-python "<=3.13", which pip rejects for every 3.13.x release
    Step("pip-molfeat.txt", "infer", ignore_requires_python=True),
    Step("pip-chemprop-3.13.txt", "infer", python="3.13"),
    Step("pip-full.txt", "full"),
)


def commands(group: str, python: str) -> list[list[str]]:
    """
    Return the pip commands that install the pip layer of a group.

    Parameters
    ----------
    group : str
        Dependency group: infer, full, or dev.
    python : str
        Python version of the environment as major.minor.

    Returns
    -------
    list of list of str
        One pip argument list per requirements file.

    """
    result = []
    for step in STEPS:
        if GROUP_RANK[step.group] > GROUP_RANK[group] or step.python not in (
            None,
            python,
        ):
            continue
        command = ["install", "--no-deps"]
        if step.ignore_requires_python:
            command.append("--ignore-requires-python")
        result.append([*command, "--requirement", str(SPEC_DIR / step.file)])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Install the pip layer."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", choices=tuple(GROUP_RANK), required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    for command in commands(args.group, python):
        logger.info("+ pip %s", " ".join(command))
        subprocess.run([sys.executable, "-m", "pip", *command], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
