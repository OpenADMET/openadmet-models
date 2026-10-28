"""
Generate and verify the conda-lock files in devtools/locks.

Each lock file is solved from the group specs (infer.yaml, full.yaml, dev.yaml) plus one Python
fragment and one hardware fragment in devtools/conda-envs. The group specs assign the lock category
of their packages (main, full, dev), so one lock file serves every dependency group. Packages that
are installed with pip (devtools/conda-envs/pip-*.txt) are not part of the lock files.

Examples
--------
Regenerate every lock file::

    python devtools/scripts/lock_envs.py lock

Regenerate one lock file::

    python devtools/scripts/lock_envs.py lock --python 3.12 --hardware cuda

Check the specs against pyproject.toml and the lock files against the specs::

    python devtools/scripts/lock_envs.py check

Check only the specs against pyproject.toml, which needs no network and no conda-lock::

    python devtools/scripts/lock_envs.py check --specs-only

"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet

logger = logging.getLogger("lock_envs")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "devtools" / "conda-envs"
LOCK_DIR = REPO_ROOT / "devtools" / "locks"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Pinned so every contributor produces the same lock file layout
CONDA_LOCK_VERSION = "4.0.3"

PYTHON_VERSIONS = ("3.12", "3.13")

# Group spec files in order of increasing scope; a package must appear in only one of them
GROUP_SPECS = ("infer.yaml", "full.yaml", "dev.yaml")
PIP_SPECS = ("pip-infer.txt", "pip-full.txt")

# PyPI names whose conda-forge name differs, or that a hardware fragment provides
CONDA_NAMES = {"matplotlib": "matplotlib-base", "seaborn": "seaborn-base"}
HARDWARE_PROVIDED = {"torch"}

# Spec files that may contain each pyproject.toml requirement group
ALLOWED_SPECS = {
    "dependencies": ("infer.yaml",),
    "models": ("infer.yaml", "full.yaml"),
    "train": ("infer.yaml", "full.yaml"),
    "report": ("infer.yaml", "full.yaml"),
    "test": ("dev.yaml",),
}


@dataclass(frozen=True)
class Hardware:
    """
    Hardware flavor of a lock file.

    Parameters
    ----------
    platforms : tuple of str
        Conda subdirs the lock file is solved for.
    virtual_packages : str
        File name in devtools/conda-envs that declares the virtual packages (for example ``__cuda``).

    """

    platforms: tuple[str, ...]
    virtual_packages: str


HARDWARE: dict[str, Hardware] = {
    "cpu": Hardware(("osx-arm64", "linux-64"), "virtual-packages-cpu.yaml"),
    "mps": Hardware(("osx-arm64",), "virtual-packages-cpu.yaml"),
    "cuda": Hardware(("linux-64",), "virtual-packages-cuda.yaml"),
}


def lock_path(python: str, hardware: str) -> Path:
    """Return the lock file path for a Python version and hardware flavor."""
    return LOCK_DIR / f"py{python.replace('.', '')}-{hardware}-lock.yml"


def conda_lock_command() -> list[str]:
    """Return the command prefix that runs the pinned conda-lock version."""
    override = os.environ.get("CONDA_LOCK")
    if override:
        return override.split()
    if shutil.which("uvx"):
        return ["uvx", f"conda-lock@{CONDA_LOCK_VERSION}"]
    return ["conda-lock"]


def lock_arguments(python: str, hardware: str, lockfile: Path) -> list[str]:
    """Build the conda-lock ``lock`` arguments for one lock file."""
    flavor = HARDWARE[hardware]
    args = ["lock", "--lockfile", str(lockfile)]
    for name in (*GROUP_SPECS, f"python-{python}.yaml", f"hardware-{hardware}.yaml"):
        args += ["--file", str(SPEC_DIR / name)]
    for platform in flavor.platforms:
        args += ["--platform", platform]
    return args + ["--virtual-package-spec", str(SPEC_DIR / flavor.virtual_packages)]


def selected(python: str | None, hardware: str | None) -> list[tuple[str, str]]:
    """Return the (python, hardware) pairs matching the optional filters."""
    return [
        (py, hw)
        for py in PYTHON_VERSIONS
        for hw in HARDWARE
        if python in (None, py) and hardware in (None, hw)
    ]


def _normalize(name: str) -> str:
    """Return a PEP 503 normalized package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Split a requirement into its normalized name and a whitespace-free version specifier."""
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", requirement)
    if match is None:
        raise ValueError(f"Cannot parse requirement {requirement!r}")
    return _normalize(match.group(1)), re.sub(r"\s+", "", match.group(2))


def _read_spec_dependencies(path: Path) -> dict[str, str]:
    """Return the conda dependencies of a spec file as a mapping of name to specifier."""
    with path.open() as handle:
        content = yaml.safe_load(handle)
    return dict(
        _split_requirement(dep)
        for dep in content["dependencies"]
        if isinstance(dep, str)
    )


def _read_pip_requirements(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Return the requirements of a pip layer file and the packages that pull them in.

    A trailing ``# via <package>`` comment marks a requirement that pyproject.toml does not declare
    because another package in the pip layer needs it.

    Returns
    -------
    tuple of dict
        Mapping of name to the text after the name, and mapping of name to the ``via`` package.

    """
    requirements: dict[str, str] = {}
    via: dict[str, str] = {}
    for line in path.read_text().splitlines():
        text, _, comment = line.partition("#")
        text = text.strip()
        if not text:
            continue
        name, rest = _split_requirement(text)
        requirements[name] = rest
        match = re.match(r"\s*via\s+(\S+)", comment)
        if match:
            via[name] = _normalize(match.group(1))
    return requirements, via


def check_specs() -> list[str]:
    """Return disagreements between pyproject.toml, the group specs, and the pip layer files."""
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]

    groups = {"dependencies": project["dependencies"]}
    for extra, requirements in project["optional-dependencies"].items():
        if extra != "all":
            groups[extra] = requirements

    spec_dependencies = {
        name: _read_spec_dependencies(SPEC_DIR / name) for name in GROUP_SPECS
    }
    pip_requirements: dict[str, str] = {}
    pip_via: dict[str, str] = {}
    for name in PIP_SPECS:
        requirements, via = _read_pip_requirements(SPEC_DIR / name)
        pip_requirements |= requirements
        pip_via |= via

    problems: list[str] = []

    # A package in two group specs would be assigned to the later group by conda-lock
    seen: dict[str, str] = {}
    for spec_name, dependencies in spec_dependencies.items():
        for dependency in dependencies:
            if dependency in seen:
                problems.append(
                    f"{dependency} is listed in {seen[dependency]} and {spec_name}"
                )
            seen[dependency] = spec_name

    # Every pyproject.toml requirement must be a conda dependency or a pinned pip layer entry
    for group, requirements in groups.items():
        for requirement in requirements:
            name, specifier = _split_requirement(requirement)
            if name in HARDWARE_PROVIDED:
                continue

            # Git dependencies must use the same URL and commit in the pip layer
            if specifier.startswith("@"):
                if pip_requirements.get(name) != specifier:
                    problems.append(
                        f"{name}: pyproject.toml has {specifier} but the pip layer has {pip_requirements.get(name)}"
                    )
                continue

            # A pip layer pin must satisfy the specifier that pyproject.toml declares
            if name in pip_requirements:
                pinned = pip_requirements[name]
                if (
                    specifier
                    and pinned.startswith("==")
                    and pinned[2:] not in SpecifierSet(specifier)
                ):
                    problems.append(
                        f"{name}: pip layer pins {pinned[2:]} which does not satisfy {specifier}"
                    )
                continue
            conda_name = CONDA_NAMES.get(name, name)
            found = [
                spec
                for spec in ALLOWED_SPECS[group]
                if conda_name in spec_dependencies[spec]
            ]
            if not found:
                problems.append(
                    f"{name} ({group}) is missing from {' or '.join(ALLOWED_SPECS[group])}"
                )
                continue
            # A spec may tighten a constraint but must contain every clause that pyproject.toml declares
            locked_specifier = spec_dependencies[found[0]][conda_name]
            missing = (
                set(specifier.split(",")) - {""} - set(locked_specifier.split(","))
            )
            if missing:
                problems.append(
                    f"{name}: {found[0]} has {locked_specifier or 'no specifier'} "
                    f"and is missing {','.join(sorted(missing))} from pyproject.toml"
                )

    # Pip layer entries that pyproject.toml does not declare must name the pip layer package that needs them
    declared = {
        _split_requirement(requirement)[0]
        for requirements in groups.values()
        for requirement in requirements
    }
    for name in pip_requirements:
        if name in declared:
            continue
        if pip_via.get(name) not in pip_requirements:
            problems.append(
                f"{name} is in the pip layer but not declared in pyproject.toml; "
                "add a '# via <package>' comment naming a pip layer package that needs it"
            )
    return problems


def run_lock(pairs: Sequence[tuple[str, str]]) -> int:
    """Solve and write the lock files for the given pairs."""
    LOCK_DIR.mkdir(exist_ok=True)
    for python, hardware in pairs:
        lockfile = lock_path(python, hardware)
        logger.info("Locking %s", lockfile.relative_to(REPO_ROOT))

        # Remove the old file so the solve starts from the specs alone
        lockfile.unlink(missing_ok=True)
        result = subprocess.run(
            [*conda_lock_command(), *lock_arguments(python, hardware, lockfile)],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            logger.error(
                "conda-lock failed for python %s and hardware %s", python, hardware
            )
            return result.returncode
    return 0


def run_check(pairs: Sequence[tuple[str, str]], specs_only: bool) -> int:
    """Return non-zero when the specs or any lock file is out of date."""
    problems = check_specs()
    for problem in problems:
        logger.error(problem)
    failed = bool(problems)
    if specs_only:
        return 1 if failed else 0

    for python, hardware in pairs:
        lockfile = lock_path(python, hardware)
        if not lockfile.exists():
            logger.error("Missing %s", lockfile.relative_to(REPO_ROOT))
            failed = True
            continue

        # Run the input hash check on a copy: conda-lock skips the solve when the spec hashes match
        # and rewrites the copy when they do not, so a changed copy marks a stale lock file
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / lockfile.name
            shutil.copy(lockfile, copy)
            result = subprocess.run(
                [
                    *conda_lock_command(),
                    *lock_arguments(python, hardware, copy),
                    "--check-input-hash",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            up_to_date = (
                result.returncode == 0 and copy.read_bytes() == lockfile.read_bytes()
            )
        if up_to_date:
            logger.info("Up to date: %s", lockfile.relative_to(REPO_ROOT))
            continue
        logger.error(
            "Stale: %s (rerun the lock command)", lockfile.relative_to(REPO_ROOT)
        )
        failed = True
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lock or check command."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock", "check"))
    parser.add_argument(
        "--python", choices=PYTHON_VERSIONS, help="Restrict to one Python version"
    )
    parser.add_argument(
        "--hardware", choices=tuple(HARDWARE), help="Restrict to one hardware flavor"
    )
    parser.add_argument(
        "--specs-only",
        action="store_true",
        help="With check, skip the lock file freshness test",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pairs = selected(args.python, args.hardware)
    if args.command == "lock":
        return run_lock(pairs)
    return run_check(pairs, args.specs_only)


if __name__ == "__main__":
    sys.exit(main())
