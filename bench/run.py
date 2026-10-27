"""
Benchmark driver for environment tooling candidates.

The driver measures one cell (candidate, Python version, hardware flavor,
dependency group) on the current machine and appends one JSON line to a results
file. It uses only the standard library and runs on any Python 3.11 or newer::

    python3 -I bench/run.py --candidate baseline-conda --hardware cpu --dry-run

Each candidate tool is described by a TOML adapter in ``bench/adapters``. The
adapter lists shell-style command templates; the driver expands the placeholders
``{python}``, ``{hardware}``, ``{group}``, ``{repo}``, ``{env_dir}``,
``{work_dir}``, ``{cache_dir}``, ``{installer}``, ``{harness_python}`` and (for
``lock_asof`` only) ``{date}``. Templates are split with ``shlex`` before the
placeholders are expanded, so paths with spaces stay intact and no shell is
involved.

Safety rules enforced here:

* Environments are created only below ``--work-dir`` and the driver removes
  nothing outside that directory.
* ``--dry-run`` prints the full plan and executes nothing.
* The default cache mode is ``isolated``: caches are redirected below
  ``--work-dir`` through the adapter ``isolated_env`` table, so a cold run
  deletes only that directory. The ``global`` cache mode runs the adapter
  ``clear_cache`` command against the real user caches and needs
  ``--yes-clear-cache``.

User-facing output (plans, tables) uses ``print``; progress diagnostics use the
``logging`` module.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import string
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("bench")

SCHEMA_VERSION = 1
BENCH_DIR = Path(__file__).resolve().parent
WORKDIR_MARKER = ".openadmet-bench-workdir"
TAIL_LINES = 40

HARDWARE = ("cpu", "mps", "cuda")
PYTHON_VERSIONS = ("3.12", "3.13")
GROUPS = ("full", "infer")
CELLS = (("mac", "cpu"), ("mac", "mps"), ("linux", "cpu"), ("linux", "cuda"))
CACHE_MODES = ("isolated", "global")

# Phases in execution order; resolve_asof runs last in a separate source copy
PHASE_ORDER = (
    "lock_check",
    "lock",
    "install_cold",
    "install_warm",
    "import",
    "cli",
    "device_smoke",
    "unit_tests",
    "predict_smoke",
    "resolve_asof",
)
INSTALL_PHASES = ("install_cold", "install_warm")
ENV_PHASES = ("import", "cli", "device_smoke", "unit_tests", "predict_smoke")
DEFAULT_PHASES = {
    "full": (
        "lock_check",
        "install_cold",
        "install_warm",
        "import",
        "cli",
        "device_smoke",
        "unit_tests",
    ),
    "infer": (
        "lock_check",
        "install_cold",
        "install_warm",
        "import",
        "cli",
        "device_smoke",
        "predict_smoke",
    ),
}

BASE_PLACEHOLDERS = frozenset(
    {
        "python",
        "hardware",
        "group",
        "repo",
        "env_dir",
        "work_dir",
        "cache_dir",
        "harness_python",
    }
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "name",
        "status",
        "description",
        "installers",
        "tool_version",
        "lock",
        "lock_check",
        "lock_asof",
        "install",
        "run_prefix",
        "package_list",
        "package_list_format",
        "clear_cache",
        "cache_dirs",
        "env_location",
        "env",
        "isolated_env",
    }
)
PACKAGE_LIST_FORMATS = ("freeze", "conda-json", "conda-export")

IMPORT_SNIPPET = "import openadmet.models; import openadmet.models.inference.inference"
CLI_ARGS = ("openadmet", "--help")
DEVICE_TEST_PATH = "openadmet/models/tests/unit/test_devices.py"
UNIT_TEST_PATH = "openadmet/models/tests/unit"
ACCELERATOR_FOR_HARDWARE = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}

FACTS_SNIPPET = """
import json, sys
facts = {"python_version": sys.version.split()[0], "python_executable": sys.executable}
try:
    import torch
    mps = getattr(torch.backends, "mps", None)
    facts["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(mps is not None and mps.is_available()),
    }
except Exception as err:
    facts["torch_error"] = f"{type(err).__name__}: {err}"
print("BENCH_FACTS " + json.dumps(facts))
"""

DISTS_SNIPPET = """
import importlib.metadata as md, re
found = {}
for dist in md.distributions():
    label = dist.metadata["Name"]
    if label:
        found[re.sub(r"[-_.]+", "-", label).lower()] = dist.version
print("\\n".join(f"{k}=={v}" for k, v in sorted(found.items())))
"""

DEVICE_SNIPPET = """
import os, torch
expect = os.environ.get("OPENADMET_EXPECT_DEVICE", "cpu")
mps = getattr(torch.backends, "mps", None)
available = {
    "cpu": True,
    "mps": bool(mps is not None and mps.is_available()),
    "cuda": bool(torch.cuda.is_available()),
}
assert available[expect], f"{expect} is not available to torch"
left, right = torch.randn(8, 4), torch.randn(4, 3)
result = (left.to(expect) @ right.to(expect)).cpu()
torch.testing.assert_close(result, left @ right, rtol=1e-4, atol=1e-5)
print("device ok", expect)
"""


class BenchError(Exception):
    """Base class for harness errors reported without a traceback."""


class AdapterError(BenchError):
    """Raised when an adapter file is missing, malformed, or invalid."""


class RefusalError(BenchError):
    """Raised when the driver refuses a request for safety or host reasons."""


@dataclasses.dataclass(frozen=True)
class Step:
    """One command template and its failure policy."""

    template: str
    allow_failure: bool = False


@dataclasses.dataclass(frozen=True)
class InstallSpec:
    """Install commands for one hardware and group combination."""

    steps: tuple[Step, ...]
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Adapter:
    """Validated description of one candidate tool."""

    path: Path
    name: str
    status: str
    description: str
    installers: tuple[str, ...]
    tool_version: tuple[Step, ...]
    lock: tuple[Step, ...]
    lock_check: tuple[Step, ...]
    lock_asof: tuple[Step, ...]
    run_prefix: str
    package_list: tuple[Step, ...]
    package_list_format: str
    clear_cache: tuple[Step, ...]
    cache_dirs: tuple[str, ...]
    env_location: str
    env: Mapping[str, str]
    isolated_env: Mapping[str, str]
    install: Mapping[tuple[str, str], InstallSpec]

    @property
    def is_draft(self) -> bool:
        """Return True when the adapter is a skeleton that cannot run."""
        return self.status == "draft"

    def install_for(self, hardware: str, group: str) -> InstallSpec | None:
        """
        Resolve the install spec for a hardware and group combination.

        Parameters
        ----------
        hardware : str
            Hardware flavor, one of ``cpu``, ``mps``, ``cuda``.
        group : str
            Dependency group, one of ``full``, ``infer``.

        Returns
        -------
        InstallSpec or None
            The most specific match among exact, hardware wildcard, group
            wildcard, and full wildcard entries, or None when none exists.

        """
        for key in (
            (hardware, group),
            (hardware, "any"),
            ("any", group),
            ("any", "any"),
        ):
            if key in self.install:
                return self.install[key]
        return None


@dataclasses.dataclass(frozen=True)
class HostInfo:
    """Minimal host facts that determine which cells exist."""

    system: str
    machine: str
    has_nvidia_smi: bool

    @property
    def os_label(self) -> str:
        """Return ``mac`` or ``linux``, or the raw system label otherwise."""
        return {"Darwin": "mac", "Linux": "linux"}.get(self.system, self.system)


@dataclasses.dataclass(frozen=True)
class Cmd:
    """A fully expanded command ready to execute."""

    argv: tuple[str, ...]
    cwd: Path | None = None
    allow_failure: bool = False
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def render(self) -> str:
        """Return a shell-quoted display form, collapsing multi-line scripts."""
        shown = [
            f"<{a.count(chr(10)) + 1}-line inline script>" if "\n" in a else a
            for a in self.argv
        ]
        extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        prefix = f"{extra} " if extra else ""
        where = f"  (cwd={self.cwd})" if self.cwd else ""
        return f"{prefix}{shlex.join(shown)}{where}"


@dataclasses.dataclass
class Options:
    """Parsed command-line options for one cell run."""

    candidate: str
    adapter_path: Path
    python: str
    hardware: str
    group: str
    repo: Path
    work_dir: Path
    out_dir: Path
    results: Path
    phases: tuple[str, ...]
    reps: int = 3
    cold_reps: int = 1
    warm_reps: int = 1
    cache_mode: str = "isolated"
    yes_clear_cache: bool = False
    asof: str | None = None
    source: str = "copy"
    keep_env: bool = False
    timeout: float = 7200.0
    dry_run: bool = False


@dataclasses.dataclass
class Sample:
    """Outcome of one timed repetition."""

    exit_code: int
    seconds: float
    logs: list[Path]


# ----------------------------------------------------------------------------
# Adapter parsing and validation
# ----------------------------------------------------------------------------


def _field_names(template: str) -> list[str]:
    """Return the replacement-field identifiers used in a format template."""
    names = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            names.append(field)
    return names


def _check_template(
    template: str,
    allowed: frozenset[str],
    where: str,
    draft: bool,
    errors: list[str],
) -> None:
    """Append validation errors for a template with unknown placeholders."""
    if template.lstrip().startswith("TODO"):
        if not draft:
            errors.append(
                f"{where}: TODO placeholder is only valid when status = 'draft'"
            )
        return
    if "TODO" in template and not draft:
        errors.append(f"{where}: contains TODO but status is not 'draft'")
    try:
        fields = _field_names(template)
    except ValueError as err:
        errors.append(f"{where}: malformed braces ({err})")
        return
    for field in fields:
        if not field.isidentifier():
            errors.append(
                f"{where}: placeholder {{{field}}} must be a plain identifier"
            )
        elif field not in allowed:
            errors.append(f"{where}: unknown placeholder {{{field}}}")


def _parse_steps(
    value: Any,
    where: str,
    allowed: frozenset[str],
    draft: bool,
    errors: list[str],
) -> tuple[Step, ...]:
    """Normalize a string or list value into steps, recording errors."""
    if value is None or value == "" or value == []:
        return ()
    items = [value] if isinstance(value, (str, dict)) else value
    if not isinstance(items, list):
        errors.append(f"{where}: expected a string or a list of steps")
        return ()
    steps = []
    for index, item in enumerate(items):
        label = f"{where}[{index}]"
        if isinstance(item, str):
            template, allow_failure = item, False
        elif isinstance(item, dict) and isinstance(item.get("cmd"), str):
            extra = set(item) - {"cmd", "allow_failure"}
            if extra:
                errors.append(f"{label}: unknown keys {sorted(extra)}")
            template = item["cmd"]
            allow_failure = bool(item.get("allow_failure", False))
        else:
            errors.append(f"{label}: expected a string or a table with 'cmd'")
            continue
        if not template.strip():
            errors.append(f"{label}: empty command")
            continue
        _check_template(template, allowed, label, draft, errors)
        steps.append(Step(template, allow_failure))
    return tuple(steps)


def _parse_str_table(
    value: Any, where: str, allowed: frozenset[str], draft: bool, errors: list[str]
) -> dict[str, str]:
    """Validate a table of environment variable templates."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{where}: expected a table")
        return {}
    table = {}
    for key, template in value.items():
        if not isinstance(template, str):
            errors.append(f"{where}.{key}: expected a string")
            continue
        _check_template(template, allowed, f"{where}.{key}", draft, errors)
        table[key] = template
    return table


def validate_adapter(data: Mapping[str, Any], path: Path) -> Adapter:
    """
    Validate parsed adapter TOML and build an :class:`Adapter`.

    Parameters
    ----------
    data : mapping
        Parsed TOML document.
    path : pathlib.Path
        File the document came from; its stem must equal the ``name`` key.

    Returns
    -------
    Adapter
        The validated adapter.

    Raises
    ------
    AdapterError
        If any validation problem is found; the message lists all of them.

    """
    errors: list[str] = []
    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        errors.append(f"unknown top-level keys: {sorted(unknown)}")

    # Check identity and status
    if data.get("schema") != 1:
        errors.append("schema must be 1")
    label = data.get("name")
    if not isinstance(label, str) or label != path.stem:
        errors.append(f"name must equal the file stem '{path.stem}'")
    status = data.get("status", "ready")
    if status not in ("ready", "draft"):
        errors.append("status must be 'ready' or 'draft'")
    draft = status == "draft"

    # Determine the placeholder vocabulary
    installers = data.get("installers", [])
    if not isinstance(installers, list) or not all(
        isinstance(i, str) for i in installers
    ):
        errors.append("installers must be a list of executables")
        installers = []
    allowed = BASE_PLACEHOLDERS | ({"installer"} if installers else set())

    # Parse command groups
    tool_version = _parse_steps(
        data.get("tool_version"), "tool_version", allowed, draft, errors
    )
    lock = _parse_steps(data.get("lock"), "lock", allowed, draft, errors)
    lock_check = _parse_steps(
        data.get("lock_check"), "lock_check", allowed, draft, errors
    )
    lock_asof = _parse_steps(
        data.get("lock_asof"), "lock_asof", allowed | {"date"}, draft, errors
    )
    package_list = _parse_steps(
        data.get("package_list"), "package_list", allowed, draft, errors
    )
    clear_cache = _parse_steps(
        data.get("clear_cache"), "clear_cache", allowed, draft, errors
    )

    # Parse run prefix, environment location, and cache locations
    run_prefix = data.get("run_prefix", "")
    if not isinstance(run_prefix, str):
        errors.append("run_prefix must be a string")
        run_prefix = ""
    elif run_prefix:
        _check_template(run_prefix, allowed, "run_prefix", draft, errors)
    env_location = data.get("env_location", "{env_dir}")
    if not isinstance(env_location, str):
        errors.append("env_location must be a string")
        env_location = "{env_dir}"
    else:
        _check_template(env_location, allowed, "env_location", draft, errors)
    cache_dirs = data.get("cache_dirs", [])
    if not isinstance(cache_dirs, list) or not all(
        isinstance(c, str) for c in cache_dirs
    ):
        errors.append("cache_dirs must be a list of path templates")
        cache_dirs = []
    for index, template in enumerate(cache_dirs):
        _check_template(template, allowed, f"cache_dirs[{index}]", draft, errors)
    fmt = data.get("package_list_format", "freeze")
    if fmt not in PACKAGE_LIST_FORMATS:
        errors.append(f"package_list_format must be one of {PACKAGE_LIST_FORMATS}")
    env = _parse_str_table(data.get("env"), "env", allowed, draft, errors)
    isolated_env = _parse_str_table(
        data.get("isolated_env"), "isolated_env", allowed, draft, errors
    )

    # Parse the install sub-tables
    install: dict[tuple[str, str], InstallSpec] = {}
    install_raw = data.get("install", {})
    if not isinstance(install_raw, dict):
        errors.append("install must be a table of tables")
        install_raw = {}
    for hardware, groups in install_raw.items():
        if hardware not in (*HARDWARE, "any") or not isinstance(groups, dict):
            errors.append(
                f"install.{hardware}: hardware must be one of {(*HARDWARE, 'any')}"
            )
            continue
        for group, spec in groups.items():
            where = f"install.{hardware}.{group}"
            if group not in (*GROUPS, "any") or not isinstance(spec, dict):
                errors.append(f"{where}: group must be one of {(*GROUPS, 'any')}")
                continue
            extra = set(spec) - {"steps", "note"}
            if extra:
                errors.append(f"{where}: unknown keys {sorted(extra)}")
            steps = _parse_steps(
                spec.get("steps"), f"{where}.steps", allowed, draft, errors
            )
            if not steps:
                errors.append(f"{where}: steps must not be empty")
            install[(hardware, group)] = InstallSpec(steps, str(spec.get("note", "")))

    # Require a complete command vocabulary for ready adapters
    if not draft:
        for key, present in (
            ("tool_version", tool_version),
            ("run_prefix", run_prefix),
            ("clear_cache", clear_cache),
            ("install", install),
        ):
            if not present:
                errors.append(f"{key} is required when status = 'ready'")

    if errors:
        joined = "\n  - ".join(errors)
        raise AdapterError(f"invalid adapter {path}:\n  - {joined}")
    return Adapter(
        path=path,
        name=label,
        status=status,
        description=str(data.get("description", "")),
        installers=tuple(installers),
        tool_version=tool_version,
        lock=lock,
        lock_check=lock_check,
        lock_asof=lock_asof,
        run_prefix=run_prefix,
        package_list=package_list,
        package_list_format=fmt,
        clear_cache=clear_cache,
        cache_dirs=tuple(cache_dirs),
        env_location=env_location,
        env=env,
        isolated_env=isolated_env,
        install=install,
    )


def load_adapter(path: Path) -> Adapter:
    """
    Read and validate an adapter TOML file.

    Parameters
    ----------
    path : pathlib.Path
        Location of the adapter file.

    Returns
    -------
    Adapter
        The validated adapter.

    Raises
    ------
    AdapterError
        If the file is missing, is not valid TOML, or fails validation.

    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as err:
        raise AdapterError(f"adapter file not found: {path}") from err
    except tomllib.TOMLDecodeError as err:
        raise AdapterError(f"adapter {path} is not valid TOML: {err}") from err
    return validate_adapter(data, path)


# ----------------------------------------------------------------------------
# Placeholder expansion
# ----------------------------------------------------------------------------


def expand_argv(template: str, context: Mapping[str, str]) -> list[str]:
    """
    Split a command template with shlex and expand placeholders per token.

    Parameters
    ----------
    template : str
        Command template such as ``"{installer} create -p {env_dir}"``.
    context : mapping
        Placeholder values.

    Returns
    -------
    list of str
        Argument vector; no shell is involved.

    Raises
    ------
    AdapterError
        If the template cannot be split or uses a placeholder missing from
        the context.

    """
    if template.lstrip().startswith("TODO"):
        return [template]
    try:
        tokens = shlex.split(template)
    except ValueError as err:
        raise AdapterError(f"cannot split command '{template}': {err}") from err
    expanded = []
    for token in tokens:
        try:
            expanded.append(token.format_map(context))
        except (KeyError, IndexError, ValueError) as err:
            raise AdapterError(f"cannot expand '{template}': {err!r}") from err
    return expanded


def expand_text(template: str, context: Mapping[str, str]) -> str:
    """Expand placeholders in a plain string, then user and environment variables."""
    if template.lstrip().startswith("TODO"):
        return template
    try:
        text = template.format_map(context)
    except (KeyError, IndexError, ValueError) as err:
        raise AdapterError(f"cannot expand '{template}': {err!r}") from err
    return os.path.expandvars(os.path.expanduser(text))


# ----------------------------------------------------------------------------
# Host and cell rules
# ----------------------------------------------------------------------------


def detect_host() -> HostInfo:
    """Return the minimal host facts used for cell availability."""
    return HostInfo(
        system=platform.system(),
        machine=platform.machine(),
        has_nvidia_smi=shutil.which("nvidia-smi") is not None,
    )


def cell_refusal(host: HostInfo, hardware: str) -> str | None:
    """
    Explain why a hardware flavor does not exist on the host.

    Parameters
    ----------
    host : HostInfo
        Host facts.
    hardware : str
        Hardware flavor to test.

    Returns
    -------
    str or None
        A refusal reason, or None when the cell exists.

    """
    if hardware not in HARDWARE:
        return f"unknown hardware '{hardware}' (expected one of {HARDWARE})"
    if host.os_label not in ("mac", "linux"):
        return f"unsupported operating system '{host.system}'"
    if hardware == "cuda" and host.os_label == "mac":
        return "cuda cells do not exist on macOS"
    if hardware == "cuda" and not host.has_nvidia_smi:
        return "cuda cell needs an NVIDIA GPU but nvidia-smi is not on PATH"
    if hardware == "mps" and host.os_label != "mac":
        return "mps cells exist only on macOS"
    if hardware == "mps" and host.machine not in ("arm64", "aarch64"):
        return "mps cells need Apple Silicon"
    return None


def cell_id(host: HostInfo, hardware: str) -> str:
    """Return the cell identifier such as ``mac-cpu``."""
    return f"{host.os_label}-{hardware}"


def format_cell_matrix(host: HostInfo) -> str:
    """Render the four benchmark cells and their availability on the host."""
    lines = [
        f"Host: {host.system} {host.machine}, nvidia-smi: {'yes' if host.has_nvidia_smi else 'no'}",
        f"Python versions: {', '.join(PYTHON_VERSIONS)}; groups: {', '.join(GROUPS)}",
        "",
        f"{'cell':<12}{'hardware':<10}{'available':<11}reason",
    ]
    for cell_os, hardware in CELLS:
        label = f"{cell_os}-{hardware}"
        if cell_os != host.os_label:
            available, reason = "no", f"needs a {cell_os} host"
        else:
            refusal = cell_refusal(host, hardware)
            available, reason = ("yes", "") if refusal is None else ("no", refusal)
        lines.append(f"{label:<12}{hardware:<10}{available:<11}{reason}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Filesystem safety
# ----------------------------------------------------------------------------


def validate_work_dir(work_dir: Path, repo: Path) -> Path:
    """
    Check that a work directory is safe to create environments in.

    Parameters
    ----------
    work_dir : pathlib.Path
        Requested work directory.
    repo : pathlib.Path
        Candidate worktree.

    Returns
    -------
    pathlib.Path
        The resolved work directory.

    Raises
    ------
    RefusalError
        If the directory is a filesystem root, the home directory, overlaps
        the repository, or is an existing non-empty directory the driver did
        not create.

    """
    resolved = work_dir.expanduser().resolve()
    repo_resolved = repo.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise RefusalError(f"refusing work dir {resolved}: too broad")
    if repo_resolved.is_relative_to(resolved) or resolved.is_relative_to(repo_resolved):
        raise RefusalError(
            f"refusing work dir {resolved}: overlaps repository {repo_resolved}"
        )
    if resolved.exists():
        if not resolved.is_dir():
            raise RefusalError(f"work dir {resolved} is not a directory")
        if not (resolved / WORKDIR_MARKER).exists() and any(resolved.iterdir()):
            raise RefusalError(
                f"refusing work dir {resolved}: not empty and not created by this driver "
                f"(no {WORKDIR_MARKER} marker)"
            )
    return resolved


def ensure_work_dir(work_dir: Path) -> None:
    """Create the work directory and its marker file."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / WORKDIR_MARKER).write_text("created by bench/run.py\n")


def check_removable(path: Path, work_dir: Path) -> Path:
    """
    Verify that a path may be deleted.

    Parameters
    ----------
    path : pathlib.Path
        Candidate for deletion.
    work_dir : pathlib.Path
        Resolved work directory.

    Returns
    -------
    pathlib.Path
        The path to delete (the link itself when the path is a symlink).

    Raises
    ------
    RefusalError
        If the path is the work directory or is not strictly inside it.

    """
    root = work_dir.resolve()
    target = Path(os.path.abspath(path))
    parent = target.parent.resolve()
    if target.is_symlink():
        inside = parent.is_relative_to(root)
    else:
        resolved = target.resolve()
        inside = resolved != root and resolved.is_relative_to(root)
    if not inside:
        raise RefusalError(f"refusing to delete {path}: not inside work dir {root}")
    return target


def safe_rmtree(path: Path, work_dir: Path) -> bool:
    """
    Delete a directory tree only when it is inside the work directory.

    Parameters
    ----------
    path : pathlib.Path
        Directory to delete.
    work_dir : pathlib.Path
        Resolved work directory that must contain the path.

    Returns
    -------
    bool
        True when something was deleted.

    Raises
    ------
    RefusalError
        If the path is outside the work directory or the work directory has no
        marker file.

    """
    target = check_removable(path, work_dir)
    if not (work_dir / WORKDIR_MARKER).exists():
        raise RefusalError(
            f"refusing to delete: {work_dir} has no {WORKDIR_MARKER} marker"
        )
    if target.is_symlink():
        target.unlink()
        return True
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def dir_size_bytes(path: Path) -> int | None:
    """
    Measure disk usage of a directory tree.

    Parameters
    ----------
    path : pathlib.Path
        Directory to measure.

    Returns
    -------
    int or None
        Allocated bytes (``st_blocks`` times 512) with hardlinks counted once,
        or None when the path does not exist.

    """
    if not path.exists():
        return None
    seen: set[tuple[int, int]] = set()
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            info = current.lstat()
        except OSError:
            continue
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += getattr(info, "st_blocks", 0) * 512 or info.st_size
        if current.is_dir() and not current.is_symlink():
            try:
                stack.extend(current.iterdir())
            except OSError:
                continue
    return total


# ----------------------------------------------------------------------------
# Subprocess execution
# ----------------------------------------------------------------------------

LOG_HEADER_LINES = 2


def run_logged(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str],
    log_path: Path,
    timeout: float,
) -> tuple[int, float]:
    """
    Run a command with output captured to a log file.

    Parameters
    ----------
    argv : sequence of str
        Argument vector; no shell is used.
    cwd : pathlib.Path or None
        Working directory.
    env : mapping
        Complete environment for the child process.
    log_path : pathlib.Path
        File receiving a two-line header followed by combined stdout and stderr.
    timeout : float
        Seconds before the process group is killed and exit code 124 returned.

    Returns
    -------
    tuple of (int, float)
        Exit code (127 when the executable is missing) and wall-clock seconds
        measured with ``time.perf_counter``.

    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        handle.write(f"$ {shlex.join(argv)}\n# cwd={cwd}\n".encode())
        handle.flush()
        start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as err:
            handle.write(f"# failed to start: {err}\n".encode())
            return 127, time.perf_counter() - start
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            handle.write(f"\n# killed after {timeout} s timeout\n".encode())
            code = 124
        return code, time.perf_counter() - start


def read_output(log_path: Path) -> str:
    """Return a log file's captured output without its header lines."""
    text = log_path.read_text(errors="replace")
    return "".join(text.splitlines(keepends=True)[LOG_HEADER_LINES:])


def tail_lines(log_path: Path, count: int = TAIL_LINES) -> list[str]:
    """Return the last lines of a log file."""
    try:
        return log_path.read_text(errors="replace").splitlines()[-count:]
    except OSError:
        return []


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------

_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|errors?|xfailed|xpassed|warnings?|deselected|rerun)"
)
_SUMMARY_RE = re.compile(r"\bin\s+[\d.]+s\b")


def parse_pytest_summary(text: str) -> dict[str, Any] | None:
    """
    Parse the final pytest summary line.

    Parameters
    ----------
    text : str
        Combined pytest output produced with ``-q``.

    Returns
    -------
    dict or None
        Counts for passed, failed, skipped, errors, xfailed, xpassed, warnings
        and deselected, plus ``duration_s``; None when no summary line exists.

    """
    for line in reversed(text.splitlines()):
        if not _SUMMARY_RE.search(line):
            continue
        counts = {
            key: 0
            for key in (
                "passed",
                "failed",
                "skipped",
                "errors",
                "xfailed",
                "xpassed",
                "warnings",
                "deselected",
            )
        }
        for number, word in _COUNT_RE.findall(line):
            key = {"error": "errors", "warning": "warnings"}.get(word, word)
            if key in counts:
                counts[key] += int(number)
        match = re.search(r"in\s+([\d.]+)s", line)
        counts["duration_s"] = float(match.group(1)) if match else None
        return counts
    return None


def _normalize_package_name(label: str) -> str:
    """Normalize a distribution label per PEP 503."""
    return re.sub(r"[-_.]+", "-", label).lower()


def parse_package_list(text: str, fmt: str) -> list[str]:
    """
    Convert tool output into sorted ``name==version`` lines.

    Parameters
    ----------
    text : str
        Raw output of the adapter ``package_list`` command.
    fmt : str
        One of ``freeze``, ``conda-json``, ``conda-export``.

    Returns
    -------
    list of str
        Sorted, de-duplicated package lines.

    """
    lines: set[str] = set()
    if fmt == "conda-json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = data.get("packages", [])
        for entry in data:
            if isinstance(entry, dict) and "name" in entry and "version" in entry:
                lines.add(
                    f"{_normalize_package_name(entry['name'])}=={entry['version']}"
                )
    else:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "-e", "@EXPLICIT")):
                continue
            if fmt == "conda-export":
                parts = line.split("=")
                if len(parts) >= 2:
                    lines.add(f"{_normalize_package_name(parts[0])}=={parts[1]}")
            elif "==" in line:
                label, _, version = line.partition("==")
                lines.add(f"{_normalize_package_name(label)}=={version}")
            else:
                lines.add(line)
    return sorted(lines)


def package_set_hash(packages: Sequence[str]) -> str:
    """Return the sha256 of the sorted package lines."""
    payload = "\n".join(sorted(packages)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def summarize_seconds(samples: Sequence[float]) -> dict[str, Any]:
    """Return samples with median, min and max (None when empty)."""
    if not samples:
        return {"seconds": [], "median_s": None, "min_s": None, "max_s": None}
    return {
        "seconds": [round(s, 4) for s in samples],
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
    }


# ----------------------------------------------------------------------------
# Option handling
# ----------------------------------------------------------------------------


def resolve_phases(spec: str, group: str) -> tuple[str, ...]:
    """
    Expand a comma-separated phase list.

    Parameters
    ----------
    spec : str
        Phase labels; the word ``default`` expands to the group defaults.
    group : str
        Dependency group.

    Returns
    -------
    tuple of str
        Phases in execution order.

    Raises
    ------
    RefusalError
        If a phase is unknown or does not apply to the group.

    """
    chosen: set[str] = set()
    for item in (part.strip() for part in spec.split(",") if part.strip()):
        if item == "default":
            chosen.update(DEFAULT_PHASES[group])
        elif item in PHASE_ORDER:
            chosen.add(item)
        else:
            raise RefusalError(
                f"unknown phase '{item}' (expected one of {PHASE_ORDER})"
            )
    if "unit_tests" in chosen and group != "full":
        raise RefusalError("unit_tests applies to group 'full' only")
    return tuple(p for p in PHASE_ORDER if p in chosen)


def validate_options(opts: Options, adapter: Adapter, host: HostInfo) -> None:
    """
    Apply the refusal rules before anything runs.

    Parameters
    ----------
    opts : Options
        Parsed options.
    adapter : Adapter
        Validated adapter.
    host : HostInfo
        Host facts.

    Raises
    ------
    RefusalError
        If the combination does not exist on the host or violates a safety rule.

    """
    if opts.python not in PYTHON_VERSIONS:
        raise RefusalError(
            f"python must be one of {PYTHON_VERSIONS}, got '{opts.python}'"
        )
    if opts.group not in GROUPS:
        raise RefusalError(f"group must be one of {GROUPS}, got '{opts.group}'")
    reason = cell_refusal(host, opts.hardware)
    if reason:
        raise RefusalError(reason)
    if opts.reps < 1:
        raise RefusalError("--reps must be at least 1")
    if opts.cache_mode not in CACHE_MODES:
        raise RefusalError(f"cache mode must be one of {CACHE_MODES}")
    if adapter.is_draft and not opts.dry_run:
        raise RefusalError(
            f"adapter '{adapter.name}' has status 'draft' and cannot run"
        )

    # Check phase dependencies
    needs_env = [p for p in opts.phases if p in ENV_PHASES]
    has_install = any(p in opts.phases for p in INSTALL_PHASES)
    if needs_env and not has_install:
        raise RefusalError(
            f"phases {needs_env} need install_cold or install_warm in the same run"
        )
    if "install_cold" in opts.phases and opts.cold_reps < 1:
        raise RefusalError("install_cold requires --cold-reps >= 1")
    if "install_warm" in opts.phases and opts.warm_reps < 1:
        raise RefusalError("install_warm requires --warm-reps >= 1")
    if "resolve_asof" in opts.phases:
        if not opts.asof:
            raise RefusalError("resolve_asof requires --asof YYYY-MM-DD")
        try:
            date_type.fromisoformat(opts.asof)
        except ValueError as err:
            raise RefusalError(f"--asof must be YYYY-MM-DD, got '{opts.asof}'") from err

    # Check cache safety
    if "install_cold" in opts.phases:
        if (
            opts.cache_mode == "isolated"
            and not adapter.isolated_env
            and not adapter.is_draft
        ):
            raise RefusalError(
                "adapter defines no isolated_env; rerun with --cache-mode global"
            )
        if (
            opts.cache_mode == "global"
            and not opts.yes_clear_cache
            and not opts.dry_run
        ):
            raise RefusalError(
                "global cache mode purges your real package caches for cold runs; "
                "pass --yes-clear-cache to confirm"
            )


# ----------------------------------------------------------------------------
# Run state and command construction
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class RunState:
    """Everything resolved for one cell run."""

    opts: Options
    adapter: Adapter
    host: HostInfo
    run_id: str
    work_dir: Path
    env_dir: Path
    cache_dir: Path
    source_dir: Path
    neutral_cwd: Path
    installer: str | None
    installer_path: str | None
    context: dict[str, str]
    base_env: dict[str, str]
    log_counter: int = 0

    @property
    def cell(self) -> str:
        """Return the cell identifier such as ``mac-cpu``."""
        return cell_id(self.host, self.opts.hardware)

    @property
    def env_location(self) -> Path:
        """Return the measured environment path."""
        return Path(expand_text(self.adapter.env_location, self.context))

    def prefix(self) -> list[str]:
        """Return the argv prefix that runs a command inside the environment."""
        return expand_argv(self.adapter.run_prefix, self.context)

    def command_env(self) -> dict[str, str]:
        """Return the child environment: process, adapter env, and cache redirects."""
        env = dict(self.base_env)
        for key, template in self.adapter.env.items():
            env[key] = expand_text(template, self.context)
        if self.opts.cache_mode == "isolated":
            for key, template in self.adapter.isolated_env.items():
                env[key] = expand_text(template, self.context)
        return env


def build_state(
    opts: Options, adapter: Adapter, host: HostInfo, run_id: str
) -> RunState:
    """
    Resolve directories, installer, and placeholder context.

    Parameters
    ----------
    opts : Options
        Parsed options with a resolved work directory.
    adapter : Adapter
        Validated adapter.
    host : HostInfo
        Host facts.
    run_id : str
        Unique run identifier.

    Returns
    -------
    RunState
        The resolved state; no files are created.

    """
    tag = f"{adapter.name}-py{opts.python}-{opts.hardware}-{opts.group}-{run_id[-6:]}"
    env_dir = opts.work_dir / "envs" / tag
    cache_dir = opts.work_dir / "cache" / adapter.name
    source_dir = opts.repo if opts.source == "in-place" else opts.work_dir / "src" / tag

    # Select the installer executable
    installer = installer_path = None
    for candidate in adapter.installers:
        found = shutil.which(candidate)
        if found:
            installer, installer_path = candidate, found
            break
    if installer is None and adapter.installers:
        installer = adapter.installers[0]

    context = {
        "python": opts.python,
        "hardware": opts.hardware,
        "group": opts.group,
        "repo": str(source_dir),
        "env_dir": str(env_dir),
        "work_dir": str(opts.work_dir),
        "cache_dir": str(cache_dir),
        "harness_python": sys.executable,
    }
    if installer:
        context["installer"] = installer

    # Remove variables that would redirect interpreters outside the environment
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    base_env["PYTHONUNBUFFERED"] = "1"
    return RunState(
        opts=opts,
        adapter=adapter,
        host=host,
        run_id=run_id,
        work_dir=opts.work_dir,
        env_dir=env_dir,
        cache_dir=cache_dir,
        source_dir=source_dir,
        neutral_cwd=opts.work_dir / "neutral-cwd",
        installer=installer,
        installer_path=installer_path,
        context=context,
        base_env=base_env,
    )


def steps_to_cmds(
    steps: Sequence[Step],
    state: RunState,
    extra: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> list[Cmd]:
    """Expand adapter steps into commands run from the source directory."""
    context = {**state.context, **(extra or {})}
    return [
        Cmd(
            tuple(expand_argv(s.template, context)),
            cwd or state.source_dir,
            s.allow_failure,
        )
        for s in steps
    ]


def in_env(
    state: RunState,
    args: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> Cmd:
    """Build a command that runs inside the candidate environment."""
    return Cmd((*state.prefix(), *args), cwd, False, dict(env or {}))


def phase_commands(state: RunState, phase: str) -> list[Cmd]:
    """
    Return the expanded commands of a phase.

    Parameters
    ----------
    state : RunState
        Resolved run state.
    phase : str
        Phase label from :data:`PHASE_ORDER`.

    Returns
    -------
    list of Cmd
        Commands in execution order; empty when the adapter provides none.

    """
    opts, adapter = state.opts, state.adapter
    if phase == "lock_check":
        return steps_to_cmds(adapter.lock_check, state)
    if phase == "lock":
        return steps_to_cmds(adapter.lock, state)
    if phase == "resolve_asof":
        return steps_to_cmds(
            adapter.lock_asof, state, {"date": opts.asof or "YYYY-MM-DD"}
        )
    if phase in INSTALL_PHASES:
        spec = adapter.install_for(opts.hardware, opts.group)
        return steps_to_cmds(spec.steps, state) if spec else []
    if phase == "import":
        return [in_env(state, ["python", "-c", IMPORT_SNIPPET], state.neutral_cwd)]
    if phase == "cli":
        return [in_env(state, list(CLI_ARGS), state.neutral_cwd)]
    expect = {"OPENADMET_EXPECT_DEVICE": opts.hardware}
    if phase == "device_smoke":
        if opts.group == "full":
            args = [
                "python",
                "-m",
                "pytest",
                DEVICE_TEST_PATH,
                "-q",
                "-p",
                "no:cacheprovider",
            ]
            return [in_env(state, args, state.source_dir, expect)]
        return [
            in_env(state, ["python", "-c", DEVICE_SNIPPET], state.neutral_cwd, expect)
        ]
    if phase == "unit_tests":
        args = [
            "python",
            "-m",
            "pytest",
            UNIT_TEST_PATH,
            "-n",
            "auto",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        return [in_env(state, args, state.source_dir)]
    if phase == "predict_smoke":
        return predict_commands(state)
    raise BenchError(f"no commands defined for phase '{phase}'")


def predict_commands(state: RunState) -> list[Cmd]:
    """
    Return the three predict_smoke commands: build fixture, predict, check.

    The fixture script lives in the harness checkout, so it is available even
    when the candidate worktree does not contain ``bench/``.
    """
    script = str(BENCH_DIR / "fixtures" / "predict_fixture.py")
    root = state.neutral_cwd / "predict"
    model_dir, input_csv, output_csv = (
        root / "model",
        root / "input.csv",
        root / "predictions.csv",
    )
    accelerator = ACCELERATOR_FOR_HARDWARE[state.opts.hardware]
    return [
        in_env(state, ["python", script, "build", str(root)], state.neutral_cwd),
        in_env(
            state,
            [
                "openadmet",
                "predict",
                "--input-path",
                str(input_csv),
                "--input-col",
                "MY_SMILES",
                "--model-dir",
                str(model_dir),
                "--output-csv",
                str(output_csv),
                "--accelerator",
                accelerator,
            ],
            state.neutral_cwd,
        ),
        in_env(state, ["python", script, "check", str(output_csv)], state.neutral_cwd),
    ]


def destructive_actions(state: RunState, phase: str) -> list[str]:
    """
    List the deletions and cache purges a phase performs before it runs.

    Raises
    ------
    RefusalError
        If a deletion target is outside the work directory.

    """
    actions = []
    if phase == "install_cold":
        if state.opts.cache_mode == "isolated":
            check_removable(state.cache_dir, state.work_dir)
            actions.append(f"remove isolated cache dir {state.cache_dir}")
        else:
            for cmd in steps_to_cmds(state.adapter.clear_cache, state):
                actions.append(
                    f"clear_cache (GLOBAL, needs --yes-clear-cache): {cmd.render()}"
                )
    if phase in INSTALL_PHASES:
        check_removable(state.env_dir, state.work_dir)
        actions.append(f"remove env dir {state.env_dir}")
    return actions


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------


def format_plan(state: RunState) -> str:
    """
    Render the complete plan for a cell without executing anything.

    Parameters
    ----------
    state : RunState
        Resolved run state.

    Returns
    -------
    str
        Multi-line plan text.

    """
    opts, adapter = state.opts, state.adapter
    lines = [
        "DRY RUN: nothing is executed, created, or deleted",
        f"candidate : {adapter.name} (status {adapter.status}) from {adapter.path}",
        f"cell      : {state.cell}, python {opts.python}, hardware {opts.hardware}, group {opts.group}",
        f"host      : {state.host.system} {state.host.machine}",
        f"installer : {state.installer or 'n/a'}"
        + (
            f" ({state.installer_path})"
            if state.installer_path
            else " (not found on PATH)"
            if adapter.installers
            else ""
        ),
        f"cache mode: {opts.cache_mode}",
        f"repo      : {opts.repo} (source {opts.source}: {state.source_dir})",
        f"work dir  : {state.work_dir}",
        f"env dir   : {state.env_dir}",
        f"cache dir : {state.cache_dir}",
        f"log dir   : {opts.out_dir}",
        f"results   : {opts.results}",
        f"phases    : {', '.join(opts.phases)}",
        "",
        "tool_version commands:",
    ]
    lines += [f"    {c.render()}" for c in steps_to_cmds(adapter.tool_version, state)]
    reps_for = {
        "install_cold": opts.cold_reps,
        "install_warm": opts.warm_reps,
        "import": opts.reps,
        "cli": opts.reps,
    }
    for phase in opts.phases:
        lines.append("")
        lines.append(f"[{phase}] x{reps_for.get(phase, 1)}")
        if (
            phase in INSTALL_PHASES
            and adapter.install_for(opts.hardware, opts.group) is None
        ):
            lines.append("    SKIPPED: adapter has no install entry for this cell")
            continue
        for action in destructive_actions(state, phase):
            lines.append(f"    DESTRUCTIVE: {action}")
        cmds = phase_commands(state, phase)
        if not cmds:
            lines.append("    SKIPPED: adapter defines no command for this phase")
        lines += [f"    run: {c.render()}" for c in cmds]
    lines += ["", "after installs: measure env size, cache size, packages, torch facts"]
    if not opts.keep_env:
        lines.append(f"cleanup: remove env dir {state.env_dir}")
        if opts.source == "copy":
            lines.append(f"cleanup: remove source copy {state.source_dir}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------------


def _probe(argv: Sequence[str], timeout: float = 10.0) -> str | None:
    """Run a short read-only probe and return stripped stdout."""
    try:
        done = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() or None


def probe_host() -> dict[str, Any]:
    """
    Collect host facts with read-only probes.

    Returns
    -------
    dict
        Platform, machine, harness Python, CPU count and model, RAM, operating
        system version, and GPU model and driver when ``nvidia-smi`` exists.

    """
    system = platform.system()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": system,
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "harness_python": platform.python_version(),
        "harness_python_executable": sys.executable,
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:10],
    }
    if system == "Darwin":
        memsize = _probe(["sysctl", "-n", "hw.memsize"])
        info["ram_bytes"] = int(memsize) if memsize and memsize.isdigit() else None
        info["cpu_model"] = _probe(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["os_version"] = f"macOS {platform.mac_ver()[0]}"
    elif system == "Linux":
        info["ram_bytes"] = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_bytes"] = int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
        info["cpu_model"] = None
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
        try:
            info["os_version"] = platform.freedesktop_os_release().get("PRETTY_NAME")
        except OSError:
            info["os_version"] = None
    smi = (
        _probe(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        if shutil.which("nvidia-smi")
        else None
    )
    info["gpus"] = (
        [
            dict(zip(("name", "driver"), (p.strip() for p in row.split(","))))
            for row in smi.splitlines()
        ]
        if smi
        else []
    )
    return info


def git_info(path: Path) -> dict[str, Any]:
    """Return HEAD sha, branch, and dirty flag for a checkout (None values on failure)."""
    sha = _probe(["git", "-C", str(path), "rev-parse", "HEAD"])
    branch = _probe(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"])
    status = _probe(["git", "-C", str(path), "status", "--porcelain"])
    return {"sha": sha, "branch": branch, "dirty": bool(status) if sha else None}


def copy_source(repo: Path, dest: Path) -> None:
    """
    Copy tracked and untracked non-ignored files of a worktree.

    Parameters
    ----------
    repo : pathlib.Path
        Candidate worktree.
    dest : pathlib.Path
        Destination directory, created if missing.

    """
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        check=False,
    )
    dest.mkdir(parents=True, exist_ok=True)
    if listing.returncode != 0:
        shutil.copytree(
            repo,
            dest,
            ignore=shutil.ignore_patterns(".git"),
            dirs_exist_ok=True,
            symlinks=True,
        )
        return
    for rel in filter(None, listing.stdout.decode().split("\0")):
        source = repo / rel
        if source.is_dir() and not source.is_symlink():
            continue
        if not (source.exists() or source.is_symlink()):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------


def run_cmds(state: RunState, phase: str, rep: int, cmds: Sequence[Cmd]) -> Sample:
    """
    Run commands in sequence and time them together.

    Parameters
    ----------
    state : RunState
        Resolved run state.
    phase : str
        Phase label used as the log file prefix.
    rep : int
        Repetition index.
    cmds : sequence of Cmd
        Commands to run; the sequence stops at the first failure that is not
        marked ``allow_failure``.

    Returns
    -------
    Sample
        Exit code of the failing command (0 when all passed), summed seconds,
        and the log files.

    """
    logs: list[Path] = []
    total = 0.0
    exit_code = 0
    env = state.command_env()
    for index, cmd in enumerate(cmds):
        log = state.opts.out_dir / "logs" / f"{phase}-r{rep}-s{index}.log"
        logs.append(log)
        code, seconds = run_logged(
            cmd.argv,
            cwd=cmd.cwd,
            env={**env, **cmd.env},
            log_path=log,
            timeout=state.opts.timeout,
        )
        total += seconds
        if code != 0 and not cmd.allow_failure:
            exit_code = code
            break
    return Sample(exit_code, total, logs)


def phase_record(
    state: RunState, samples: Sequence[Sample], **extra: Any
) -> dict[str, Any]:
    """Build the result entry of a phase from its samples."""
    failed = [s for s in samples if s.exit_code != 0]
    record: dict[str, Any] = {
        "status": "failed" if failed else "ok",
        "exit_codes": [s.exit_code for s in samples],
        **summarize_seconds([s.seconds for s in samples]),
        "logs": [
            str(p.relative_to(state.opts.out_dir)) for s in samples for p in s.logs
        ],
    }
    if failed and failed[-1].logs:
        record["tail"] = tail_lines(failed[-1].logs[-1])
    record.update(extra)
    return record


def skipped_record(reason: str) -> dict[str, Any]:
    """Build the result entry of a skipped phase."""
    return {"status": "skipped", "reason": reason}


class CellRun:
    """Executes the phases of one cell and assembles the result record."""

    def __init__(self, state: RunState) -> None:
        """
        Bind the run state.

        Parameters
        ----------
        state : RunState
            Resolved run state for the cell.

        """
        self.state = state
        self.opts = state.opts
        self.phases: dict[str, dict[str, Any]] = {}
        self.record: dict[str, Any] = {}
        self.env_ready = False

    def execute(self) -> dict[str, Any]:
        """
        Run every requested phase and return the result record.

        Returns
        -------
        dict
            JSON-serializable record; see ``bench/schema.md``.

        """
        state, opts = self.state, self.opts
        started = time.perf_counter()
        self.prepare()
        try:
            for phase in opts.phases:
                if phase == "resolve_asof":
                    continue
                self.run_phase(phase)
            if "resolve_asof" in opts.phases:
                self.run_resolve_asof()
        finally:
            self.cleanup()
        self.record["phases"] = self.phases
        self.record["wall_seconds_total"] = round(time.perf_counter() - started, 3)
        self.finalize_status()
        return self.record

    # Preparation ------------------------------------------------------------

    def prepare(self) -> None:
        """Create directories, copy the source, and record identity facts."""
        state, opts = self.state, self.opts
        ensure_work_dir(state.work_dir)
        opts.out_dir.mkdir(parents=True, exist_ok=True)
        state.neutral_cwd.mkdir(parents=True, exist_ok=True)
        if opts.source == "copy":
            copy_source(opts.repo, state.source_dir)
        self.record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": state.run_id,
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "candidate": state.adapter.name,
                "adapter_status": state.adapter.status,
                "cell": {
                    "id": state.cell,
                    "os": state.host.os_label,
                    "hardware": opts.hardware,
                    "python": opts.python,
                    "group": opts.group,
                },
                "installer": {
                    "candidates": list(state.adapter.installers),
                    "selected": state.installer,
                    "path": state.installer_path,
                },
                "options": {
                    "phases": list(opts.phases),
                    "reps": opts.reps,
                    "cold_reps": opts.cold_reps,
                    "warm_reps": opts.warm_reps,
                    "cache_mode": opts.cache_mode,
                    "source": opts.source,
                    "asof": opts.asof,
                },
                "host": probe_host(),
                "env": None,
                "torch": None,
                "git": {
                    "candidate": git_info(opts.repo),
                    "harness": git_info(BENCH_DIR),
                },
                "paths": {
                    "env_dir": str(state.env_dir),
                    "cache_dir": str(state.cache_dir),
                    "log_dir": str(opts.out_dir),
                },
            }
        )
        self.record["tool_versions"] = self.collect_tool_versions()

    def collect_tool_versions(self) -> dict[str, str]:
        """Run the adapter tool_version commands and record their first lines."""
        versions = {}
        for index, cmd in enumerate(
            steps_to_cmds(self.state.adapter.tool_version, self.state)
        ):
            sample = run_cmds(
                self.state,
                "tool_version",
                index,
                [dataclasses.replace(cmd, allow_failure=True)],
            )
            text = read_output(sample.logs[0]).strip() if sample.logs else ""
            versions[shlex.join(cmd.argv)] = "\n".join(text.splitlines()[:5])
        return versions

    def cleanup(self) -> None:
        """Remove the environment and source copy unless --keep-env is given."""
        state, opts = self.state, self.opts
        if opts.keep_env:
            return
        safe_rmtree(state.env_dir, state.work_dir)
        if opts.source == "copy":
            safe_rmtree(state.source_dir, state.work_dir)

    # Phase dispatch -----------------------------------------------------------

    def run_phase(self, phase: str) -> None:
        """Run one phase and store its result entry."""
        state, opts = self.state, self.opts
        if phase in ENV_PHASES and not self.env_ready:
            self.phases[phase] = skipped_record("environment is not installed")
            return
        if phase in INSTALL_PHASES:
            self.run_install(phase)
            return
        cmds = phase_commands(state, phase)
        if not cmds:
            self.phases[phase] = skipped_record(
                "adapter defines no command for this phase"
            )
            return
        if phase in ("import", "cli"):
            samples = [run_cmds(state, phase, rep, cmds) for rep in range(opts.reps)]
            self.phases[phase] = phase_record(state, samples)
        elif phase == "predict_smoke":
            self.run_predict(cmds)
        else:
            sample = run_cmds(state, phase, 0, cmds)
            extra: dict[str, Any] = {}
            if phase in ("device_smoke", "unit_tests"):
                text = read_output(sample.logs[-1]) if sample.logs else ""
                extra["mode"] = "pytest" if "pytest" in cmds[0].argv else "inline"
                extra["pytest"] = (
                    parse_pytest_summary(text) if extra["mode"] == "pytest" else None
                )
            self.phases[phase] = phase_record(state, [sample], **extra)

    def run_install(self, phase: str) -> None:
        """Run cold or warm install repetitions and measure the resulting environment."""
        state, opts = self.state, self.opts
        spec = state.adapter.install_for(opts.hardware, opts.group)
        if spec is None:
            self.phases[phase] = skipped_record(
                "adapter has no install entry for this cell"
            )
            return
        reps = opts.cold_reps if phase == "install_cold" else opts.warm_reps
        samples: list[Sample] = []
        cmds = phase_commands(state, phase)
        for rep in range(reps):
            failed_prep = self.prepare_install(phase)
            if failed_prep is not None:
                samples.append(failed_prep)
                break
            sample = run_cmds(state, phase, rep, cmds)
            samples.append(sample)
            if sample.exit_code != 0:
                break
        extra: dict[str, Any] = {"note": spec.note} if spec.note else {}
        extra["cache_size_bytes"] = self.cache_size_bytes()
        self.phases[phase] = phase_record(state, samples, **extra)
        self.env_ready = bool(samples) and samples[-1].exit_code == 0
        # Measure once, after the final install of the run
        last_install = phase == "install_warm" or "install_warm" not in opts.phases
        if self.env_ready and last_install:
            self.measure_environment()

    def prepare_install(self, phase: str) -> Sample | None:
        """
        Clear caches (cold only) and remove the previous environment.

        Returns
        -------
        Sample or None
            A failed sample when the adapter clear_cache command fails, else None.

        """
        state, opts = self.state, self.opts
        if phase == "install_cold":
            if opts.cache_mode == "isolated":
                logger.info("removing isolated cache %s", state.cache_dir)
                safe_rmtree(state.cache_dir, state.work_dir)
            else:
                logger.warning("running adapter clear_cache against global caches")
                cleared = run_cmds(
                    state,
                    "clear_cache",
                    0,
                    steps_to_cmds(state.adapter.clear_cache, state),
                )
                if cleared.exit_code != 0:
                    return cleared
        logger.info("removing env dir %s", state.env_dir)
        safe_rmtree(state.env_dir, state.work_dir)
        return None

    # Environment facts --------------------------------------------------------

    def cache_size_bytes(self) -> dict[str, int | None]:
        """Measure the cache directories for the active cache mode."""
        state = self.state
        if self.opts.cache_mode == "isolated":
            paths = [state.cache_dir]
        else:
            paths = [
                Path(expand_text(t, state.context)) for t in state.adapter.cache_dirs
            ]
        return {str(p): dir_size_bytes(p) for p in paths}

    def measure_environment(self) -> None:
        """Record env size, package list and hash, python and torch facts."""
        state, opts = self.state, self.opts
        location = state.env_location
        if location != state.env_dir:
            check_removable(location, state.work_dir)
        env_info: dict[str, Any] = {
            "location": str(location),
            "size_bytes": dir_size_bytes(location),
        }

        # Record python and torch facts from inside the environment
        facts_cmd = in_env(state, ["python", "-c", FACTS_SNIPPET], state.neutral_cwd)
        sample = run_cmds(state, "facts", 0, [facts_cmd])
        facts = {}
        if sample.exit_code == 0:
            for line in read_output(sample.logs[0]).splitlines():
                if line.startswith("BENCH_FACTS "):
                    try:
                        facts = json.loads(line[len("BENCH_FACTS ") :])
                    except json.JSONDecodeError:
                        facts = {"parse_error": line}
        self.record["env_facts"] = facts
        self.record["torch"] = facts.get("torch")

        # Record python distributions (comparable across tools)
        dists_cmd = in_env(state, ["python", "-c", DISTS_SNIPPET], state.neutral_cwd)
        sample = run_cmds(state, "dists", 0, [dists_cmd])
        dists = (
            parse_package_list(read_output(sample.logs[0]), "freeze")
            if sample.exit_code == 0
            else []
        )
        env_info["python_dist_count"] = len(dists)
        env_info["python_dist_hash"] = package_set_hash(dists) if dists else None
        sidecar = opts.out_dir / "python-dists.txt"
        sidecar.write_text("\n".join(dists) + "\n")

        # Record the full package list from the adapter, else the distributions
        packages = dists
        source = "importlib.metadata"
        if state.adapter.package_list:
            cmds = steps_to_cmds(state.adapter.package_list, state)
            sample = run_cmds(state, "package_list", 0, cmds)
            if sample.exit_code == 0:
                packages = parse_package_list(
                    read_output(sample.logs[-1]), state.adapter.package_list_format
                )
                source = "adapter"
        env_info["package_list_source"] = source
        env_info["n_packages"] = len(packages)
        env_info["package_set_hash"] = package_set_hash(packages) if packages else None
        packages_file = opts.out_dir / "packages.txt"
        packages_file.write_text("\n".join(packages) + "\n")
        env_info["package_list_file"] = packages_file.name
        env_info["python_dist_file"] = sidecar.name
        self.record["env"] = env_info

    # Specialized phases -------------------------------------------------------

    def run_predict(self, cmds: Sequence[Cmd]) -> None:
        """Run fixture build, predict, and output check; time the predict step."""
        state = self.state
        build = run_cmds(state, "predict_smoke_build", 0, cmds[:1])
        if build.exit_code != 0:
            self.phases["predict_smoke"] = phase_record(state, [build], stage="build")
            return
        predict = run_cmds(state, "predict_smoke", 0, cmds[1:2])
        if predict.exit_code != 0:
            self.phases["predict_smoke"] = phase_record(
                state,
                [predict],
                stage="predict",
                fixture_seconds=round(build.seconds, 4),
            )
            return
        check = run_cmds(state, "predict_smoke_check", 0, cmds[2:])
        sample = Sample(check.exit_code, predict.seconds, [*predict.logs, *check.logs])
        self.phases["predict_smoke"] = phase_record(
            state, [sample], stage="check", fixture_seconds=round(build.seconds, 4)
        )

    def run_resolve_asof(self) -> None:
        """Re-lock to a past date in a fresh source copy."""
        state, opts = self.state, self.opts
        if not state.adapter.lock_asof:
            self.phases["resolve_asof"] = skipped_record(
                "adapter defines no lock_asof command"
            )
            return
        asof_dir = state.work_dir / "src" / f"{state.source_dir.name}-asof"
        scratch = dataclasses.replace(
            state, source_dir=asof_dir, context={**state.context, "repo": str(asof_dir)}
        )
        copy_source(opts.repo, asof_dir)
        try:
            cmds = phase_commands(scratch, "resolve_asof")
            sample = run_cmds(scratch, "resolve_asof", 0, cmds)
            self.phases["resolve_asof"] = phase_record(state, [sample], asof=opts.asof)
        finally:
            safe_rmtree(asof_dir, state.work_dir)

    def finalize_status(self) -> None:
        """Record the overall status, failing phase, and failure tail."""
        statuses = {name: entry["status"] for name, entry in self.phases.items()}
        failed = [name for name in self.opts.phases if statuses.get(name) == "failed"]
        if failed:
            self.record["status"] = "failed"
            self.record["failed_phase"] = failed[0]
            self.record["failure_tail"] = self.phases[failed[0]].get("tail", [])
        elif statuses and all(s == "skipped" for s in statuses.values()):
            self.record["status"] = "skipped"
            self.record["failed_phase"] = None
            self.record["failure_tail"] = []
        else:
            self.record["status"] = "ok"
            self.record["failed_phase"] = None
            self.record["failure_tail"] = []


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a single line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=False) + "\n")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Benchmark one environment-tooling cell and append a JSON line of results.",
    )
    parser.add_argument("--candidate", help="adapter label, e.g. baseline-conda")
    parser.add_argument(
        "--adapter",
        type=Path,
        help="adapter TOML path (default: bench/adapters/<candidate>.toml)",
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version of the environment (default 3.12)",
    )
    parser.add_argument("--hardware", choices=HARDWARE, help="hardware flavor")
    parser.add_argument(
        "--group",
        choices=GROUPS,
        default="full",
        help="dependency group (default full)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="candidate worktree (default: the harness worktree root)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="parent of all environments and caches (default: system temp)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="per-run logs and sidecars (default: bench/results/runs/<run_id>)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="JSONL file to append (default: bench/results/results.jsonl)",
    )
    parser.add_argument(
        "--phases",
        default="default",
        help="comma list of phases; 'default' expands per group",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="repetitions of the import and cli phases (default 3)",
    )
    parser.add_argument(
        "--cold-reps", type=int, default=1, help="cold install repetitions (default 1)"
    )
    parser.add_argument(
        "--warm-reps", type=int, default=1, help="warm install repetitions (default 1)"
    )
    parser.add_argument(
        "--cache-mode",
        choices=CACHE_MODES,
        default="isolated",
        help="isolated caches under the work dir (default) or the real global caches",
    )
    parser.add_argument(
        "--yes-clear-cache",
        action="store_true",
        help="allow the adapter clear_cache command to purge global caches",
    )
    parser.add_argument("--asof", help="date for the resolve_asof phase, YYYY-MM-DD")
    parser.add_argument(
        "--source",
        choices=("copy", "in-place"),
        default="copy",
        help="run from a copy of the worktree (default) or in place",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="leave the environment and source copy on disk",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200.0,
        help="seconds allowed per command (default 7200)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and execute nothing"
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="print the cell matrix for this host and exit",
    )
    parser.add_argument(
        "--check-adapters",
        action="store_true",
        help="validate every adapter file and exit",
    )
    return parser


def default_work_dir() -> Path:
    """Return the default work directory below the system temp directory."""
    return Path(tempfile.gettempdir()) / "openadmet-bench"


def check_adapters(adapter_dir: Path) -> int:
    """Validate every adapter in a directory and print one line each."""
    problems = 0
    for path in sorted(adapter_dir.glob("*.toml")):
        try:
            adapter = load_adapter(path)
            print(f"ok       {path.name} (status {adapter.status})")
        except AdapterError as err:
            problems += 1
            print(f"INVALID  {err}")
    return 2 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program label; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 for a successful run or plan, 1 when the cell ran but failed, 2 for
        a refusal or usage error.

    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    host = detect_host()
    try:
        if args.list_cells:
            print(format_cell_matrix(host))
            return 0
        if args.check_adapters:
            return check_adapters(BENCH_DIR / "adapters")
        return run_from_args(args, host)
    except BenchError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def run_from_args(args: argparse.Namespace, host: HostInfo) -> int:
    """Validate arguments, then print the plan or run the cell."""
    if not args.candidate and not args.adapter:
        raise RefusalError("--candidate (or --adapter) is required")
    if not args.hardware:
        raise RefusalError("--hardware is required")
    adapter_path = args.adapter or BENCH_DIR / "adapters" / f"{args.candidate}.toml"
    adapter = load_adapter(adapter_path)
    repo = (args.repo or BENCH_DIR.parent).resolve()
    work_dir = validate_work_dir(args.work_dir or default_work_dir(), repo)
    run_id = (
        datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    opts = Options(
        candidate=adapter.name,
        adapter_path=adapter_path,
        python=args.python,
        hardware=args.hardware,
        group=args.group,
        repo=repo,
        work_dir=work_dir,
        out_dir=(args.out_dir or BENCH_DIR / "results" / "runs" / run_id).resolve(),
        results=(args.results or BENCH_DIR / "results" / "results.jsonl").resolve(),
        phases=resolve_phases(args.phases, args.group),
        reps=args.reps,
        cold_reps=args.cold_reps,
        warm_reps=args.warm_reps,
        cache_mode=args.cache_mode,
        yes_clear_cache=args.yes_clear_cache,
        asof=args.asof,
        source=args.source,
        keep_env=args.keep_env,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    validate_options(opts, adapter, host)
    state = build_state(opts, adapter, host, run_id)
    if opts.dry_run:
        print(format_plan(state))
        return 0
    if adapter.installers and state.installer_path is None:
        raise RefusalError(
            f"none of the installers {list(adapter.installers)} is on PATH"
        )

    # Skip the cell when the adapter does not support it
    if (
        any(p in INSTALL_PHASES for p in opts.phases)
        and adapter.install_for(opts.hardware, opts.group) is None
    ):
        record = skipped_cell_record(
            state, "adapter has no install entry for this cell"
        )
        append_jsonl(opts.results, record)
        print(f"skipped: {record['reason']}")
        return 0

    record = CellRun(state).execute()
    append_jsonl(opts.results, record)
    print(
        f"{record['status']}: {state.cell} py{opts.python} {opts.group} -> {opts.results}"
    )
    return 0 if record["status"] != "failed" else 1


def skipped_cell_record(state: RunState, reason: str) -> dict[str, Any]:
    """Build a minimal record for a cell the adapter does not support."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": state.run_id,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "candidate": state.adapter.name,
        "adapter_status": state.adapter.status,
        "cell": {
            "id": state.cell,
            "os": state.host.os_label,
            "hardware": state.opts.hardware,
            "python": state.opts.python,
            "group": state.opts.group,
        },
        "host": probe_host(),
        "git": {"candidate": git_info(state.opts.repo), "harness": git_info(BENCH_DIR)},
        "status": "skipped",
        "reason": reason,
        "failed_phase": None,
        "failure_tail": [],
        "phases": {},
    }


if __name__ == "__main__":
    sys.exit(main())
