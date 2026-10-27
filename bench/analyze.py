"""
Summarize benchmark results as Markdown tables.

Reads one or more JSONL files written by ``bench/run.py`` and prints, for every
cell (cell identifier, Python version, group), one table per topic with one
column per candidate. The script uses only the standard library (pandas is not
needed)::

    python3 -I bench/analyze.py bench/results/results.jsonl other-host.jsonl
    python3 -I bench/analyze.py results.jsonl --python 3.12 --group full

Tables: run outcomes, phase timings (median with min-max in seconds), disk
footprint, pytest counts, reproducibility of package-set hashes, and the
time-travel resolve check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

TIMED_PHASES = (
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
MIB = 1024 * 1024

Record = dict[str, Any]
CellKey = tuple[str, str, str]


def load_records(paths: Iterable[Path]) -> list[Record]:
    """
    Read JSON lines from several files.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        JSONL files; blank lines are ignored and malformed lines are reported
        on stderr and skipped.

    Returns
    -------
    list of dict
        Parsed records in file order.

    """
    records = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                print(f"warning: {path}:{number}: {err}", file=sys.stderr)
    return records


def cell_key(record: Record) -> CellKey:
    """Return the (cell id, python, group) key of a record."""
    cell = record.get("cell", {})
    return (cell.get("id", "?"), cell.get("python", "?"), cell.get("group", "?"))


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_seconds(samples: Sequence[float]) -> str:
    """Format samples as ``median (min-max)`` in seconds."""
    if not samples:
        return "-"
    return f"{statistics.median(samples):.1f} ({min(samples):.1f}-{max(samples):.1f})"


def fmt_mib(value: float | None) -> str:
    """Format bytes as MiB."""
    return "-" if value is None else f"{value / MIB:.0f}"


def phase_seconds(records: Iterable[Record], phase: str) -> list[float]:
    """Pool the timing samples of a phase over records where it succeeded."""
    pooled: list[float] = []
    for record in records:
        entry = record.get("phases", {}).get(phase)
        if entry and entry.get("status") == "ok":
            pooled.extend(entry.get("seconds", []))
    return pooled


def group_records(
    records: Iterable[Record],
) -> dict[CellKey, dict[str, list[Record]]]:
    """Group records by cell key, then by candidate."""
    grouped: dict[CellKey, dict[str, list[Record]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[cell_key(record)][record.get("candidate", "?")].append(record)
    return grouped


def outcome_table(by_candidate: dict[str, list[Record]]) -> str:
    """Build the run outcome table (ok, failed, skipped counts and success rate)."""
    rows = []
    for candidate, runs in sorted(by_candidate.items()):
        counts = {
            s: sum(1 for r in runs if r.get("status") == s)
            for s in ("ok", "failed", "skipped")
        }
        attempted = counts["ok"] + counts["failed"]
        rate = f"{100 * counts['ok'] / attempted:.0f}%" if attempted else "-"
        failing = sorted({r.get("failed_phase") for r in runs if r.get("failed_phase")})
        rows.append(
            [
                candidate,
                str(len(runs)),
                str(counts["ok"]),
                str(counts["failed"]),
                str(counts["skipped"]),
                rate,
                ", ".join(failing) or "-",
            ]
        )
    return md_table(
        [
            "candidate",
            "runs",
            "ok",
            "failed",
            "skipped",
            "success rate",
            "failing phases",
        ],
        rows,
    )


def timing_table(by_candidate: dict[str, list[Record]]) -> str:
    """Build the phase timing table: rows are phases, columns are candidates."""
    candidates = sorted(by_candidate)
    rows = []
    for phase in TIMED_PHASES:
        cells = [fmt_seconds(phase_seconds(by_candidate[c], phase)) for c in candidates]
        if any(cell != "-" for cell in cells):
            rows.append([phase, *cells])
    return md_table(["phase (s)", *candidates], rows) if rows else "_No timed phases._"


def footprint_table(by_candidate: dict[str, list[Record]]) -> str:
    """Build the disk footprint and package count table."""
    rows = []
    for candidate, runs in sorted(by_candidate.items()):
        sizes = [
            r["env"]["size_bytes"]
            for r in runs
            if r.get("env") and r["env"].get("size_bytes")
        ]
        counts = [
            r["env"]["n_packages"]
            for r in runs
            if r.get("env") and r["env"].get("n_packages")
        ]
        caches = []
        for run in runs:
            entry = run.get("phases", {}).get("install_cold", {})
            sizes_by_dir = entry.get("cache_size_bytes") or {}
            total = sum(v for v in sizes_by_dir.values() if v)
            if total:
                caches.append(total)
        rows.append(
            [
                candidate,
                fmt_mib(statistics.median(sizes) if sizes else None),
                fmt_mib(statistics.median(caches) if caches else None),
                f"{statistics.median(counts):.0f}" if counts else "-",
            ]
        )
    return md_table(
        ["candidate", "env size (MiB)", "cache after cold (MiB)", "packages"], rows
    )


def pytest_table(by_candidate: dict[str, list[Record]]) -> str:
    """Build the pytest count table from the latest successful or failed run."""
    rows = []
    for candidate, runs in sorted(by_candidate.items()):
        for phase in ("device_smoke", "unit_tests"):
            entries = [
                r["phases"][phase]
                for r in runs
                if r.get("phases", {}).get(phase, {}).get("pytest")
            ]
            if not entries:
                continue
            counts = entries[-1]["pytest"]
            rows.append(
                [
                    candidate,
                    phase,
                    str(counts["passed"]),
                    str(counts["failed"]),
                    str(counts["errors"]),
                    str(counts["skipped"]),
                ]
            )
    return (
        md_table(["candidate", "phase", "passed", "failed", "errors", "skipped"], rows)
        if rows
        else "_No pytest results._"
    )


def hash_verdict(hashes: Sequence[str]) -> str:
    """Describe whether a list of hashes is identical."""
    if not hashes:
        return "n/a"
    distinct = len(set(hashes))
    return (
        f"identical (n={len(hashes)})"
        if distinct == 1
        else f"DIFFERENT ({distinct} hashes in {len(hashes)})"
    )


def reproducibility_table(by_candidate: dict[str, list[Record]]) -> str:
    """
    Build the reproducibility table.

    Within one host the package-set hash must be identical across repetitions;
    across hosts with the same platform and machine it must match as well.
    """
    rows = []
    for candidate, runs in sorted(by_candidate.items()):
        for field, label in (
            ("package_set_hash", "package set"),
            ("python_dist_hash", "python dists"),
        ):
            by_host: dict[str, list[str]] = defaultdict(list)
            by_platform: dict[str, dict[str, list[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for run in runs:
                digest = (run.get("env") or {}).get(field)
                if not digest:
                    continue
                host = run.get("host", {})
                host_id = host.get("hostname_hash", "?")
                platform_id = f"{host.get('system', '?')}/{host.get('machine', '?')}"
                by_host[host_id].append(digest)
                by_platform[platform_id][host_id].append(digest)
            if not by_host:
                continue
            within = "; ".join(
                f"{h}: {hash_verdict(v)}" for h, v in sorted(by_host.items())
            )
            across = []
            for platform_id, hosts in sorted(by_platform.items()):
                digests = [d for values in hosts.values() for d in values]
                across.append(
                    f"{platform_id}: {hash_verdict(digests)} over {len(hosts)} host(s)"
                )
            rows.append([candidate, label, within, "; ".join(across)])
    return (
        md_table(
            [
                "candidate",
                "hash",
                "across repetitions (per host)",
                "across hosts (per platform)",
            ],
            rows,
        )
        if rows
        else "_No package hashes._"
    )


def asof_table(by_candidate: dict[str, list[Record]]) -> str:
    """Build the time-travel resolve table."""
    rows = []
    for candidate, runs in sorted(by_candidate.items()):
        for run in runs:
            entry = run.get("phases", {}).get("resolve_asof")
            if not entry:
                continue
            first_error = "-"
            if entry.get("status") == "failed" and entry.get("tail"):
                first_error = entry["tail"][-1].strip()[:120]
            elif entry.get("status") == "skipped":
                first_error = entry.get("reason", "-")
            seconds = entry.get("median_s")
            rows.append(
                [
                    candidate,
                    str((run.get("options") or {}).get("asof", entry.get("asof", "-"))),
                    entry["status"],
                    "-" if seconds is None else f"{seconds:.1f}",
                    first_error,
                ]
            )
    return (
        md_table(["candidate", "as-of date", "status", "seconds", "detail"], rows)
        if rows
        else "_No resolve_asof results._"
    )


def render(records: Sequence[Record]) -> str:
    """
    Render all tables for the given records.

    Parameters
    ----------
    records : sequence of dict
        Result records from ``bench/run.py``.

    Returns
    -------
    str
        Markdown document.

    """
    if not records:
        return "_No records._\n"
    sections = [f"# Benchmark comparison ({len(records)} runs)"]
    for key, by_candidate in sorted(group_records(records).items()):
        cell, python, group = key
        sections.append(f"## {cell}, Python {python}, group {group}")
        installers = sorted(
            {
                f"{c}: {r['installer']['selected']}"
                for c, runs in by_candidate.items()
                for r in runs
                if (r.get("installer") or {}).get("selected")
            }
        )
        if installers:
            sections.append("Installers used: " + ", ".join(installers))
        for title, table in (
            ("Run outcomes", outcome_table),
            ("Timings", timing_table),
            ("Disk footprint", footprint_table),
            ("Pytest counts", pytest_table),
            ("Reproducibility", reproducibility_table),
            ("Time-travel resolve", asof_table),
        ):
            sections.append(f"### {title}\n\n{table(by_candidate)}")
    return "\n\n".join(sections) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program label.

    Returns
    -------
    int
        Exit code, 0 on success.

    """
    parser = argparse.ArgumentParser(
        description="Summarize bench/run.py results as Markdown."
    )
    parser.add_argument("results", nargs="+", type=Path, help="one or more JSONL files")
    parser.add_argument(
        "--candidate", action="append", help="restrict to a candidate (repeatable)"
    )
    parser.add_argument("--cell", help="restrict to a cell id such as mac-cpu")
    parser.add_argument("--python", help="restrict to a Python version")
    parser.add_argument("--group", help="restrict to a dependency group")
    args = parser.parse_args(argv)
    records = load_records(args.results)
    selected = [
        r
        for r in records
        if (not args.candidate or r.get("candidate") in args.candidate)
        and (not args.cell or cell_key(r)[0] == args.cell)
        and (not args.python or cell_key(r)[1] == args.python)
        and (not args.group or cell_key(r)[2] == args.group)
    ]
    sys.stdout.write(render(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
