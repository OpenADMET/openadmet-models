# Environment tooling benchmark harness

This directory contains the harness for the environment-tooling comparison in OpenADMET/openadmet-models#615. It measures one cell at a time on the current machine and appends one JSON line per run to a results file. The harness lives on the `env/harness` branch and does not merge to `main`.

The driver and the analysis script use only the Python standard library (3.11 or newer) and run with any system Python:

```bash
python3 -I bench/run.py --list-cells
python3 -I bench/run.py --candidate baseline-conda --hardware cpu --group full --dry-run
python3 -I bench/analyze.py bench/results/results.jsonl
```

## Cells

A cell is a candidate, a Python version (`3.12` or `3.13`), a hardware flavor, and a dependency group.

| Cell | Hardware flavor | Exists on |
|---|---|---|
| `mac-cpu` | `cpu` | macOS |
| `mac-mps` | `mps` (Metal Performance Shaders, Apple Silicon only) | macOS on arm64 |
| `linux-cpu` | `cpu` | Linux |
| `linux-cuda` | `cuda` (NVIDIA GPU) | Linux with `nvidia-smi` on `PATH` |

The `full` group contains everything including test dependencies. The `infer` group contains only what is needed to load a trained model and predict. The driver refuses combinations that do not exist on the host, such as `cuda` on a Mac. `--list-cells` prints the matrix for the current machine.

## Running a cell

```bash
python3 -I bench/run.py \
  --candidate baseline-conda --python 3.12 --hardware cpu --group full \
  --repo /path/to/candidate/worktree \
  --work-dir /big/disk/openadmet-bench \
  --results bench/results/results.jsonl
```

Always run with `--dry-run` first. The dry run prints the host, every path, every command with placeholders expanded, and every destructive action, and it executes nothing.

| Option | Meaning |
|---|---|
| `--candidate`, `--adapter` | Adapter label (`bench/adapters/<label>.toml`) or an explicit adapter path |
| `--python`, `--hardware`, `--group` | The cell coordinates (defaults: `3.12`, none, `full`) |
| `--repo` | Candidate worktree to install from (default: the harness worktree) |
| `--work-dir` | Parent of all environments and caches (default: `openadmet-bench` in the system temp directory) |
| `--out-dir` | Logs and sidecar files for the run (default: `bench/results/runs/<run_id>`) |
| `--results` | JSONL file to append to (default: `bench/results/results.jsonl`) |
| `--phases` | Comma list of phases; `default` expands per group; add `lock` or `resolve_asof` explicitly |
| `--reps` | Repetitions of the quick `import` and `cli` phases (default 3) |
| `--cold-reps`, `--warm-reps` | Repetitions of cold and warm installs (default 1 each) |
| `--cache-mode` | `isolated` (default) or `global` |
| `--yes-clear-cache` | Confirms that the adapter `clear_cache` command may purge the real caches (global mode only) |
| `--asof YYYY-MM-DD` | Date for the `resolve_asof` phase |
| `--source` | `copy` (default) runs from a copy of the worktree; `in-place` runs in the worktree |
| `--keep-env` | Leave the environment and source copy on disk after the run |
| `--timeout` | Seconds allowed per command (default 7200) |
| `--check-adapters` | Validate every adapter file and exit |

Exit codes: 0 for a successful run, plan, or skipped cell; 1 when the cell ran and a phase failed; 2 for a refusal or usage error.

### Phases

Phases run in this order. A failed install skips every phase that needs the environment; other failures do not stop later phases.

| Phase | What is timed | Default |
|---|---|---|
| `lock_check` | Adapter `lock_check` command on the unmodified source copy | yes |
| `lock` | Adapter `lock` command (regenerates the lockfile in the source copy; later installs use it) | no |
| `install_cold` | Install after the caches are cleared, repeated `--cold-reps` times | yes |
| `install_warm` | Install with populated caches, repeated `--warm-reps` times | yes |
| `import` | `python -c "import openadmet.models; import openadmet.models.inference.inference"`, `--reps` times | yes |
| `cli` | `openadmet --help`, `--reps` times | yes |
| `device_smoke` | `pytest openadmet/models/tests/unit/test_devices.py` with `OPENADMET_EXPECT_DEVICE=<hardware>` (group `full`); an equivalent inline torch check (group `infer`, where pytest is absent) | yes |
| `unit_tests` | `pytest openadmet/models/tests/unit -n auto -q` | group `full` only |
| `predict_smoke` | `openadmet predict` on a generated model directory (see Limitations) | group `infer` only |
| `resolve_asof` | Adapter `lock_asof` command with `{date}` filled from `--asof`, run in a fresh source copy | no |

The `import`, `cli`, and `predict_smoke` commands run from an empty directory so that the installed package is imported instead of the source tree. The test phases run from the source copy, as CI does.

After the last install the driver records environment size, package list, package-set hash, and Python and torch facts, then runs the checks. Unless `--keep-env` is given, the environment and source copy are removed at the end.

## Running on GitHub Actions

The workflow `.github/workflows/env_benchmark.yaml` runs the harness on GitHub runners and aggregates the results. One job covers one operating system, Python version, and group, and runs every candidate in that job so the candidates share a host and a network. Candidates run in a rotated order in each round. The script `bench/ci/run_matrix.sh` contains the loop and validates every input before it reaches a command line.

| Cells | Runner |
|---|---|
| `linux-cpu` | `ubuntu-latest` |
| `mac-cpu` | `macos-latest` (arm64) |
| `linux-cuda` | AWS g4dn runner, only when the `run_gpu` input is true |
| `mac-mps` | Not available on hosted runners; run it on a physical Mac with `run.py` |

Inputs: `candidates` (space-separated adapter names), `rounds` (default 3), `run_gpu` (default false, starts a paid instance), and `asof_date` (adds the time-travel resolve phase). A push to a branch matching `env/bench/**` runs the workflow with the defaults, for example `git push origin env/harness:env/bench/run1`. Each candidate branch is checked out from the refs listed in the workflow `env` block.

Each job uploads a results directory (`.jsonl` plus the per-run logs), and the `aggregate` job writes the `analyze.py` tables to the job summary and uploads `combined-results`. Caches are disabled in the tool setup steps and the harness redirects them below the work directory, so cold installs are cold. A candidate that fails a phase is recorded in the results and does not fail the job; a job fails only when no cell ran.

To check the loop without installing anything, run the script with `DRY_RUN=1` and the variables listed at the top of `bench/ci/run_matrix.sh`.

## Safety

* Environments are created only below `--work-dir`. The driver creates a `.openadmet-bench-workdir` marker there and refuses to use an existing non-empty directory without it, and refuses `/`, the home directory, and any directory that contains or is inside the repository.
* Every deletion goes through one function that resolves the path and refuses anything that is not strictly inside `--work-dir` (a symlink is unlinked and its target is untouched).
* In the default `isolated` cache mode, the adapter `isolated_env` table redirects every package and download cache to `<work-dir>/cache/<candidate>`. A cold run deletes only that directory and your real caches are untouched. This also makes cold runs comparable across machines.
* In `global` mode a cold run executes the adapter `clear_cache` command against the real caches (for example `micromamba clean --all`, `uv cache clean`). The driver refuses to start without `--yes-clear-cache`. Use this mode only on a machine where purging caches is acceptable.
* The candidate worktree is copied (tracked and untracked files that are not ignored) into `--work-dir`, so lockfile regeneration and `pip install .` do not modify the worktree. `--source in-place` disables this.
* `--dry-run` starts no process and creates no file or directory.
* Commands are split with `shlex` and run without a shell.

Environments and caches are large (several GB each). Point `--work-dir` at a disk with room; on many Linux systems the default temp directory is a small RAM disk.

## Adapters

One TOML file per candidate lives in `bench/adapters/`. The file stem must equal the `name` key. Run `python3 -I bench/run.py --check-adapters` to validate all of them.

```toml
schema = 1                      # required, must be 1
name = "uv"                     # required, equals the file stem
status = "ready"                # "ready" or "draft"; draft adapters may contain TODO and cannot run
description = "..."             # free text

installers = ["micromamba", "mamba"]   # optional; first executable on PATH becomes {installer}

tool_version = ["uv --version"]        # string or list; output is recorded
lock         = "..."                   # optional; regenerate the lockfile from the manifest
lock_check   = "..."                   # optional; verify the lockfile is current, no install
lock_asof    = "... {date} ..."        # optional; re-lock as of a past date (YYYY-MM-DD)
run_prefix   = "uv run --no-sync"      # required; prefix that runs a command inside the environment
package_list = "..."                   # optional; prints the full package list
package_list_format = "freeze"         # freeze | conda-json | conda-export
clear_cache  = ["..."]                 # required; purges real caches (global mode only)
cache_dirs   = ["~/.cache/uv"]         # real cache paths whose size is recorded in global mode
env_location = "{env_dir}"             # directory measured as environment size; default {env_dir}

[env]                                  # extra environment variables for every command
UV_PROJECT_ENVIRONMENT = "{env_dir}"

[isolated_env]                         # extra variables in isolated cache mode
UV_CACHE_DIR = "{cache_dir}/uv"

[group_aliases]                        # optional; maps a harness group to the tool's own environment group
full = "dev"                           # exposed as {tool_group}; defaults to the harness group

[install.<hardware>.<group>]           # hardware: cpu|mps|cuda|any; group: full|infer|any
note  = "free text recorded with the result"
steps = ["cmd one", "cmd two"]
```

Rules:

* A command value is a string or a list. A list item is a string or an inline table `{ cmd = "...", allow_failure = true }`. An empty or omitted value means "not applicable" and the phase is skipped.
* Install lookup order is `<hardware>.<group>`, `<hardware>.any`, `any.<group>`, `any.any`. A cell with no match is recorded with `status = "skipped"`.
* Placeholders: `{python}` (for example `3.12`), `{python_nodot}` (`312`), `{tool_group}` (the group, or its alias), `{hardware}`, `{group}`, `{repo}` (the source copy), `{env_dir}`, `{work_dir}`, `{cache_dir}`, `{installer}` (only when `installers` is present), `{harness_python}`, and `{date}` (only in `lock_asof`). Write `{{` and `}}` for literal braces.
* Each command is split with `shlex` first and placeholders are expanded per token, so values with spaces stay intact. There is no shell, so pipes, `&&`, and `$VAR` are not interpreted; use a list of steps and the `[env]` tables instead.
* Install steps run with the source copy as working directory and must create the environment at `{env_dir}`. `run_prefix` must make `python`, `pytest`, and `openadmet` resolve inside that environment.
* A `ready` adapter must define `tool_version`, `run_prefix`, `clear_cache`, and at least one `install` table, and must contain no `TODO`. Unknown keys and placeholders are errors.
* A `draft` adapter is a skeleton: `--dry-run` works, a real run is refused.

The `baseline-conda` adapter selects `micromamba` when it is on `PATH` and falls back to `mamba`. The result records which one ran in `installer.selected`.

## Result records

Each run appends one JSON line. Field definitions are in [schema.md](schema.md). Logs for every command, `packages.txt` (the package list that was hashed) and `python-dists.txt` (Python distributions from `importlib.metadata`, comparable across tools) are stored in the run's `--out-dir`.

## Running a cell on another machine

On a Mac or Linux box with a checkout of this branch:

```bash
git clone <repo> && cd <repo> && git checkout env/harness
# Candidate worktree to test; for the baseline, the harness checkout itself works
python3 -I bench/run.py --list-cells
python3 -I bench/run.py --candidate baseline-conda --python 3.12 --hardware cpu --group full \
  --work-dir /path/with/free/space --results mac-m2-cpu.jsonl --dry-run
# Remove --dry-run to run for real; installs download several GB
python3 -I bench/run.py --candidate baseline-conda --python 3.12 --hardware cpu --group full \
  --work-dir /path/with/free/space --results mac-m2-cpu.jsonl
```

Send back the `.jsonl` file (and the `runs/<run_id>` directory when a run failed, since the logs explain why). The host record contains CPU model, RAM, operating system version, GPU model and driver, and a short hash of the hostname.

Run the same cell at least three times (`--cold-reps 3` or three invocations) so the analysis can check reproducibility.

## Analysis

```bash
python3 -I bench/analyze.py a.jsonl b.jsonl [--candidate uv] [--cell mac-cpu] [--python 3.12] [--group full]
```

`analyze.py` uses only the standard library. For every cell it prints Markdown tables: run outcomes and success rate, median (min-max) seconds per phase with one column per candidate, disk footprint, pytest counts, reproducibility, and time-travel resolve results. Reproducibility compares package-set hashes across repetitions on one host and across hosts of the same platform and machine. Compare tools with the `python dists` hash, because the `package set` hash covers whatever the adapter lists (conda libraries for conda tools, Python packages for pip tools).

## Limitations

* `predict_smoke` builds its fixture with `bench/fixtures/predict_fixture.py`, which mirrors the `fingerprint_model_dir` fixture from `openadmet/models/tests/unit/conftest.py`. That conftest imports pytest, so it cannot run in the `infer` group (pytest is a test dependency); the script duplicates the recipe layout instead. It exercises RDKit, molfeat, the YAML loader, the model loader, and `openadmet predict` with a constant dummy regressor. It does not exercise trained chemprop, lightgbm, or any other backend, and a change to the recipe layout in the package requires updating the script. If the fixture build fails, the failure is the signal.
* The `infer` group for `device_smoke` uses an inline matmul check, not the Lightning trainer test, because pytest is absent from that group.
* `import` times the first import after install, so it includes any bytecode compilation.
* `pip install git+https://...` dependencies are fetched from the network in every install, warm or cold, so warm installs of tools that build those wheels will not reach a pure-cache time.
* The first warm install without a prior cold install in the same run uses whatever the cache already contains.
* Timings are wall clock on a possibly busy machine; the harness does not control background load. Run on an idle machine and repeat.
* `package_list` for `baseline-conda` parses `micromamba list --json`; that format was written from the documented conda output and has not been run against this version of mamba.

## Tests

```bash
python -m pytest bench/tests -q
```

The tests use fake adapters whose commands are `echo`, `mkdir`, `touch`, and `sh`, and synthetic result records. They need pytest but no network and no package manager.
