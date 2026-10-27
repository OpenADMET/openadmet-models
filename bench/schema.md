# Result record schema

`bench/run.py` appends one JSON object per line to the results file. `schema_version` is 1. Times are wall-clock seconds from `time.perf_counter`.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Always 1 |
| `run_id` | string | `YYYYMMDDTHHMMSSZ-<6 hex>`; also the log directory name by default |
| `timestamp_utc` | string | ISO 8601 start time in UTC |
| `candidate` | string | Adapter label, for example `baseline-conda` |
| `adapter_status` | string | `ready` or `draft` |
| `cell` | object | `id` (`mac-cpu`, `mac-mps`, `linux-cpu`, `linux-cuda`), `os`, `hardware`, `python` (requested minor version), `group` (`full` or `infer`) |
| `installer` | object | `candidates` (adapter list), `selected` (executable that ran, or null), `path` |
| `options` | object | `phases`, `reps`, `cold_reps`, `warm_reps`, `cache_mode`, `source`, `asof` |
| `tool_versions` | object | Maps each `tool_version` command to its first five output lines |
| `host` | object | See below |
| `git` | object | `candidate` and `harness`, each with `sha`, `branch`, `dirty` (null when git is unavailable) |
| `paths` | object | `env_dir`, `cache_dir`, `log_dir` used by the run |
| `phases` | object | Maps phase label to a phase entry (see below) |
| `env` | object or null | Environment facts after the last install (see below); null when no install succeeded |
| `env_facts` | object | `python_version`, `python_executable`, and `torch` or `torch_error`; present after a successful install |
| `torch` | object or null | `version`, `cuda_version`, `cuda_available`, `mps_available` |
| `status` | string | `ok`, `failed`, or `skipped` |
| `failed_phase` | string or null | First phase (in execution order) with status `failed` |
| `failure_tail` | list of string | Last 40 lines of the failing phase log; empty when ok |
| `wall_seconds_total` | float | Total run time including environment measurement and cleanup |
| `reason` | string | Only on skipped cells: why the cell was skipped |

A cell for which the adapter has no install entry produces a short record with `status = "skipped"`, `reason`, `host`, `git`, and empty `phases`.

## `host`

`platform`, `system`, `machine`, `kernel`, `cpu_count`, `cpu_model`, `ram_bytes`, `os_version` (macOS version or Linux `PRETTY_NAME`), `harness_python` and `harness_python_executable` (interpreter that ran the driver), `hostname_hash` (first 10 hex digits of the sha256 of the hostname), and `gpus` (list of `{name, driver}` from `nvidia-smi`, empty when absent).

## Phase entries

Every entry has `status` (`ok`, `failed`, `skipped`). Skipped entries have `reason`. Executed entries have:

| Field | Meaning |
|---|---|
| `exit_codes` | One exit code per repetition (124 is a timeout, 127 a missing executable) |
| `seconds` | One duration per repetition |
| `median_s`, `min_s`, `max_s` | Summary of `seconds` |
| `logs` | Log file paths relative to `paths.log_dir` |
| `tail` | Last 40 lines of the failing log (failed phases only) |

Phase-specific fields:

| Phase | Extra fields |
|---|---|
| `install_cold`, `install_warm` | `note` (adapter install note, optional), `cache_size_bytes` (map of cache directory to bytes after the phase) |
| `device_smoke`, `unit_tests` | `mode` (`pytest` or `inline`), `pytest` (counts below, or null) |
| `predict_smoke` | `stage` (`build`, `predict`, or `check`: the last stage reached), `fixture_seconds` (fixture build time); `seconds` is the `openadmet predict` time |
| `resolve_asof` | `asof` (the requested date); on failure `tail` contains the solver error |

`pytest` counts: `passed`, `failed`, `skipped`, `errors`, `xfailed`, `xpassed`, `warnings`, `deselected`, `duration_s`, parsed from the final `-q` summary line.

The `import` and `cli` phases repeat `--reps` times; `install_cold` repeats `--cold-reps` times and `install_warm` repeats `--warm-reps` times. All other phases run once.

## `env`

| Field | Meaning |
|---|---|
| `location` | Measured directory (adapter `env_location`) |
| `size_bytes` | Allocated disk usage with hardlinks counted once (comparable with `du`) |
| `n_packages` | Entries in the hashed package list |
| `package_list_source` | `adapter` (the `package_list` command) or `importlib.metadata` |
| `package_set_hash` | sha256 of the sorted `name==version` lines of that list, each followed by a newline |
| `package_list_file` | Sidecar file in `paths.log_dir` with the hashed list |
| `python_dist_count`, `python_dist_hash`, `python_dist_file` | Same for Python distributions from `importlib.metadata` (names normalized per PEP 503); comparable across tools |
