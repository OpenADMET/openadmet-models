#!/usr/bin/env bash
# Run the benchmark harness for every candidate, Python version, and group on this host
#
# All inputs arrive as environment variables so that workflow inputs are never
# interpolated into a command line:
#   HARNESS_DIR  checkout of the env/harness branch
#   CAND_ROOT    directory holding one candidate checkout per adapter name
#   WORK_DIR     scratch directory for environments and caches (must not exist yet)
#   RESULTS      JSONL file to append to
#   OUT_ROOT     directory for per-run logs and sidecar files
#   HARDWARE     cpu, mps, or cuda
#   PYTHONS      space-separated Python versions, e.g. "3.12 3.13"
#   BENCH_GROUPS space-separated dependency groups, e.g. "full infer"
#   CANDIDATES   space-separated adapter names
#   ROUNDS       repetitions of the whole candidate list per Python and group
#   ASOF_DATE    optional YYYY-MM-DD; adds the resolve_asof phase
#   DRY_RUN      optional; any non-empty value passes --dry-run so nothing is executed
# Candidates run in a rotated order each round so that no candidate always runs
# first on a host that warms up or degrades over the job.

set -uo pipefail

for var in HARNESS_DIR CAND_ROOT WORK_DIR RESULTS OUT_ROOT HARDWARE PYTHONS BENCH_GROUPS CANDIDATES ROUNDS; do
    if [[ -z "${!var:-}" ]]; then
        echo "error: ${var} is not set" >&2
        exit 2
    fi
done

# Validate every value that reaches a command line
name_re='^[a-z][a-z0-9-]*$'
version_re='^3\.[0-9]+$'
date_re='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
[[ "${HARDWARE}" =~ ^(cpu|mps|cuda)$ ]] || { echo "error: bad HARDWARE" >&2; exit 2; }
[[ "${ROUNDS}" =~ ^[1-9][0-9]?$ ]] || { echo "error: ROUNDS must be 1 to 99" >&2; exit 2; }
for c in ${CANDIDATES}; do
    [[ "${c}" =~ ${name_re} && -f "${HARNESS_DIR}/bench/adapters/${c}.toml" ]] || { echo "error: unknown candidate ${c}" >&2; exit 2; }
done
for p in ${PYTHONS}; do
    [[ "${p}" =~ ${version_re} ]] || { echo "error: bad Python version ${p}" >&2; exit 2; }
done
for g in ${BENCH_GROUPS}; do
    [[ "${g}" =~ ^(full|infer)$ ]] || { echo "error: bad group ${g}" >&2; exit 2; }
done

phase_args=()
if [[ -n "${ASOF_DATE:-}" ]]; then
    [[ "${ASOF_DATE}" =~ ${date_re} ]] || { echo "error: bad ASOF_DATE" >&2; exit 2; }
    phase_args=(--phases default,resolve_asof --asof "${ASOF_DATE}")
fi

dry_args=()
if [[ -n "${DRY_RUN:-}" ]]; then
    dry_args=(--dry-run)
fi

read -r -a candidates <<< "${CANDIDATES}"
n_candidates=${#candidates[@]}
mkdir -p "${OUT_ROOT}"

ran=0
refused=0
failed=0

for py in ${PYTHONS}; do
    for group in ${BENCH_GROUPS}; do
        for ((round = 0; round < ROUNDS; round++)); do
            for ((i = 0; i < n_candidates; i++)); do
                cand="${candidates[$(((i + round) % n_candidates))]}"
                label="r${round}-${cand}-py${py}-${HARDWARE}-${group}"

                echo "::group::${label}"
                df -h "${RUNNER_TEMP:-/tmp}" | tail -n 1
                python3 -I "${HARNESS_DIR}/bench/run.py" \
                    --candidate "${cand}" \
                    --python "${py}" \
                    --hardware "${HARDWARE}" \
                    --group "${group}" \
                    --repo "${CAND_ROOT}/${cand}" \
                    --work-dir "${WORK_DIR}" \
                    --results "${RESULTS}" \
                    --out-dir "${OUT_ROOT}/${label}" \
                    --cache-mode isolated \
                    --cold-reps 1 --warm-reps 1 --reps 3 \
                    --timeout 3600 \
                    ${phase_args[@]+"${phase_args[@]}"} \
                    ${dry_args[@]+"${dry_args[@]}"}
                status=$?
                echo "::endgroup::"

                # Exit 0 is a clean run, 1 is a failed phase (recorded), 2 is a refusal such as a draft adapter
                case "${status}" in
                    0) ran=$((ran + 1)) ;;
                    1) ran=$((ran + 1)); failed=$((failed + 1)); echo "::warning::${label} had a failing phase" ;;
                    *) refused=$((refused + 1)); echo "::warning::${label} was refused (exit ${status})" ;;
                esac

                # Free disk between runs; the cold phase of the next run recreates its cache
                rm -rf -- "${WORK_DIR:?}/cache" "${WORK_DIR:?}/envs" "${WORK_DIR:?}/src"
            done
        done
    done
done

echo "runs completed: ${ran} (with failing phases: ${failed}); refused: ${refused}"
if ((ran == 0)); then
    echo "::error::no cell ran; check that the adapters are ready and not drafts"
    exit 3
fi
