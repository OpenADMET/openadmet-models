#!/bin/bash
# Build a throwaway venv from pip specs and probe OpenMP import orders in it
# Usage: [SKIP_INSTALL=1] pkgs_screen.sh LABEL PYTHON_VERSION SPEC...
# Prints the OpenMP dylib install names found in the venv, then one result row per import order
set -u

label="$1"
python_version="$2"
shift 2

venv="${RUNNER_TEMP:-/tmp}/venv-$label"
here="$(cd "$(dirname "$0")" && pwd)"

# Create the environment and install the specs, unless the caller already built the venv
if [ -z "${SKIP_INSTALL:-}" ]; then
  uv venv --quiet --python "$python_version" "$venv"
  if ! uv pip install --quiet --python "$venv/bin/python" "$@"; then
    echo "RESULT $label install-failed"
    exit 0
  fi
fi

# Show the install name and rpath of every OpenMP runtime and of the libraries that link one
echo "== $label: OpenMP runtimes"
find "$venv" -name '*omp*.dylib' | sort | while read -r lib; do
  echo "$lib: id=$(otool -D "$lib" | tail -1)"
done
for lib in $(find "$venv" \( -name 'lib_lightgbm.dylib' -o -name 'libxgboost*.dylib' -o -name 'libtorch_cpu.dylib' \) | sort); do
  echo "$lib"
  otool -L "$lib" | grep -E 'libi?omp' || true
  otool -l "$lib" | awk '/cmd LC_RPATH/ {f = 1} f && $1 == "path" {print "  rpath", $2; f = 0}'
done

# Probe the import orders and print one row per order
echo "== $label: probe"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-30}" python3 -I "$here/run_probe.py" "$venv/bin/python" > "${RUNNER_TEMP:-/tmp}/probe-$label.txt" 2>&1
sed -n '/^| import order/,$p' "${RUNNER_TEMP:-/tmp}/probe-$label.txt"
sed -n '/^| import order/,$p' "${RUNNER_TEMP:-/tmp}/probe-$label.txt" | tail -n +3 | sed "s/^/RESULT $label /"
