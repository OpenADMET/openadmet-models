#!/bin/sh
# Point the OpenMP runtimes bundled in PyPI wheels at the Homebrew libomp so one runtime is loaded per process
#
# The macOS wheels of torch and scikit-learn bundle their own libomp, while lightgbm and xgboost link the
# Homebrew copy. Loading several runtimes into one process crashes or stalls inside LightGBM and XGBoost.
# Run this after every `uv sync` that installs or reinstalls torch or scikit-learn.
#
# Usage: share_libomp_macos.sh [environment-directory]   (default: $UV_PROJECT_ENVIRONMENT or .venv)

set -eu

venv="${1:-${UV_PROJECT_ENVIRONMENT:-.venv}}"

# Locate the Homebrew runtime
libomp="$(brew --prefix libomp)/lib/libomp.dylib"
if [ ! -f "$libomp" ]; then
  echo "Homebrew libomp not found at $libomp; run: brew install libomp" >&2
  exit 1
fi

# Replace each bundled copy with a link to the Homebrew runtime
linked=0
for lib in "$venv"/lib/python*/site-packages/sklearn/.dylibs/libomp.dylib \
           "$venv"/lib/python*/site-packages/torch/lib/libomp.dylib; do
  if [ -e "$lib" ] || [ -L "$lib" ]; then
    ln -sf "$libomp" "$lib"
    echo "Linked $lib -> $libomp"
    linked=$((linked + 1))
  fi
done

if [ "$linked" -eq 0 ]; then
  echo "No bundled libomp found under $venv; nothing to link" >&2
fi
