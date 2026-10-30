#!/bin/sh
# Make every OpenMP user in a uv environment load the libomp bundled with torch, so one runtime exists per process
#
# The macOS wheels of torch and scikit-learn each bundle a libomp, while lightgbm and xgboost link
# @rpath/libomp.dylib and search the Homebrew location. Loading several runtimes into one process
# stalls or crashes inside LightGBM and XGBoost. This script links the scikit-learn copy to the torch
# copy and points the lightgbm and xgboost libraries at the torch directory, so Homebrew libomp is not needed.
# Run this after every `uv sync` that installs or reinstalls torch, scikit-learn, lightgbm, or xgboost.
# Requires install_name_tool and codesign from the Xcode command line tools.
#
# Usage: share_libomp_macos.sh [environment-directory]   (default: $UV_PROJECT_ENVIRONMENT or .venv)

set -eu

venv="${1:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
rpath_new="@loader_path/../../torch/lib"
patched=0

for site in "$venv"/lib/python*/site-packages; do
  torch_omp="$site/torch/lib/libomp.dylib"
  if [ ! -f "$torch_omp" ]; then
    continue
  fi

  # Replace the scikit-learn copy with a link to the torch runtime
  sk_omp="$site/sklearn/.dylibs/libomp.dylib"
  if [ -e "$sk_omp" ] || [ -L "$sk_omp" ]; then
    ln -sf "$torch_omp" "$sk_omp"
    echo "Linked $sk_omp -> $torch_omp"
  fi

  # Point the libraries that search for libomp at the torch directory, ahead of Homebrew and MacPorts paths
  for lib in "$site/lightgbm/lib/lib_lightgbm.dylib" "$site/xgboost/lib/libxgboost.dylib"; do
    if [ ! -f "$lib" ]; then
      continue
    fi

    changed=0
    has_new=0
    for rpath in $(otool -l "$lib" | awk '/cmd LC_RPATH/ {f = 1} f && $1 == "path" {print $2; f = 0}'); do
      case "$rpath" in
        "$rpath_new")
          has_new=1
          ;;
        *libomp*)
          # Rewrite the first libomp search path to the torch directory and drop the others
          if [ "$has_new" -eq 0 ]; then
            install_name_tool -rpath "$rpath" "$rpath_new" "$lib"
            has_new=1
          else
            install_name_tool -delete_rpath "$rpath" "$lib"
          fi
          changed=1
          ;;
      esac
    done

    # Add the search path when the library had no libomp entry to rewrite
    if [ "$has_new" -eq 0 ]; then
      install_name_tool -add_rpath "$rpath_new" "$lib"
      changed=1
    fi

    # Re-sign after editing, since editing invalidates the signature
    if [ "$changed" -eq 1 ]; then
      codesign --force --sign - "$lib"
      echo "Pointed $lib at $rpath_new"
    fi
    patched=$((patched + 1))
  done
done

if [ "$patched" -eq 0 ]; then
  echo "No lightgbm or xgboost library found under $venv; nothing to patch" >&2
fi
