#!/bin/sh
# Apply one OpenMP sharing variant to a uv environment on macOS
# Usage: apply_variant.sh VARIANT VENV_DIR
# Prints DYLD_INSERT_LIBRARIES (or nothing) to stdout as "KEY=VALUE" lines for the caller to export
set -eu

variant="$1"
venv="$2"
site="$(echo "$venv"/lib/python*/site-packages)"
torch_omp="$site/torch/lib/libomp.dylib"
brew_omp="$(brew --prefix libomp 2>/dev/null || true)/lib/libomp.dylib"

link_all_to() {
    target="$1"
    for lib in "$site/sklearn/.dylibs/libomp.dylib" "$site/torch/lib/libomp.dylib"; do
        if [ "$lib" != "$target" ] && { [ -e "$lib" ] || [ -L "$lib" ]; }; then
            ln -sf "$target" "$lib"
        fi
    done
}

case "$variant" in
none | brew-only)
    ;;
brew-relink)
    link_all_to "$brew_omp"
    ;;
torch-loader)
    # Put a symlink to the torch runtime next to the LightGBM and XGBoost libraries
    link_all_to "$torch_omp"
    ln -sf "$torch_omp" "$site/lightgbm/lib/libomp.dylib"
    ln -sf "$torch_omp" "$site/xgboost/lib/libomp.dylib"
    ;;
torch-rpath)
    # Add an rpath to the torch runtime directory on the two libraries that need @rpath/libomp.dylib
    link_all_to "$torch_omp"
    for lib in "$site/lightgbm/lib/lib_lightgbm.dylib" "$site/xgboost/lib/libxgboost.dylib"; do
        install_name_tool -add_rpath "@loader_path/../../torch/lib" "$lib"
        codesign --force --sign - "$lib"
    done
    ;;
preload-brew)
    echo "DYLD_INSERT_LIBRARIES=$brew_omp"
    ;;
preload-torch)
    echo "DYLD_INSERT_LIBRARIES=$torch_omp"
    ;;
*)
    echo "unknown variant $variant" >&2
    exit 2
    ;;
esac
