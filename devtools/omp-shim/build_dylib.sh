#!/bin/sh
# Build the arm64 libomp shim that re-exports torch's runtime, for committing as src/omp_shim/libomp.dylib
# Usage: build_dylib.sh OUTPUT_FILE
set -eu

out="$1"
work="$(mktemp -d)"
echo "" > "$work/empty.c"

# Link a placeholder whose install name is torch's relative path, then re-export it from the shim
clang -dynamiclib -arch arm64 -mmacosx-version-min=11.0 -o "$work/placeholder.dylib" "$work/empty.c" \
  -install_name "@loader_path/../torch/lib/libomp.dylib"
clang -dynamiclib -arch arm64 -mmacosx-version-min=11.0 -o "$out" "$work/empty.c" \
  -install_name @rpath/libomp.dylib -Wl,-reexport_library,"$work/placeholder.dylib"
codesign --force --sign - "$out"
otool -L "$out"
rm -rf "$work"
