#!/bin/sh
# Build a libomp shim: install name @rpath/libomp.dylib, re-exports the libomp bundled with torch
# Usage: build_shim.sh SITE_PACKAGES OUTPUT_DIR
set -eu

site="$1"
out="$2"
mkdir -p "$out"
echo "" > "$out/empty.c"
clang -dynamiclib -arch arm64 -mmacosx-version-min=11.0 -o "$out/libomp.dylib" "$out/empty.c" \
  -install_name @rpath/libomp.dylib \
  -Wl,-reexport_library,"$site/torch/lib/libomp.dylib"

# Point the re-export at the torch copy relative to the shim location (out is a sibling of torch under site-packages)
old="$(otool -L "$out/libomp.dylib" | awk '/libomp/ && NR > 2 {print $1}' | head -1)"
echo "re-export target recorded as: $old"
install_name_tool -change "$old" "@loader_path/${SHIM_REL:-../torch/lib/libomp.dylib}" "$out/libomp.dylib"
codesign --force --sign - "$out/libomp.dylib"
otool -L "$out/libomp.dylib"
otool -l "$out/libomp.dylib" | grep -A3 -E "LC_ID_DYLIB|LC_REEXPORT" | grep -E "cmd|name"
