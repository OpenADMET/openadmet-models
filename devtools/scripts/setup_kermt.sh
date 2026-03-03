#!/usr/bin/env bash
set -euo pipefail

KERMT_GIT_URL="${KERMT_GIT_URL:-https://github.com/NVIDIA-Digital-Bio/KERMT.git}"
KERMT_REF="${KERMT_REF:-main}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [target_dir]"
  exit 1
fi

if [[ $# -eq 1 ]]; then
  target_dir="$1"
else
  target_dir="${KERMT_REPO_PATH:-$HOME/.openadmet/KERMT}"
fi

target_dir="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$target_dir")"
target_parent="$(dirname "$target_dir")"

mkdir -p "$target_parent"

if [[ -d "$target_dir/.git" ]]; then
  echo "Updating existing KERMT checkout at: $target_dir"
  git -C "$target_dir" fetch --depth 1 origin "$KERMT_REF"
  git -C "$target_dir" checkout -q FETCH_HEAD
else
  if [[ -e "$target_dir" ]]; then
    echo "Target path exists and is not a git checkout: $target_dir"
    exit 1
  fi
  echo "Cloning KERMT into: $target_dir"
  git clone --depth 1 --branch "$KERMT_REF" "$KERMT_GIT_URL" "$target_dir"
fi

if [[ ! -f "$target_dir/main.py" ]]; then
  echo "KERMT checkout is missing main.py: $target_dir/main.py"
  exit 1
fi

echo
echo "KERMT setup complete."
echo "Export this in your shell before running OpenADMET KERMT workflows:"
echo "export KERMT_REPO_PATH=\"$target_dir\""
