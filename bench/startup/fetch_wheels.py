"""Download the locked macOS arm64 wheels of the named packages and check them against the lock hashes.

Usage: fetch_wheels.py UV_LOCK OUTPUT_DIR PACKAGE [PACKAGE ...]
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
import urllib.request
from pathlib import Path


def main(lock: str, out_dir: str, packages: list[str]) -> int:
    """Download one wheel per package and return the number of failures."""
    data = tomllib.loads(Path(lock).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in packages:
        entry = next(p for p in data["package"] if p["name"] == name)
        wheels = [w for w in entry["wheels"] if "macosx" in w["url"] and "arm64" in w["url"]]
        if len(wheels) != 1:
            print(f"{name}: expected one macOS arm64 wheel, found {len(wheels)}", file=sys.stderr)
            failures += 1
            continue
        wheel = wheels[0]
        target = out / wheel["url"].rsplit("/", 1)[-1]
        urllib.request.urlretrieve(wheel["url"], target)
        digest = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"{target.name} {digest} matches lock: {digest == wheel['hash']}")
        failures += digest != wheel["hash"]
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
