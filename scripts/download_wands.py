"""Download the WANDS dataset (Wayfair, MIT licence) into data/wands/.

Usage (from repo root):  python scripts/download_wands.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = ("product.csv", "query.csv", "label.csv")
TARGET = Path(__file__).resolve().parents[1] / "data" / "wands"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = TARGET / name
        if dest.exists() and "--force" not in sys.argv:
            print(f"{name}: already present ({dest.stat().st_size:,} bytes), skipping")
            continue
        print(f"{name}: downloading...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        tmp.replace(dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
        print(f"{name}: {dest.stat().st_size:,} bytes, sha256 {digest}…")


if __name__ == "__main__":
    main()
