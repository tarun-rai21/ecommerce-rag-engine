"""Copy the local Qdrant collection to a managed Qdrant (e.g. Qdrant Cloud) as a snapshot,
so the 43k products are never re-embedded (about an hour on a laptop CPU).

    python deploy/upload_snapshot.py

Run it yourself: it asks for the cluster URL and API key and never prints or stores the
key. Needs the local stack running (Qdrant on localhost:6333).
"""
import argparse
import getpass
import tempfile
import time
from pathlib import Path

import requests

LOCAL = "http://localhost:6333"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default="wands_products")
    args = p.parse_args()

    url = input("Qdrant Cloud cluster URL (https://...:6333): ").strip().rstrip("/")
    key = getpass.getpass("Qdrant Cloud API key (input hidden): ").strip()
    headers = {"api-key": key}
    requests.get(f"{url}/collections", headers=headers, timeout=30).raise_for_status()  # fail fast on a bad URL/key

    print(f"Creating a snapshot of '{args.collection}' locally...")
    r = requests.post(f"{LOCAL}/collections/{args.collection}/snapshots", params={"wait": "true"}, timeout=600)
    r.raise_for_status()
    name = r.json()["result"]["name"]
    path = Path(tempfile.mkdtemp()) / name
    with requests.get(f"{LOCAL}/collections/{args.collection}/snapshots/{name}", stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with path.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    requests.delete(f"{LOCAL}/collections/{args.collection}/snapshots/{name}", timeout=60)
    print(f"Snapshot: {path.stat().st_size / 1e6:.0f} MB. Uploading (this can take a few minutes)...")

    start = time.perf_counter()
    with path.open("rb") as fh:
        r = requests.post(f"{url}/collections/{args.collection}/snapshots/upload",
                          params={"priority": "snapshot", "wait": "true"}, headers=headers,
                          files={"snapshot": (name, fh, "application/octet-stream")}, timeout=3600)
    r.raise_for_status()
    path.unlink()
    info = requests.get(f"{url}/collections/{args.collection}", headers=headers, timeout=30).json()["result"]
    sparse = info["config"]["params"]["sparse_vectors"]["sparse"]
    print(f"Done in {time.perf_counter() - start:.0f} s: {info['points_count']} points, "
          f"BM25 IDF modifier: {sparse.get('modifier')}")


if __name__ == "__main__":
    main()
