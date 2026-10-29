"""Parallel, resumable PTB-XL downloader from PhysioNet's open S3 mirror.

Equivalent to `wget -r -N -c -np` on https://physionet.org/files/ptb-xl/1.0.3/ but much faster
(physionet.org serves one file at a time slowly). Skips files whose size already matches.

Usage: python data/download_ptbxl.py [--out data/raw/files/ptb-xl/1.0.3] [--hr] [--workers 16]
Default fetches CSVs + records100 (100 Hz). Add --hr to also fetch records500 (500 Hz).
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "ptb-xl/1.0.3/"


def list_objects(prefix):
    token, out = None, []
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(requests.get(BUCKET, params=params, timeout=60).content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            out.append((c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)))
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return out
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def fetch(key, size, dest: Path, retries=5):
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return False
    for attempt in range(retries):
        try:
            headers = {"Range": f"bytes={have}-"} if 0 < have < size else {}
            with requests.get(f"{BUCKET}/{key}", headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "ab" if r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return True
        except Exception as e:
            have = dest.stat().st_size if dest.exists() else 0
            print(f"retry {attempt + 1} {key}: {e}", file=sys.stderr)
    raise RuntimeError(f"failed: {key}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/files/ptb-xl/1.0.3")
    ap.add_argument("--hr", action="store_true", help="also download 500 Hz records")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    objects = [(k, s) for k, s in list_objects(PREFIX) if not k.endswith("/")
               and (a.hr or "/records500/" not in k)]
    total = sum(s for _, s in objects)
    print(f"{len(objects)} files, {total / 1e9:.2f} GB", flush=True)
    done = new = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futures = [ex.submit(fetch, k, s, Path(a.out) / k[len(PREFIX):]) for k, s in objects]
        for f in as_completed(futures):
            done += 1
            new += bool(f.result())
            if done % 1000 == 0:
                print(f"{done}/{len(objects)} files ({new} downloaded)", flush=True)
    print(f"done: {done} files, {new} newly downloaded")
