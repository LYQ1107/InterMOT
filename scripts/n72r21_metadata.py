"""Bounded official metadata acquisition with logged direct-first transport.

No data snapshot, video, crop, credential, CAPTCHA or gated-file bypass.
GitHub CSVs describe split membership, not permission to access gated media.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

from scripts.n72r21_common import ASSETS, ROOT, OUT, write_json, sha256, storage, utcnow


def fetch(url, destination, max_bytes=8 << 20, direct_first=True):
    destination = Path(destination)
    if not destination.resolve().is_relative_to(ASSETS.resolve()):
        raise ValueError("source download destination must be inside N72R21 assets")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {"url": url, "path": str(destination), "status": "REUSED_METADATA", "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    attempts = []
    for direct in ([True, False] if direct_first else [False]):
        storage(max_bytes)
        partial = destination.with_name(destination.name + ".partial")
        args = ["curl", "-4", "--location", "--fail", "--silent", "--show-error", "--connect-timeout", "5", "--max-time", "18",
                "--max-filesize", str(max_bytes), "--output", str(partial), "--write-out", "%{http_code}|%{size_download}|%{time_total}|%{url_effective}"]
        if direct: args.extend(["--noproxy", "*"])
        started = time.monotonic()
        result = subprocess.run(args + [url], capture_output=True, text=True, timeout=22)
        parts = result.stdout.strip().split("|", 3)
        # Signed redirects must not be copied into research reports.
        effective = urlsplit(parts[3]) if len(parts) == 4 else None
        attempts.append({"direct_no_proxy": direct, "existing_proxy_fallback": not direct, "exit_code": result.returncode,
                         "http": parts[0] if parts else None, "elapsed_s": round(time.monotonic() - started, 3),
                         "bytes": partial.stat().st_size if partial.exists() else 0, "error": result.stderr.strip()[:400],
                         "effective_origin_path": urlunsplit((effective.scheme, effective.netloc, effective.path, "", "")) if effective else None})
        if result.returncode == 0:
            if partial.stat().st_size > max_bytes: raise RuntimeError("metadata size cap exceeded")
            partial.replace(destination)
            return {"url": url, "path": str(destination), "status": "DOWNLOADED_METADATA", "bytes": destination.stat().st_size,
                    "sha256": sha256(destination), "attempts": attempts, "utc": utcnow()}
        # Retain partial/error evidence; do not use an unsuccessful response as data.
    return {"url": url, "path": str(destination), "status": "METADATA_NETWORK_OR_ACCESS_BLOCKED", "attempts": attempts}


def run():
    source = ASSETS / "official_sources/CHIRLA"
    record = fetch("https://api.github.com/repos/bdager/CHIRLA/git/trees/main?recursive=1", source / "TREE.json")
    if record["status"] == "METADATA_NETWORK_OR_ACCESS_BLOCKED":
        write_json("datasets/DOWNLOAD_MANIFEST.json", {"metadata": [record], "video_downloads": [], "full_snapshot_download": False})
        raise RuntimeError("official metadata tree inaccessible")
    tree = json.loads((source / "TREE.json").read_text())
    revision = tree["sha"]
    selected = []
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob": continue
        if (path in {"README.md", "LICENSE", "LICENSE.md"}
            or path.startswith("downloader/") and path.endswith((".md", ".py", ".json"))
            or path.startswith("benchmark/") and path.endswith((".md", ".csv", ".py", ".json"))
            or path.startswith("data/") and path.endswith(".md")):
            if entry.get("size", 0) > 8 << 20: continue
            selected.append(entry)
    if sum(e.get("size", 0) for e in selected) > 128 << 20:
        raise RuntimeError("D0 metadata budget exceeded")
    def acquire(entry):
        return fetch(f"https://raw.githubusercontent.com/bdager/CHIRLA/{revision}/{entry['path']}", source / entry["path"])
    records = [record]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(acquire, selected):
            records.append(row)
            print(json.dumps({"source_file": str(Path(row["path"]).relative_to(source)), "status": row["status"], "bytes": row.get("bytes")}), flush=True)
    hf = fetch("https://huggingface.co/api/datasets/bdager/CHIRLA", ASSETS / "official_sources/CHIRLA_HF_INFO.json")
    records.append(hf)
    # A public card/API can be inspected without accepting terms. Files are not
    # requested if metadata says they are gated.
    gate = None
    if hf["status"] != "METADATA_NETWORK_OR_ACCESS_BLOCKED":
        info = json.loads(Path(hf["path"]).read_text())
        gate = info.get("gated")
    doi = fetch("https://doi.org/10.57760/sciencedb.20543", ASSETS / "official_sources/CHIRLA_SCIENCEDB_LANDING.html")
    records.append(doi)
    manifest = {"utc": utcnow(), "metadata": records, "github_revision": revision, "hf_access_gate": gate,
                "video_downloads": [], "crop_downloads": [], "full_snapshot_download": False,
                "final_media_accessed": False, "direct_failures_precede_proxy_fallback": True}
    write_json("datasets/DOWNLOAD_MANIFEST.json", manifest)
    write_json("datasets/DATASET_SOURCE_AUDIT.json", {"official_github": "https://github.com/bdager/CHIRLA", "official_hf": "https://huggingface.co/datasets/bdager/CHIRLA",
               "official_sciencedb_doi": "https://doi.org/10.57760/sciencedb.20543", "hf_gate": gate,
               "hf_files_download_authorized": False if gate else None, "transport_records": records,
               "baidu_first_direct_attempt": "Two recorded no-proxy failures; existing v4.0.2 CLI located, password/cookie login not used.",
               "unverified_landing_id_probe_not_accepted_as_source": True})
    print(json.dumps({"metadata_files": len(records), "hf_gate": gate, "github_revision": revision, "video_downloads": 0}))


if __name__ == "__main__": run()
