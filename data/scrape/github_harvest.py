#!/usr/bin/env python3
"""Harvest Rust source from crates.io published tarballs (license-filtered).

Named github_harvest.py for repo-layout continuity, but the primary source is
crates.io: published crates carry an authoritative SPDX `license` field per
version, which GitHub HEAD does not. A GitHub fallback can be added later for
repos that aren't published as crates.

Outputs (under --workdir, default data/work):
  raw/<crate>-<version>/...      extracted .rs files (only those that pass)
  provenance.jsonl               one record per KEPT file:
                                 {crate, version, org, crate_license, path,
                                  sha256, size_bytes, source_url}
  harvest_manifest.jsonl         {path, crate, version, license} per kept file
  harvest_stats.json             counters for the dedup/filter funnel

License policy: see sources.yaml. Rejections are COUNTED and the crate/file is
skipped; nothing unlicensed ever reaches the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import requests
import yaml

CRATES_API = "https://crates.io/api/v1/crates"

# Per-file license headers that contradict the crate-level grant.
_BAD_FILE_LICENSE = re.compile(
    r"SPDX-License-Identifier:\s*(?![ (]*(?:MIT|Apache-2\.0|BSD-2-Clause|BSD-3-Clause)\b)",
)


def spdx_allowed(expr: str | None, allowlist: set[str]) -> bool:
    """Strict allowlist check: every license token must be allowlisted.

    Deliberately stricter than SPDX OR-election (see sources.yaml). Operators
    are ignored; any non-operator token outside the allowlist rejects.
    """
    if not expr:
        return False
    tokens = re.split(r"[\s()/]+", expr)  # legacy crates use "MIT/Apache-2.0"
    seen_license = False
    for tok in tokens:
        if not tok or tok in {"OR", "AND", "WITH"}:
            continue
        if tok not in allowlist:
            return False
        seen_license = True
    return seen_license


class CratesClient:
    def __init__(self, user_agent: str, rate_limit_s: float):
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = user_agent
        self.rate_limit_s = rate_limit_s
        self._last = 0.0

    def _throttle(self):
        wait = self._last + self.rate_limit_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, **kw) -> requests.Response:
        self._throttle()
        r = self.sess.get(url, timeout=60, **kw)
        r.raise_for_status()
        return r


def newest_stable(versions: list[dict]) -> dict | None:
    for v in versions:  # API returns newest-first
        if not v["yanked"] and "-" not in v["num"]:  # skip 1.0.0-rc.1 etc.
            return v
    return None


def harvest_crate(client: CratesClient, spec: dict, cfg: dict, workdir: Path,
                  allowlist: set[str], stats: dict, max_files: int,
                  prov_out, manifest_out) -> None:
    name = spec["name"]
    meta = client.get(f"{CRATES_API}/{name}").json()
    ver = newest_stable(meta["versions"])
    if ver is None:
        print(f"  {name}: no stable version, skipping", file=sys.stderr)
        stats["crates_skipped_no_version"] += 1
        return

    license_expr = ver.get("license")
    if not spdx_allowed(license_expr, allowlist):
        print(f"  {name}: license {license_expr!r} not allowlisted -> REJECT crate",
              file=sys.stderr)
        stats["crates_rejected_license"] += 1
        return

    num = ver["num"]
    dl_url = f"{CRATES_API}/{name}/{num}/download"
    tarball = client.get(dl_url).content
    stats["crates_kept"] += 1

    include_dirs = tuple(cfg["include_dirs"])
    excludes = cfg["exclude_path_substrings"]
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
        for member in tf:
            if kept >= max_files:
                break
            if not member.isfile() or not member.name.endswith(".rs"):
                continue
            # member.name = "<crate>-<version>/src/lib.rs"
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not rel.startswith(include_dirs):
                continue
            if any(x in f"/{rel}" for x in excludes):
                continue
            stats["files_seen"] += 1
            content = tf.extractfile(member).read()
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                stats["files_rejected_encoding"] += 1
                continue
            if _BAD_FILE_LICENSE.search(text):
                stats["files_rejected_file_license"] += 1
                continue

            out_path = workdir / "raw" / f"{name}-{num}" / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            sha = hashlib.sha256(content).hexdigest()
            prov_out.write(json.dumps({
                "crate": name, "version": num, "org": spec.get("org"),
                "crate_license": license_expr, "path": str(out_path),
                "rel_path": rel, "sha256": sha, "size_bytes": len(content),
                "source_url": dl_url,
            }) + "\n")
            manifest_out.write(json.dumps({
                "path": str(out_path), "crate": name, "version": num,
                "license": license_expr,
            }) + "\n")
            kept += 1
            stats["files_kept"] += 1
    print(f"  {name} {num} [{license_expr}]: kept {kept} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default=Path(__file__).parent / "sources.yaml")
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    ap.add_argument("--max-files-per-crate", type=int, default=None,
                    help="override sources.yaml harvest.max_files_per_crate")
    ap.add_argument("--crates", nargs="*", default=None,
                    help="subset of crate names (tiny-test runs)")
    args = ap.parse_args()

    src = yaml.safe_load(Path(args.sources).read_text())
    cfg = src["harvest"]
    allowlist = set(src["license_allowlist"])
    max_files = args.max_files_per_crate or cfg["max_files_per_crate"]
    specs = [c for c in src["crates"]
             if args.crates is None or c["name"] in args.crates]

    args.workdir.mkdir(parents=True, exist_ok=True)
    stats = {k: 0 for k in [
        "crates_kept", "crates_rejected_license", "crates_skipped_no_version",
        "files_seen", "files_kept", "files_rejected_encoding",
        "files_rejected_file_license"]}

    client = CratesClient(cfg["user_agent"], cfg["rate_limit_seconds"])
    with open(args.workdir / "provenance.jsonl", "w") as prov, \
         open(args.workdir / "harvest_manifest.jsonl", "w") as manifest:
        for spec in specs:
            try:
                harvest_crate(client, spec, cfg, args.workdir, allowlist,
                              stats, max_files, prov, manifest)
            except requests.RequestException as e:
                print(f"  {spec['name']}: fetch failed ({e}), skipping",
                      file=sys.stderr)
                stats["crates_skipped_no_version"] += 1

    (args.workdir / "harvest_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
