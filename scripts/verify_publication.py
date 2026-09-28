#!/usr/bin/env python3
"""Read-only, anonymous verification of the public W4A16 model release.

Run after uploads have completed. Large artifacts are checked against the hosting
service's content digest; selected small HF files are also downloaded afresh.
Existing receipt files are never overwritten. Requires huggingface_hub and httpx.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ImportError:  # huggingface_hub 2.x bundles the renamed compatible client.
    import httpx2 as httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def hashes(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def local_inventory(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir(), f"Local directory missing: {root}")
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in (".git", ".cache"):
            continue
        require(not path.is_symlink(), f"Unexpected symlink: {relative}")
        if path.is_file():
            inventory[relative.as_posix()] = hashes(path)
    require(bool(inventory), f"Empty local directory: {root}")
    return inventory


def verify_checksum_list(root: Path, inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require("SHA256SUMS" in inventory, f"SHA256SUMS missing in {root}")
    declared: dict[str, str] = {}
    for line_number, line in enumerate((root / "SHA256SUMS").read_text().splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"Malformed SHA256SUMS line {line_number}")
        digest, name = match.groups()
        if name.startswith("./"):
            name = name[2:]
        p = PurePosixPath(name)
        require(not p.is_absolute() and ".." not in p.parts and str(p) == name,
                f"Unsafe checksum path: {name}")
        require(name not in declared, f"Duplicate checksum path: {name}")
        require(name in inventory, f"Checksum file missing locally: {name}")
        require(inventory[name]["sha256"] == digest.lower(), f"Local checksum mismatch: {name}")
        declared[name] = digest.lower()
    mandatory = set(inventory) - {"SHA256SUMS", ".gitattributes"}
    require(mandatory <= set(declared), f"Files omitted from SHA256SUMS: {sorted(mandatory - set(declared))}")
    return {"path": "SHA256SUMS", "sha256": inventory["SHA256SUMS"]["sha256"],
            "verified_entry_count": len(declared)}


def get_json(client: httpx.Client, url: str) -> Any:
    response = client.get(url, headers={"Accept": "application/vnd.github+json"})
    require(response.status_code == 200, f"Anonymous HTTP {response.status_code}: {url}")
    return response.json()


def download_hash(client: httpx.Client, url: str, expected: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    actual_size = 0
    with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
        require(response.status_code == 200, f"Anonymous download HTTP {response.status_code}: {url}")
        for block in response.iter_bytes(1024 * 1024):
            actual_size += len(block)
            require(actual_size <= expected["size"], f"Download larger than local artifact: {url}")
            digest.update(block)
    require(actual_size == expected["size"], f"Downloaded size mismatch: {url}")
    actual_sha = digest.hexdigest()
    require(actual_sha == expected["sha256"], f"Downloaded SHA256 mismatch: {url}")
    return {"url": url, "size": actual_size, "sha256": actual_sha,
            "anonymous": True, "fresh_stream_download": True, "passed": True}


def lfs_field(lfs: Any, key: str) -> Any:
    return lfs.get(key) if isinstance(lfs, dict) else getattr(lfs, key, None)


def select_path(inventory: dict[str, Any], supplied: str | None, kind: str) -> str:
    if supplied:
        require(supplied in inventory, f"Requested {kind} spot-check file missing: {supplied}")
        return supplied
    if kind == "adapter":
        matches = [name for name in inventory if Path(name).name == "adapter_fp16.pt"]
    else:
        matches = [name for name in inventory if Path(name).name == "manifest.json"
                   and ("w4" in name.lower() or "base" in name.lower())]
    require(len(matches) == 1, f"Need explicit --{kind}-path; found {kind} candidates: {matches}")
    return matches[0]


def verify_hf(args: argparse.Namespace, client: httpx.Client, receipt: dict[str, Any]) -> None:
    root = args.hf_local_dir.resolve()
    local = local_inventory(root)
    section: dict[str, Any] = {"repo_id": args.hf_repo, "anonymous": True,
                               "local_sha256sums": verify_checksum_list(root, local)}
    receipt["hugging_face"] = section
    for required in ("README.md", "RELEASE_MANIFEST.json"):
        require(required in local, f"HF release bundle missing {required}")
    api = HfApi(endpoint="https://huggingface.co", token=False)
    info = api.model_info(args.hf_repo, revision=args.hf_revision, files_metadata=True, token=False)
    require(info.private is False, "HF repository is not public")
    require(info.gated is False, f"HF repository must be ungated; got {info.gated!r}")
    require(bool(info.sha), "HF revision SHA missing")
    revision = info.sha
    section.update({"revision": revision, "requested_revision": args.hf_revision,
                    "public": True, "gated": False,
                    "url": f"https://huggingface.co/{args.hf_repo}", "files": []})
    remote = {item.path: item for item in api.list_repo_tree(
        args.hf_repo, recursive=True, revision=revision, token=False) if isinstance(item, RepoFile)}
    allowed_extra = {".gitattributes"} - set(local)
    require(set(local) <= set(remote), f"HF files missing: {sorted(set(local) - set(remote))}")
    require(not (set(remote) - set(local) - allowed_extra),
            f"Unexpected HF files: {sorted(set(remote) - set(local) - allowed_extra)}")
    for name, expected in local.items():
        item = remote[name]
        require(item.size == expected["size"], f"HF file size mismatch: {name}")
        if item.lfs is not None:
            sha = lfs_field(item.lfs, "sha256")
            size = lfs_field(item.lfs, "size")
            require(sha == expected["sha256"], f"HF LFS SHA256 mismatch: {name}")
            require(size == expected["size"], f"HF LFS size mismatch: {name}")
            verification = {"method": "server_lfs_sha256", "server_sha256": sha}
        else:
            require(item.blob_id == expected["git_blob_sha1"], f"HF Git blob SHA1 mismatch: {name}")
            verification = {"method": "server_git_blob_sha1", "server_git_blob_sha1": item.blob_id}
        section["files"].append({"path": name, **expected, **verification, "passed": True})
    section.update({"file_count": len(local), "remote_file_count": len(remote),
                    "verified_bytes": sum(item["size"] for item in local.values()),
                    "ignored_server_defaults": sorted(set(remote) - set(local)), "download_checks": []})
    spots = [select_path(local, args.base_manifest_path, "base-manifest"),
             select_path(local, args.adapter_path, "adapter"), "README.md", "RELEASE_MANIFEST.json", "SHA256SUMS"]
    for name in dict.fromkeys(spots):
        require(local[name]["size"] <= 32 * 1024 * 1024, f"Spot-check unexpectedly large: {name}")
        url = hf_hub_url(args.hf_repo, name, revision=revision)
        section["download_checks"].append({"path": name, **download_hash(client, url, local[name])})
    section["passed"] = True
    print(f"HF verified: {len(local)} files, revision {revision}", flush=True)


def verify_github(args: argparse.Namespace, client: httpx.Client, receipt: dict[str, Any]) -> None:
    root = args.gh_assets_dir.resolve()
    local = local_inventory(root)
    require(all("/" not in name for name in local), "GitHub release assets directory must be flat")
    section: dict[str, Any] = {"repo_id": args.gh_repo, "anonymous": True,
                               "local_sha256sums": verify_checksum_list(root, local)}
    receipt["github"] = section
    api = f"https://api.github.com/repos/{args.gh_repo}"
    repo_info = get_json(client, api)
    require(repo_info.get("private") is False and repo_info.get("visibility") == "public",
            "GitHub repository is not public")
    release = get_json(client, f"{api}/releases/tags/{quote(args.tag, safe='')}")
    require(release.get("draft") is False, "GitHub release is still a draft")
    require(release.get("tag_name") == args.tag, "GitHub release tag mismatch")
    require(bool(release.get("published_at")), "GitHub release publication time missing")
    ref = get_json(client, f"{api}/git/ref/tags/{quote(args.tag, safe='')}")
    obj = ref["object"]
    for _ in range(5):
        if obj["type"] == "commit":
            break
        require(obj["type"] == "tag", "GitHub tag points to unsupported object")
        tag_object = get_json(client, f"{api}/git/tags/{obj['sha']}")
        obj = tag_object["object"]
    require(obj["type"] == "commit", "GitHub tag recursion did not resolve to a commit")
    if args.expected_git_commit:
        require(obj["sha"] == args.expected_git_commit, "GitHub release tag commit mismatch")
    section.update({"public": True, "release_id": release["id"], "tag": args.tag,
                    "url": release["html_url"], "draft": False,
                    "prerelease": release.get("prerelease"), "published_at": release["published_at"],
                    "target_commitish": release.get("target_commitish"), "resolved_tag_commit": obj["sha"],
                    "assets": [], "download_checks": []})
    assets = []
    page = 1
    while True:
        batch = get_json(client, f"{api}/releases/{release['id']}/assets?per_page=100&page={page}")
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    remote = {item["name"]: item for item in assets}
    require(len(remote) == len(assets), "Duplicate GitHub release asset names")
    require(set(remote) == set(local),
            f"GitHub release assets mismatch: missing={sorted(set(local)-set(remote))}, extra={sorted(set(remote)-set(local))}")
    for name, expected in local.items():
        item = remote[name]
        require(item.get("state") == "uploaded", f"GitHub asset not fully uploaded: {name}")
        require(item["size"] == expected["size"], f"GitHub asset size mismatch: {name}")
        server_digest = item.get("digest")
        url = item["browser_download_url"]
        if server_digest is not None:
            require(server_digest == f"sha256:{expected['sha256']}", f"GitHub asset SHA256 mismatch: {name}")
            method = "server_sha256"
        else:
            print(f"GitHub has no digest for {name}; verifying full anonymous download", flush=True)
            section["download_checks"].append({"path": name, **download_hash(client, url, expected)})
            method = "full_anonymous_download_sha256"
        section["assets"].append({"name": name, "asset_id": item["id"], **expected,
                                  "server_digest": server_digest, "verification_method": method,
                                  "download_url": url, "passed": True})
    # Also demonstrate fresh anonymous artifact retrieval on GitHub even when all
    # large assets have server digests. Check the checksum list and manifest.
    for name in ("SHA256SUMS", "RELEASE_MANIFEST.json"):
        if name in remote and not any(row["path"] == name for row in section["download_checks"]):
            section["download_checks"].append({"path": name, **download_hash(
                client, remote[name]["browser_download_url"], local[name])})
    section.update({"asset_count": len(assets), "verified_bytes": sum(item["size"] for item in local.values()),
                    "passed": True})
    print(f"GitHub verified: {len(assets)} assets, release {release['id']}, tag {args.tag}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-local-dir", type=Path, required=True)
    parser.add_argument("--gh-assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-repo", default="EndlessChasing/Mamb2_8B_W4A16_Recall")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--gh-repo", default="EndlessChasing/mamb2_8B_W4A16_Recall")
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--base-manifest-path", help="Path relative to HF bundle")
    parser.add_argument("--adapter-path", help="Path relative to HF bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Reserve the destination before expensive hashing/network operations.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.output.open("x")
    except FileExistsError:
        print(f"Refusing to overwrite receipt: {args.output}", file=sys.stderr)
        return 2
    receipt: dict[str, Any] = {
        "schema": "mamba2-w4a16-publication-verification-v1", "complete": False, "passed": False,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "anonymous remote metadata plus fresh small-file downloads; no remote mutations",
        "limitations": ["Large HF LFS files and GitHub assets with server SHA256 are not fully downloaded again.",
                        "This verifies publication integrity and access, not a new model quality evaluation."],
    }
    code = 1
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=20),
                          headers={"User-Agent": "mamba2-w4a16-publication-verifier/1.0"}) as client:
            verify_hf(args, client, receipt)
            verify_github(args, client, receipt)
        receipt.update({"complete": True, "passed": True})
        code = 0
    except VerificationError as exc:
        receipt["error"] = str(exc)
        print(f"Verification failed: {exc}", file=sys.stderr)
    except Exception as exc:
        # Third-party exceptions can contain signed redirect URLs. Avoid recording
        # credentials from those URLs; the operation and exception class suffice.
        receipt["error"] = f"Verification interrupted by {type(exc).__name__}; retry into a new receipt."
        print(receipt["error"], file=sys.stderr)
    finally:
        receipt["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    print(f"Receipt: {args.output}; passed={receipt['passed']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
