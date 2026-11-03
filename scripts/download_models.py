#!/usr/bin/env python3
"""Download pinned assets, verify bytes, and refuse unmanaged or changed files."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
def download_timeout(explicit=None):
    value = explicit if explicit is not None else os.environ.get("VIDUEU_DOWNLOAD_TIMEOUT", "600")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("download_timeout_must_be_integer") from exc
    if value <= 0:
        raise ValueError("download_timeout_must_be_positive")
    return value


def safe_path(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or any(x in {"..", "."} for x in part.parts) or "\\" in relative or ":" in relative:
        raise ValueError(f"unsafe_asset_path:{relative}")
    path = root / relative
    if not path.absolute().is_relative_to(root.absolute()):
        raise ValueError("asset_path_escape")
    for p in (root, path, *path.parents):
        if p.is_symlink():
            raise ValueError(f"asset_symlink:{p}")
    if path.resolve() != path.absolute():
        raise ValueError(f"asset_path_not_canonical:{path}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    items = yaml.safe_load(Path(path).read_text())["resources"]
    seen = set()
    for item in items:
        if item["id"] in seen or not item["id"].replace("_", "").isalnum():
            raise ValueError("invalid_or_duplicate_asset_id")
        seen.add(item["id"])
        safe_path(Path.cwd(), item["target"])
        for relative in item["required"]:
            safe_path(Path.cwd(), relative)
        if item["kind"] in {"hf", "git"} and not (len(item["revision"]) == 40 and all(c in "0123456789abcdef" for c in item["revision"])):
            raise ValueError("asset_requires_pinned_revision")
        if item["kind"] in {"url", "zip"} and len(item.get("sha256", "")) != 64:
            raise ValueError("asset_requires_sha256")
    return items


def signature(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()


def validate_files(target, item):
    for relative in item["required"]:
        path = safe_path(target, relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing_asset_file:{path}")
    # An index alone is not a weight: require every referenced shard.
    for p in target.rglob("*.safetensors.index.json"):
        mapping = json.loads(p.read_text())["weight_map"]
        for relative in set(mapping.values()):
            shard = safe_path(p.parent, relative)
            if not shard.is_file() or not shard.stat().st_size:
                raise ValueError(f"missing_weight_shard:{shard}")


def verify(root, item):
    target = safe_path(root, item["target"])
    receipt = safe_path(root, f".vidueu-assets/{item['id']}.json")
    state = json.loads(receipt.read_text())
    if state.get("signature") != signature(item) or not state.get("complete"):
        raise ValueError(f"asset_not_verified:{item['id']}")
    if not state.get("files"):
        raise ValueError("asset_receipt_has_no_file_hashes")
    validate_files(target, item)
    for relative, expected in state["files"].items():
        if sha256(safe_path(target, relative)) != expected:
            raise ValueError(f"asset_hash_mismatch:{item['id']}:{relative}")
    return state


def download_ssl_context():
    """Retain TLS verification when a custom Python has no default CA store."""
    context = ssl.create_default_context()
    explicit = os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR")
    if not explicit and context.cert_store_stats()["x509_ca"] == 0:
        try:
            import certifi
        except ImportError as exc:
            raise ValueError("missing_ca_store:install_certifi_or_set_SSL_CERT_FILE") from exc
        context.load_verify_locations(cafile=certifi.where())
    return context


def urlopen_with_308(request, *, timeout, context):
    """Follow mirror-specific 308 redirects while retaining a Range header."""
    current = request
    for _ in range(5):
        try:
            return urllib.request.urlopen(current, timeout=timeout, context=context)
        except urllib.error.HTTPError as exc:
            if exc.code != 308 or not exc.headers.get("Location"):
                raise
            headers = dict(current.header_items()) if isinstance(current, urllib.request.Request) else {}
            origin = current.full_url if isinstance(current, urllib.request.Request) else current
            current = urllib.request.Request(urljoin(origin, exc.headers["Location"]), headers=headers)
    raise ValueError("download_redirect_limit_exceeded")


RETRY_ATTEMPTS = 5


def retry_delay(exc, attempt):
    response = getattr(exc, "response", None)
    headers = getattr(exc, "headers", None) or getattr(response, "headers", {})
    value = headers.get("Retry-After", "")
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            delay = 2 ** attempt
    return max(0, min(delay, 60))


def transient_error(exc):
    response = getattr(exc, "response", None)
    code = getattr(exc, "code", None) or getattr(response, "status_code", None)
    if code is not None:
        return code in {408, 429} or 500 <= code < 600
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError,
                            http.client.IncompleteRead))


def retry_metadata(operation):
    # Hub metadata uses requests, while large file transfers use urllib.
    import requests
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return operation()
        except (requests.RequestException, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            transient = transient_error(exc) or (
                getattr(exc, "response", None) is None
                and isinstance(exc, (requests.Timeout, requests.ConnectionError)))
            if not transient or attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(retry_delay(exc, attempt))


def quarantine_partial(partial):
    # Preserve corrupt bytes for diagnosis; never delete or overwrite an asset.
    for number in range(1, 10000):
        target = partial.with_name(partial.name + f".corrupt-{number}")
        if target.exists() or target.is_symlink():
            continue
        partial.rename(target)
        print(json.dumps({"status": "partial_quarantined", "path": str(target)}), flush=True)
        return
    raise ValueError("download_partial_quarantine_names_exhausted")


def fetch_url(url, destination, expected, *, expected_size=None, timeout=None, digest=sha256):
    timeout = download_timeout(timeout)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"refuse_existing_mismatched_asset:{destination}")
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"refuse_existing_mismatched_asset_size:{destination}")
        return
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_symlink():
        raise ValueError("download_partial_is_symlink")
    # Resume verified byte ranges. A server ignoring Range must restart, not append.
    if partial.exists() and digest(partial) == expected and (expected_size is None or partial.stat().st_size == expected_size):
        partial.replace(destination)
        return
    if expected_size is not None and partial.exists() and partial.stat().st_size >= expected_size:
        quarantine_partial(partial)
    for attempt in range(RETRY_ATTEMPTS):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"}) if offset else url
        try:
            with urlopen_with_308(request, timeout=timeout, context=download_ssl_context()) as response:
                status = getattr(response, "status", None)
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("download_invalid_content_range")
                    if expected_size is not None and not content_range.endswith(f"/{expected_size}"):
                        raise ValueError("download_invalid_content_range_size")
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output)
            if expected_size is not None and partial.stat().st_size != expected_size:
                if partial.stat().st_size > expected_size:
                    quarantine_partial(partial)
                if attempt < RETRY_ATTEMPTS - 1:
                    continue
                raise ValueError(f"download_size_mismatch:{url}")
            if digest(partial) != expected:
                quarantine_partial(partial)
                if attempt < RETRY_ATTEMPTS - 1:
                    continue
                raise ValueError(f"download_hash_mismatch:{url}")
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            # A stale complete partial without known size can receive HTTP 416.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and partial.exists():
                quarantine_partial(partial)
            elif not transient_error(exc):
                raise
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(retry_delay(exc, attempt))
    raise ValueError(f"download_failed:{url}")


def download(root, item, endpoint=None, *, timeout=None):
    target = safe_path(root, item["target"])
    receipt = safe_path(root, f".vidueu-assets/{item['id']}.json")
    stamp = signature(item)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("signature") != stamp:
            raise ValueError("refuse_changed_asset_definition")
        if previous.get("complete"):
            verify(root, item)
            return
    else:
        # RAFT weights may share the managed RAFT code directory.
        shared = item["kind"] == "zip" and (root / ".vidueu-assets/raft_code.json").is_file()
        if target.exists() and any(target.iterdir()) and not shared:
            raise ValueError(f"refuse_unmanaged_existing_directory:{target}")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"signature": stamp, "complete": False}))
    target.parent.mkdir(parents=True, exist_ok=True)
    kind = item["kind"]
    files = []
    if kind == "hf":
        from huggingface_hub import HfApi, hf_hub_url
        api = HfApi(endpoint=endpoint)
        info = retry_metadata(lambda: api.model_info(item["repo_id"], revision=item["revision"],
                                                    files_metadata=True, timeout=download_timeout(timeout)))
        expected_files = [s for s in info.siblings if any(fnmatch.fnmatchcase(s.rfilename, pat) for pat in item["patterns"])]
        for entry in expected_files:
            existing = safe_path(target, entry.rfilename)
            if existing.exists():
                if entry.lfs:
                    matches = sha256(existing) == entry.lfs.sha256
                else:
                    data = existing.read_bytes()
                    matches = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry.blob_id
                if not matches:
                    raise ValueError(f"refuse_existing_mismatched_asset:{existing}")
        for entry in expected_files:
            path = safe_path(target, entry.rfilename)
            path.parent.mkdir(parents=True, exist_ok=True)
            fetch_url(hf_hub_url(item["repo_id"], entry.rfilename, revision=item["revision"], endpoint=endpoint),
                      path,
                      entry.lfs.sha256 if entry.lfs else entry.blob_id,
                      expected_size=entry.size, timeout=timeout,
                      digest=sha256 if entry.lfs else git_blob_sha1)
            if entry.size is not None and path.stat().st_size != entry.size:
                raise ValueError(f"hub_size_mismatch:{entry.rfilename}")
            lfs = entry.lfs
            if lfs:
                if sha256(path) != lfs.sha256:
                    raise ValueError(f"hub_sha256_mismatch:{entry.rfilename}")
            else:
                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if blob != entry.blob_id:
                    raise ValueError(f"hub_blob_mismatch:{entry.rfilename}")
            files.append(path)
    elif kind == "git":
        created = not target.exists()
        if not target.exists():
            subprocess.run(["git", "clone", "--no-checkout", item["url"], str(target)], check=True)
        if subprocess.check_output(["git", "-C", str(target), "config", "--get", "remote.origin.url"], text=True).strip() != item["url"]:
            raise ValueError("code_origin_mismatch")
        status = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if status and not created:
            raise ValueError("refuse_dirty_code_checkout")
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["revision"]], check=True)
        names = subprocess.check_output(["git", "-C", str(target), "ls-files", "-z"]).decode().split("\0")
        files = [safe_path(target, name) for name in names if name and (target / name).is_file()]
    elif kind in {"url", "zip"}:
        target.mkdir(parents=True, exist_ok=True)
        location = safe_path(target, item["filename"])
        fetch_url(item["url"], location, item["sha256"], timeout=timeout)
        files = [location]
        if kind == "zip":
            with zipfile.ZipFile(location) as archive:
                for name in item["members"]:
                    info = archive.getinfo(name)
                    destination = safe_path(target, name)
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("archive_symlink")
                    data = archive.read(info)
                    if destination.exists() and destination.read_bytes() != data:
                        raise ValueError(f"refuse_overwrite:{destination}")
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as output:
                            output.write(data)
                    files.append(destination)
    else:
        raise ValueError("unknown_asset_kind")
    validate_files(target, item)
    state = {"signature": stamp, "complete": True, "source": item,
             "files": {p.relative_to(target).as_posix(): sha256(p) for p in files}}
    receipt.write_text(json.dumps(state, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "manifests/models.yaml")
    p.add_argument("--model-root", type=Path, default=ROOT / "model")
    p.add_argument("--only", nargs="+", help="Explicit asset ids (no implicit large downloads)")
    p.add_argument("--endpoint", help="Explicit Hugging Face endpoint/mirror")
    p.add_argument("--download-timeout", type=int, help="Per-request timeout in seconds; default 600")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true", help="Hash-check receipts locally; no network")
    p.add_argument("--offline", action="store_true", help="Equivalent to --verify")
    a = p.parse_args(argv)
    items = load_manifest(a.manifest)
    known = {i["id"] for i in items}
    if a.only and not set(a.only) <= known:
        p.error("unknown_asset_id")
    if not a.list and not a.only:
        p.error("Specify --only; use --list to inspect available assets")
    selected = [i for i in items if not a.only or i["id"] in a.only]
    root = a.model_root.expanduser().absolute()
    for item in selected:
        target = safe_path(root, item["target"])
        if a.list or a.dry_run:
            print(json.dumps({**item, "destination": str(target)}))
        elif a.verify or a.offline:
            verify(root, item)
            print(json.dumps({"id": item["id"], "status": "verified"}))
        else:
            root.mkdir(parents=True, exist_ok=True)
            download(root, item, a.endpoint, timeout=a.download_timeout)
            print(json.dumps({"id": item["id"], "status": "verified"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Asset preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
