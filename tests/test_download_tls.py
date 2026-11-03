"""Keep certificate verification enabled on Python builds with no system CA file."""
import hashlib
import io
import ssl
import sys
from types import SimpleNamespace

import pytest

from scripts import download_models as assets


def empty_context(monkeypatch):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(assets.ssl, "create_default_context", lambda: context)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    return context


def test_empty_default_store_loads_trusted_certifi_roots(monkeypatch):
    context = empty_context(monkeypatch)
    import certifi
    assert assets.download_ssl_context() is context
    assert context.cert_store_stats()["x509_ca"] > 0
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_explicit_trust_store_is_not_replaced(monkeypatch):
    context = empty_context(monkeypatch)
    monkeypatch.setenv("SSL_CERT_FILE", "/explicit/trusted/roots.pem")
    assert assets.download_ssl_context() is context
    assert context.cert_store_stats()["x509_ca"] == 0


def test_populated_default_store_does_not_load_another_bundle(monkeypatch):
    import certifi
    context = empty_context(monkeypatch)
    context.load_verify_locations(cafile=certifi.where())
    monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: "/must-not-be-read"))
    assert assets.download_ssl_context() is context


def test_url_download_passes_verified_context_and_checks_hash(tmp_path, monkeypatch):
    context = empty_context(monkeypatch)
    monkeypatch.setattr(assets, "download_ssl_context", lambda: context)
    data = b"fixture model bytes"

    def response(url, *, timeout, context):
        assert timeout == 600
        assert url.get_header("User-agent") == "VidUEU-Downloader/0.1"
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        return io.BytesIO(data)

    monkeypatch.setattr(assets.urllib.request, "urlopen", response)
    destination = tmp_path / "model.pt"
    assets.fetch_url("https://official.example/model.pt", destination, hashlib.sha256(data).hexdigest(), expected_size=len(data))
    assert destination.read_bytes() == data
    assert not (tmp_path / "model.pt.partial").exists()


def test_partial_download_resumes_only_at_confirmed_range(tmp_path, monkeypatch):
    data = b"abcdefgh"
    destination = tmp_path / "model.pt"
    (tmp_path / "model.pt.partial").write_bytes(data[:3])

    def response(request, **kwargs):
        assert request.get_header("Range") == "bytes=3-"
        stream = io.BytesIO(data[3:])
        stream.status = 206
        stream.headers = {"Content-Range": "bytes 3-7/8"}
        return stream

    monkeypatch.setattr(assets.urllib.request, "urlopen", response)
    assets.fetch_url("https://official.example/model.pt", destination, hashlib.sha256(data).hexdigest(), expected_size=len(data))
    assert destination.read_bytes() == data


def test_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    data = b"abcdefgh"
    destination = tmp_path / "model.pt"
    (tmp_path / "model.pt.partial").write_bytes(data[:3])
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data))
    assets.fetch_url("https://official.example/model.pt", destination, hashlib.sha256(data).hexdigest(), expected_size=len(data))
    assert destination.read_bytes() == data


def test_truncated_response_is_rejected_before_publish(tmp_path, monkeypatch):
    destination = tmp_path / "large-model.safetensors"
    calls = []

    def response(request, **kwargs):
        calls.append(request.get_header("Range") if hasattr(request, "get_header") else None)
        stream = io.BytesIO(b"abc")
        return stream

    monkeypatch.setattr(assets.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="download_size_mismatch"):
        assets.fetch_url("https://official.example/large-model", destination,
                         hashlib.sha256(b"abcdefgh").hexdigest(), expected_size=8)
    assert calls == [None] + ["bytes=3-"] * (assets.RETRY_ATTEMPTS - 1)
    assert not destination.exists()


def test_git_blob_digest_matches_git_object_hash(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}\n")
    expected = hashlib.sha1(b"blob 3\0{}\n").hexdigest()
    assert assets.git_blob_sha1(path) == expected


def test_308_redirect_keeps_range_header(monkeypatch):
    calls = []
    response = io.BytesIO(b"ok")
    response.status = 200
    response.headers = {}

    def open_url(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise assets.urllib.error.HTTPError(
                request.full_url, 308, "redirect", {"Location": "https://cdn.example/file"}, None)
        assert request.full_url == "https://cdn.example/file"
        assert request.get_header("Range") == "bytes=3-"
        return response

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_url)
    request = assets.urllib.request.Request("https://mirror.example/file", headers={"Range": "bytes=3-"})
    assert assets.urlopen_with_308(request, timeout=600, context=ssl.create_default_context()) is response


def test_corrupt_complete_partial_is_quarantined_before_restart(tmp_path, monkeypatch):
    data = b"abcdefgh"
    destination = tmp_path / "model.pt"
    partial = tmp_path / "model.pt.partial"
    partial.write_bytes(b"XXXXXXXX")
    calls = []

    def response(request, **kwargs):
        calls.append(request.get_header("Range") if hasattr(request, "get_header") else None)
        return io.BytesIO(data)

    monkeypatch.setattr(assets.urllib.request, "urlopen", response)
    assets.fetch_url("https://official.example/model.pt", destination,
                     hashlib.sha256(data).hexdigest(), expected_size=len(data))
    assert destination.read_bytes() == data
    assert calls == [None]
    quarantined = list(tmp_path.glob("model.pt.partial.corrupt-*"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"XXXXXXXX"


def test_metadata_retries_transient_http_failure(monkeypatch):
    import requests
    response = requests.Response()
    response.status_code = 503
    error = requests.HTTPError("temporary", response=response)
    calls = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise error
        return {"siblings": []}

    monkeypatch.setattr(assets.time, "sleep", lambda delay: None)
    assert assets.retry_metadata(operation) == {"siblings": []}
    assert len(calls) == 3
