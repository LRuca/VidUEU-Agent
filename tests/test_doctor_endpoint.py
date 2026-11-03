"""Probe actual loopback HTTP endpoints without generation calls or credentials."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import pytest

from scripts.doctor import probe_endpoint


@pytest.fixture
def endpoint():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/protected/v1/models" and self.headers.get("Authorization") != "Bearer fixture-only":
                self.send_response(401)
                self.end_headers()
                return
            body = json.dumps({"data": [{"id": "local-qwen"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_vlm_without_key_is_probed_and_environment_proxy_is_bypassed(endpoint, monkeypatch):
    base, seen = endpoint
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    assert probe_endpoint({"base_url": base + "/v1", "model": "local-qwen"},
                          proxy_mode="direct") == {"model": "local-qwen", "http": 200}
    assert seen == [("/v1/models", None)]


def test_auth_required_is_decided_by_server(endpoint):
    base, seen = endpoint
    with pytest.raises(HTTPError) as error:
        probe_endpoint({"base_url": base + "/protected/v1", "model": "local-qwen"}, proxy_mode="direct")
    assert error.value.code == 401
    assert len(seen) == 1


def test_key_file_auth_matches_vision_client(endpoint, tmp_path):
    base, seen = endpoint
    key = tmp_path / "fixture.key"
    key.write_text("fixture-only\n")
    probe_endpoint({"base_url": base + "/protected/v1", "model": "local-qwen", "api_key_file": str(key)},
                   proxy_mode="direct")
    assert seen == [("/protected/v1/models", "Bearer fixture-only")]


def test_model_mismatch_is_still_rejected(endpoint):
    base, _ = endpoint
    with pytest.raises(ValueError, match="vlm_served_model_mismatch"):
        probe_endpoint({"base_url": base + "/v1", "model": "wrong-model"}, proxy_mode="direct")


def test_explicit_proxy_is_actually_used(endpoint):
    base, seen = endpoint
    probe_endpoint({"base_url": "http://vidueu-test.invalid/v1", "model": "local-qwen"},
                   proxy_mode="explicit", proxy_url=base)
    assert seen == [("http://vidueu-test.invalid/v1/models", None)]
