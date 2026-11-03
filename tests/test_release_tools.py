import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import download_models as assets
from vidueu_config import load_config, redact


def test_config_substitutes_scalars_without_changing_yaml_structure(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('models:\n  qwen_llm:\n    api:\n      api_key: ${TEST_KEY}\nvalue: $TEST_VALUE\n')
    cfg = load_config(path, environ={"TEST_KEY": "x: y # z", "TEST_VALUE": "false"})
    assert cfg["models"]["qwen_llm"]["api"]["api_key"] == "x: y # z"
    assert cfg["value"] == "false"
    assert "x: y" not in json.dumps(redact(cfg))


@pytest.mark.parametrize("text,pattern", [
    ('x: ${MISSING}\n', 'unexpanded_environment_variables:MISSING'),
    ('x: 1\nx: 2\n', 'duplicate_config_key'),
    ('- x\n', 'config_must_be_object'),
    ('models: []\n', 'config_section_must_be_object:models'),
])
def test_config_fails_early(tmp_path, text, pattern):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=pattern):
        load_config(path, environ={})


def test_pinned_manifest_and_dry_run_do_not_create_directories(tmp_path, capsys):
    manifest = assets.ROOT / "manifests/models.yaml"
    items = assets.load_manifest(manifest)
    assert len(items) == 13
    root = tmp_path / "new-models"
    assert assets.main(["--manifest", str(manifest), "--model-root", str(root), "--only", "yolo", "--dry-run"]) == 0
    assert not root.exists()
    assert "yolov8m.pt" in capsys.readouterr().out


@pytest.mark.parametrize("relative", ["../outside", "/outside", "a/../../b", "a\\b", "C:/outside"])
def test_asset_boundary_rejects_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        assets.safe_path(tmp_path, relative)


def test_download_receipt_detects_tamper_and_refuses_unmanaged_files(tmp_path):
    source = tmp_path / "official.fixture"
    source.write_bytes(b"pinned fixture bytes")
    item = {"id": "fixture", "kind": "url", "url": source.as_uri(),
            "sha256": assets.sha256(source), "target": "fixture", "filename": "weight.pt", "required": ["weight.pt"]}
    root = tmp_path / "models"
    root.mkdir()
    assets.download(root, item)
    assert assets.verify(root, item)["complete"]
    target = root / "fixture/weight.pt"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        assets.download(root, item)
    assert target.read_bytes() == b"changed"
    item2 = {**item, "id": "unknown", "target": "unknown"}
    (root / "unknown").mkdir()
    (root / "unknown/private.txt").write_text("untouched")
    with pytest.raises(ValueError, match="unmanaged"):
        assets.download(root, item2)


def test_asset_verifier_rejects_missing_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"tensor": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="missing_weight_shard"):
        assets.validate_files(tmp_path, {"required": ["config.json"]})


@pytest.mark.skipif(os.name != "posix", reason="Linux run manager")
def test_asset_boundary_rejects_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assets.safe_path(tmp_path, "link/file")


@pytest.mark.skipif(os.name != "posix", reason="Linux run manager")
@pytest.mark.parametrize("tmux", [False, True])
def test_managed_run_stops_only_owned_process_and_preserves_logs(tmp_path, monkeypatch, tmux):
    import shutil
    from scripts import manage_run as manager
    if tmux and not shutil.which("tmux"):
        pytest.skip("tmux not installed")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data.jsonl").write_text('preserved\n')
    code = "import os,time,json;print(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','http_proxy','HTTPS_PROXY']}),flush=True);time.sleep(90)"
    monkeypatch.setenv("http_proxy", "http://invalid.example:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid.example:2")
    monkeypatch.setattr(manager, "build_command", lambda a: ([sys.executable, "-u", "-c", code], {"CUDA_VISIBLE_DEVICES": "4,5"}, {}))
    a = argparse.Namespace(run_dir=run_dir, kind="pipeline", tmux=tmux, dry_run=False)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(90)"], start_new_session=True)
    try:
        assert manager.start(a) == 0
        deadline = time.monotonic() + 5
        while not manager.read_state(run_dir, "pipeline").get("child") and time.monotonic() < deadline:
            time.sleep(0.05)
        state = manager.read_state(run_dir, "pipeline")
        assert manager.same_process(state["child"])
        with pytest.raises(ValueError, match="already_active"):
            manager.start(a)
        assert manager.stop(run_dir, "pipeline") == 0
        assert unrelated.poll() is None
        assert not manager.same_process(state["child"])
        assert manager.status(run_dir, "pipeline")["status"] == "stopped"
        assert (run_dir / "data.jsonl").read_text() == 'preserved\n'
        log = Path(state["log"]).read_text()
        env = json.loads(log.splitlines()[0])
        assert env == {"CUDA_VISIBLE_DEVICES": "4,5", "http_proxy": None, "HTTPS_PROXY": None}
        # An explicit second start uses a new log and leaves previous artifacts.
        assert manager.start(a) == 0
        assert manager.stop(run_dir, "pipeline") == 0
        assert Path(state["log"]).read_text() == log
    finally:
        manager.stop(run_dir, "pipeline")
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="Linux process identities")
def test_stale_pid_cannot_stop_unrelated_process(tmp_path):
    from scripts import manage_run as manager
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(90)"], start_new_session=True)
    try:
        fake = manager.identity(child.pid)
        fake["start_ticks"] = "0"
        manager.write_json(manager.state_path(tmp_path, "pipeline"), {"worker": fake, "status": "running"})
        manager.stop(tmp_path, "pipeline")
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_omniparser_child_uses_parent_logical_gpu_map(tmp_path, monkeypatch):
    from executor.tasks.omniparser import OmniParserExecutor
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    seen = []
    def run(cmd, **kwargs):
        seen.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        payload = json.loads(Path(cmd[cmd.index("--input-json") + 1]).read_text())
        assert payload["model_config"]["device"] == "cuda:0"
        Path(cmd[cmd.index("--output-json") + 1]).write_text("{}")
        return argparse.Namespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    executor = OmniParserExecutor({"models": {"omniparser": {"repo_root": str(tmp_path), "device": "cuda:1"}}})
    assert executor._run_subprocess([]) == {}
    assert seen == ["7"]
    executor.config["models"]["omniparser"]["device"] = "cuda:2"
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        executor._run_subprocess([])


def test_doctor_help_and_bad_config_never_import_torch_or_call_provider(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    help_result = subprocess.run([sys.executable, str(assets.ROOT / "scripts/doctor.py"), "--help"], env=env, capture_output=True, text=True)
    assert help_result.returncode == 0
    config = tmp_path / "bad.yaml"
    config.write_text("models:\n  token: ${VIDUEU_TEST_UNSET_VARIABLE}\n")
    result = subprocess.run([sys.executable, str(assets.ROOT / "scripts/doctor.py"), "--config", str(config)], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "VIDUEU_TEST_UNSET_VARIABLE" in result.stdout


def test_setup_offline_dry_run_is_side_effect_free(tmp_path):
    prefix = tmp_path / "environments"
    result = subprocess.run([sys.executable, str(assets.ROOT / "scripts/setup_env.py"),
                             "--profile", "dev", "--env-prefix", str(prefix),
                             "--wheelhouse", str(tmp_path / "wheels"), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    installs = [command for command in plan["commands"] if "install" in command]
    assert installs and all("--no-index" in command for command in installs)
    assert not prefix.exists()


def test_setup_refuses_existing_environment(tmp_path):
    (tmp_path / "pipeline").mkdir()
    marker = tmp_path / "pipeline/keep.txt"
    marker.write_text("untouched")
    result = subprocess.run([sys.executable, str(assets.ROOT / "scripts/setup_env.py"),
                             "--env-prefix", str(tmp_path), "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "existing environment" in result.stderr
    assert marker.read_text() == "untouched"


@pytest.mark.skipif(os.name != "posix", reason="Linux process identity")
def test_orphan_worker_never_causes_duplicate_launch(tmp_path, monkeypatch):
    import signal
    from scripts import manage_run as manager
    run_dir = tmp_path / "orphan"
    command = [sys.executable, "-c", "import time;time.sleep(90)"]
    monkeypatch.setattr(manager, "build_command", lambda a: (command, {"CUDA_VISIBLE_DEVICES": ""}, {}))
    a = argparse.Namespace(run_dir=run_dir, kind="pipeline", tmux=False, dry_run=False)
    try:
        manager.start(a)
        deadline = time.monotonic() + 5
        while not manager.read_state(run_dir, "pipeline").get("child") and time.monotonic() < deadline:
            time.sleep(0.05)
        state = manager.read_state(run_dir, "pipeline")
        os.kill(state["worker"]["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while manager.same_process(state["worker"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert manager.owned_child(state)
        with pytest.raises(ValueError, match="orphan_child_still_active"):
            manager.start(a)
        manager.stop(run_dir, "pipeline")
        assert not manager.same_process(state["child"])
    finally:
        manager.stop(run_dir, "pipeline")
