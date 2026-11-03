"""Deployment contracts without installing packages, network access, or GPUs."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts import deploy
from scripts.download_models import load_manifest
from vidueu_config import load_config


def options(tmp_path):
    return ["--env-prefix", str(tmp_path / "envs"), "--model-root", str(tmp_path / "models"),
            "--work-dir", str(tmp_path / "work"), "--profile", "pipeline"]


def test_dry_run_covers_all_assets_in_dependency_order_without_writes(tmp_path, capsys):
    assert deploy.main([*options(tmp_path), "--dry-run", "--endpoint", "https://mirror.example"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [v["profile"] for v in result["environments"]] == ["pipeline"]
    only_start = result["download"].index("--only") + 1
    only_end = result["download"].index("--endpoint")
    assert set(result["download"][only_start:only_end]) == set(deploy.PROFILE_ASSETS["pipeline"])
    assert deploy.PROFILE_ASSETS["pipeline"].index("raft_code") < deploy.PROFILE_ASSETS["pipeline"].index("raft_weights")
    assert "omni_code" not in result["download"]
    assert "--verify" in result["verify"]
    assert "https://mirror.example" in result["download"]
    assert len(result["runtime_checks"]) == 1
    assert all("--inference" in argv for argv in result["runtime_checks"])
    assert not list(tmp_path.iterdir())


def test_refuses_existing_unmanaged_environment_before_any_install(tmp_path, monkeypatch):
    mock_host(monkeypatch)
    (tmp_path / "envs/pipeline").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="not managed"):
        deploy.main(options(tmp_path))
    assert not calls
    assert (tmp_path / "envs/pipeline").is_dir()


def test_explicit_python_must_execute_with_matching_version(monkeypatch):
    monkeypatch.setattr(deploy.shutil, "which", lambda value: value)
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *a, **kw: "3.12")
    assert deploy.discover_python("3.12", "/custom/python") == "/custom/python"
    with pytest.raises(ValueError, match="Python 3.10 unavailable"):
        deploy.discover_python("3.10", "/custom/python")


def test_pending_managed_install_is_resumed_without_claiming_complete(tmp_path, capsys):
    target = tmp_path / "envs/pipeline"
    target.mkdir(parents=True)
    (target / ".vidueu-setup.json").write_text('{"recipe":"pending","completed":1}')
    assert deploy.main([*options(tmp_path), "--dry-run"]) == 0
    item = json.loads(capsys.readouterr().out)["environments"][0]
    assert "--resume" in item["argv"]
    assert not item["reuse"]


def test_low_space_fails_before_creating_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    args = SimpleNamespace(env_prefix=tmp_path / "envs", model_root=tmp_path / "models",
                           work_dir=tmp_path / "work", min_free_gib=20)
    with pytest.raises(ValueError, match="Insufficient free space"):
        deploy.prepare_workspace(args)
    assert not list(tmp_path.iterdir())


def mock_host(monkeypatch):
    monkeypatch.setattr(deploy, "prepare_workspace", lambda args: None)
    monkeypatch.setattr(deploy, "discover_python", lambda version, explicit: f"python{version}")
    monkeypatch.setattr(deploy, "discover_cuda", lambda: "/usr/local/cuda-12.4")
    monkeypatch.setattr(deploy.sys, "platform", "linux")
    monkeypatch.setattr(deploy.shutil, "which", lambda value: value)
    monkeypatch.setattr(deploy.subprocess, "check_output",
                        lambda argv, **kw: "3.12" if "3.12" in argv[0] else "3.10")


def test_installs_then_downloads_and_verifies_and_reuses_receipts(tmp_path, monkeypatch, capsys):
    mock_host(monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "setup_env.py" in argv[1]:
            profile = argv[argv.index("--profile") + 1]
            target = tmp_path / "envs" / profile / "bin"
            target.mkdir(parents=True, exist_ok=True)
            (target / "python").touch()

    monkeypatch.setattr(deploy.subprocess, "run", run)
    assert deploy.main(options(tmp_path)) == 0
    assert len(calls) == 4
    assert "download_models.py" in calls[1][1]
    assert "--verify" in calls[2]
    assert "--inference" in calls[-1]
    assert len(list((tmp_path / "envs").rglob(".vidueu-deploy.json"))) == 1
    calls.clear()
    assert deploy.main(options(tmp_path)) == 0
    assert "setup_env.py" in calls[0][1] and "--resume" in calls[0]
    assert "download_models.py" in calls[1][1]
    assert "--verify" in calls[2]
    receipt = tmp_path / "envs/pipeline/.vidueu-deploy.json"
    receipt.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="recipe changed"):
        deploy.main([*options(tmp_path), "--dry-run"])


def test_profile_signature_depends_on_packages_not_runtime_probe(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "requirements").mkdir(parents=True)
    (root / "scripts").mkdir()
    requirement = root / "requirements/pipeline.txt"
    requirement.write_text("paddlepaddle==3.2.2\n")
    probe = root / "scripts/check_runtime.py"
    probe.write_text("before")
    monkeypatch.setattr(deploy, "ROOT", root)
    before = deploy.profile_signature("pipeline")
    probe.write_text("fixed adapter validation")
    assert deploy.profile_signature("pipeline") == before
    requirement.write_text("paddlepaddle==3.3.0\n")
    assert deploy.profile_signature("pipeline") != before


def test_failed_install_never_publishes_receipt_or_downloads(tmp_path, monkeypatch):
    mock_host(monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(deploy.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        deploy.main(options(tmp_path))
    assert len(calls) == 1
    assert not list(tmp_path.rglob(".vidueu-deploy.json"))


def test_failed_real_adapter_check_prevents_deployment_success(tmp_path, monkeypatch, capsys):
    mock_host(monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "setup_env.py" in argv[1]:
            target = tmp_path / "envs" / argv[argv.index("--profile") + 1] / "bin"
            target.mkdir(parents=True)
            (target / "python").touch()
        if "check_runtime.py" in argv[1]:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(deploy.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        deploy.main(options(tmp_path))
    assert "Deployment complete" not in capsys.readouterr().out
    assert len(calls) == 4


def test_pipeline_example_copies_to_root_and_compiles(tmp_path):
    path = tmp_path / "pipeline.local.yaml"
    path.write_bytes((deploy.ROOT / "configs/pipeline.example.yaml").read_bytes())
    cfg = load_config(path, environ={"DEEPSEEK_API_KEY": "fixture-only"})
    assert Path(cfg["runtime"]["output_root"]) == tmp_path / "outputs/run-001"
    assert cfg["models"]["qwen_llm"]["api"]["model"] == "deepseek-flash"


def test_iteration_example_uses_real_backends_and_complete_command_fields():
    cfg = yaml.safe_load((deploy.ROOT / "configs/iteration.example.yaml").read_text())
    assert cfg["loop"]["trainer_backend"] == "bagel"
    assert cfg["loop"]["evaluator_backend"] == "benchmark"
    assert cfg["loop"]["actual_sample_cap"] == 0
    assert set(cfg["benchmark"]["commands"]) == {"MMDU", "GEdit-Bench-En", "WEAVEbench"}
    for spec in cfg["benchmark"]["commands"].values():
        assert all(spec[key] for key in ("argv", "subset", "case_ids", "source_files", "media_files", "judge", "prompt_version", "decoding"))


def test_default_installs_every_tool_in_default_example(tmp_path, capsys):
    assert deploy.main(["--env-prefix", str(tmp_path / "envs"), "--model-root", str(tmp_path / "models"),
                        "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert {v["profile"] for v in plan["environments"]} == {"pipeline", "omniparser"}
    assert {"omni_code", "omni_weights", "florence"} <= set(plan["download"])
    assert not list(tmp_path.iterdir())


def test_all_profiles_include_training_and_local_vision_assets_without_writes(tmp_path, capsys):
    deploy.main(["--profile", "all", "--env-prefix", str(tmp_path / "envs"),
                 "--model-root", str(tmp_path / "models"), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert {p["profile"] for p in plan["environments"]} == set(deploy.PROFILES)
    assert set(deploy.ASSETS) <= set(plan["download"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tools,profiles", [(["ocr"], ["pipeline"]),
                                          (["ocr", "omniparser"], ["pipeline", "omniparser"])])
def test_config_drives_dependencies_even_with_explicit_pipeline(tmp_path, capsys, tools, profiles):
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"version": 1, "tools": tools}))
    assert deploy.main([*options(tmp_path), "--config", str(config), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [v["profile"] for v in result["environments"]] == profiles
    assert "whisper" not in result["download"] and "yolo" not in result["download"]


def test_custom_paths_are_wired_without_keys_and_user_edits_are_preserved(tmp_path, monkeypatch):
    args = SimpleNamespace(config=None, launch_dir=tmp_path / "launch dir",
                           env_prefix=tmp_path / "custom envs", model_root=tmp_path / "custom models")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-must-not-be-copied")
    launcher = deploy.publish_artifacts(args, ["pipeline", "omniparser"])
    config = args.launch_dir / "pipeline.local.yaml"
    text = config.read_text()
    assert "secret-must-not-be-copied" not in text
    cfg = load_config(config)
    assert cfg["_vidueu"]["python"] == str(args.env_prefix / "pipeline/bin/python")
    assert cfg["models"]["omniparser"]["python_executable"] == str(args.env_prefix / "omniparser/bin/python")
    assert str(args.model_root / "RAFT") == cfg["optical_flow"]["raft"]["repo_root"]
    assert str(args.env_prefix / "pipeline/bin/python") in Path(launcher).read_text()
    deploy.publish_artifacts(args, ["pipeline", "omniparser"])
    config.write_text(text + "# user edit\n")
    with pytest.raises(ValueError, match="Refuse overwriting"):
        deploy.publish_artifacts(args, ["pipeline", "omniparser"])
    assert config.read_text().endswith("# user edit\n")
