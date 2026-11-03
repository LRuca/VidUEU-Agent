"""Interrupted installs may resume only with the exact owned recipe."""
import json
import hashlib
import subprocess

import pytest

from scripts import setup_env


def test_resume_skips_successful_steps_and_rejects_changed_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda value: value)
    monkeypatch.setattr(setup_env.subprocess, "check_output", lambda *a, **kw: "3.10")
    calls = []

    def fail_second(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(setup_env.subprocess, "run", fail_second)
    args = ["--profile", "dev", "--env-prefix", str(tmp_path), "--resume"]
    with pytest.raises(subprocess.CalledProcessError):
        setup_env.main(args)
    state = tmp_path / "dev/.vidueu-setup.json"
    assert json.loads(state.read_text())["completed"] == 1
    calls.clear()
    monkeypatch.setattr(setup_env.subprocess, "run", lambda argv, **kw: calls.append(argv))
    assert setup_env.main(args) == 0
    assert not any("venv" in argv for argv in calls)
    assert calls[-1][-3:] == ["-m", "pip", "check"]
    calls.clear()
    with pytest.raises(SystemExit):
        setup_env.main([*args, "--index-url", "https://different.example/simple"])
    assert not calls


def test_resume_refuses_unmanaged_directory(tmp_path):
    (tmp_path / "dev").mkdir()
    with pytest.raises(SystemExit):
        setup_env.main(["--profile", "dev", "--env-prefix", str(tmp_path), "--resume"])


def managed_plan(tmp_path, monkeypatch, capsys):
    root = tmp_path / "source"
    (root / "requirements").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "requirements/dev.txt").write_text("pytest==8.3.5\n")
    (root / "scripts/check_runtime.py").write_text("# runtime v1\n")
    monkeypatch.setattr(setup_env, "ROOT", root)
    monkeypatch.setattr(setup_env.shutil, "which", lambda value: value)
    monkeypatch.setattr(setup_env.subprocess, "check_output", lambda *a, **kw: "3.10")
    args = ["--profile", "dev", "--env-prefix", str(tmp_path / "envs"), "--resume"]
    assert setup_env.main([*args, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    return root, args, plan["commands"]


def test_verified_legacy_receipt_rechecks_without_pip_install(tmp_path, monkeypatch, capsys):
    root, args, commands = managed_plan(tmp_path, monkeypatch, capsys)
    legacy_commands = commands[:-2] + [commands[-1]]
    digest = hashlib.sha256(json.dumps(legacy_commands, sort_keys=True).encode())
    digest.update((root / "requirements/dev.txt").read_bytes())
    target = tmp_path / "envs/dev"
    target.mkdir(parents=True)
    receipt = target / ".vidueu-setup.json"
    original = json.dumps({"recipe": digest.hexdigest(), "completed": len(legacy_commands)})
    receipt.write_text(original)
    assert setup_env.main([*args, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["resume_reason"] == "verified_legacy_recipe"
    assert plan["pending_commands"] == commands[-2:]
    assert receipt.read_text() == original
    calls = []
    monkeypatch.setattr(setup_env.subprocess, "run", lambda argv, **kw: calls.append(argv))
    assert setup_env.main(args) == 0
    assert calls == commands[-2:]
    saved = json.loads(receipt.read_text())
    assert saved["install_recipe"] and saved["completed"] == len(commands)


def test_probe_change_and_repeat_install_only_rerun_validation(tmp_path, monkeypatch, capsys):
    root, args, commands = managed_plan(tmp_path, monkeypatch, capsys)
    calls = []
    monkeypatch.setattr(setup_env.subprocess, "run", lambda argv, **kw: calls.append(argv))
    assert setup_env.main(args) == 0
    capsys.readouterr()
    calls.clear()
    assert setup_env.main(args) == 0
    assert calls == commands[-2:]
    capsys.readouterr()
    (root / "scripts/check_runtime.py").write_text("# repaired runtime v2\n")
    calls.clear()
    assert setup_env.main([*args, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["resume_reason"] == "runtime_checks_updated"
    assert setup_env.main(args) == 0
    assert calls == commands[-2:]
    (root / "requirements/dev.txt").write_text("pytest==9.0.0\n")
    calls.clear()
    with pytest.raises(SystemExit):
        setup_env.main(args)
    assert calls == []


def test_failed_legacy_validation_preserves_original_receipt(tmp_path, monkeypatch, capsys):
    root, args, commands = managed_plan(tmp_path, monkeypatch, capsys)
    legacy_commands = commands[:-2] + [commands[-1]]
    digest = hashlib.sha256(json.dumps(legacy_commands, sort_keys=True).encode())
    digest.update((root / "requirements/dev.txt").read_bytes())
    target = tmp_path / "envs/dev"
    target.mkdir(parents=True)
    receipt = target / ".vidueu-setup.json"
    original = json.dumps({"recipe": digest.hexdigest(), "completed": len(legacy_commands)})
    receipt.write_text(original)

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(setup_env.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup_env.main(args)
    assert receipt.read_text() == original


def test_bagel_video_backend_is_optional_for_image_training(monkeypatch):
    root = setup_env.ROOT / "vendor/bagel"
    requirements = (root / "requirements.txt").read_text()
    assert not any(line.strip().startswith("decord") for line in requirements.splitlines())
    pytest.importorskip("numpy", reason="Runtime test needs BAGEL's NumPy dependency")
    import importlib.util
    import sys
    monkeypatch.setitem(sys.modules, "decord", None)
    spec = importlib.util.spec_from_file_location("bagel_video_optional", root / "data/video_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_frame_indices(2, 8, sample="middle") == [1, 5]
    for action in (lambda: module.read_frames_decord("unused.mp4", 8),
                   lambda: module.decode_video_byte(b""),
                   lambda: module.sample_mp4_frames("unused.mp4"),
                   lambda: module.sample_mp4_frames_by_indices("unused.mp4", [0])):
        with pytest.raises(ImportError, match="Encoded-video training requires"):
            action()


def test_bagel_parses_help_before_cuda_initialization():
    import ast
    source = (setup_env.ROOT / "vendor/bagel/train/pretrain_unified_navit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = [ast.unparse(node) for node in main.body]
    parser = next(i for i, text in enumerate(statements) if "parse_args_into_dataclasses" in text)
    cuda = next(i for i, text in enumerate(statements) if "torch.cuda.is_available" in text)
    assert parser < cuda
