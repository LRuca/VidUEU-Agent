import json
import os
import socket
from pathlib import Path

import pytest
import yaml
from PIL import Image

from vidueu.cli import main
from vidueu.config import template, required_assets
from vidueu.export import export_run, verify_package
from vidueu.smoke import offline_guards, run_smoke
from vidueu_config import load_config, redact


def small(tmp_path, **values):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**template(), **values}), encoding="utf-8")
    return path


def test_small_config_overrides_before_expansion_and_env_is_not_mutated(tmp_path, monkeypatch):
    path = small(tmp_path, input="${MISSING_INPUT}", output="${MISSING_OUTPUT}", tools=["ocr"])
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=fixture-secret\nLABEL=one\nALIAS=${LABEL}\n")
    monkeypatch.setenv("LABEL", "process-wins")
    before = dict(os.environ)
    cfg = load_config(path, overrides={"input": str(tmp_path / "videos"), "output": str(tmp_path / "out")})
    assert cfg["runtime"]["input_path"] == str(tmp_path / "videos")
    assert cfg["runtime"]["enabled_tools"] == ["ocr"]
    assert cfg["models"]["qwen_llm"]["api"]["api_key"] == "fixture-secret"
    assert cfg["models"]["qwen_llm"]["api"]["max_api_calls"] == 100
    assert "omniparser" not in cfg["models"]
    assert "whisper" not in required_assets(cfg)
    assert "fixture-secret" not in json.dumps(redact(cfg))
    assert dict(os.environ) == before


def test_external_vision_auth_and_low_protocol_preserve_user_configuration(tmp_path):
    path = small(tmp_path, vlm={"base_url": "https://provider.invalid/v1", "model": "reasoning-vision",
                               "api_key_env": "VISION_KEY", "wire_profile": "openai", "reasoning_effort": "low"},
                 llm={"reasoning_effort": "low"})
    cfg = load_config(path, environ={"VISION_KEY": "fixture-secret"})
    api = cfg["models"]["qwen_vlm"]["api"]
    assert api["api_key"] == "fixture-secret"
    assert api["wire_profile"] == "openai" and api["reasoning_effort"] == "low"
    assert cfg["models"]["qwen_llm"]["api"]["thinking"] is None


def test_tool_venv_executable_symlink_is_not_resolved_to_base_interpreter(tmp_path):
    target = tmp_path / "base-python"
    target.touch()
    link = tmp_path / "venv/bin/python"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    cfg = load_config(small(tmp_path, tool_python=str(link)), environ={})
    assert cfg["models"]["omniparser"]["python_executable"] == str(link)


@pytest.mark.parametrize("values,code", [
    ({"hardware": {"gpus": [0, 0]}}, "invalid_gpu_list"),
    ({"hardware": {"gpus": [True]}}, "invalid_gpu_list"),
    ({"source_name": "../escape"}, "invalid_source_name"),
    ({"tools": ["unknown"]}, "invalid_config_choices"),
    ({"capabilities": []}, "empty_capabilities"),
    ({"budget": {"max_api_attempts": 0}}, "invalid_api_budget"),
    ({"network": {"proxy_mode": "explicit"}}, "invalid_proxy_mode"),
    ({"llm": {"api_key": "inline-secret"}}, "unknown_config_field"),
])
def test_small_config_rejects_invalid_fields(tmp_path, values, code):
    with pytest.raises(ValueError, match=code):
        load_config(small(tmp_path, **values), environ={})


@pytest.mark.parametrize("dotenv,code", [
    ("X=${Y}\nY=${X}\n", "dotenv_reference_cycle"),
    ("X=$(shell)\n", "shell_substitution_disallowed"),
    ("X=${MISSING}\n", "unexpanded_environment_variables"),
])
def test_dotenv_fails_closed(tmp_path, dotenv, code):
    path = small(tmp_path)
    (tmp_path / ".env").write_text(dotenv)
    with pytest.raises(ValueError, match=code):
        load_config(path, environ={})


def workspace(tmp_path):
    source = tmp_path / "runs" / "source"
    (source / "output").mkdir(parents=True)
    (source / "logs" / "debug").mkdir(parents=True)
    (source / "temp").mkdir()
    image = source / "temp" / "frame.png"
    Image.new("RGB", (24, 24), "red").save(image)
    data = source / "output" / "data.jsonl"
    row = {"benchmark": "edit", "source_img": str(image), "target_img": str(image),
           "input_row": 8, "meta": {"video_path": "/source/video.mp4", "frame_paths": [str(image)]}}
    data.write_text(json.dumps(row) + "\n")
    (source / "logs" / "debug" / "v_seg000_METADATA.json").write_text(json.dumps({"metadata": {"frames": [{"img": str(image)}]}}))
    return source, data, image


def test_export_dry_run_and_portable_package_preserve_schema_and_provenance(tmp_path):
    source, data, image = workspace(tmp_path)
    original = data.read_bytes()
    target = tmp_path / "package"
    plan = export_run(source.parent, target, dry_run=True)
    assert not target.exists() and plan["counts"] == {"edit": 1, "metadata": 1}
    export_run(source.parent, target)
    relocated = tmp_path / "moved"
    target.rename(relocated)
    assert verify_package(relocated)["image_count"] == 1
    row = json.loads((relocated / "edit.jsonl").read_text())
    assert row["input_row"] == 8 and row["meta"]["video_path"] == "/source/video.mp4"
    assert row["source_img"].startswith("images/") and (relocated / row["target_img"]).is_file()
    assert row["_export_provenance"]["source_line"] == 1
    assert data.read_bytes() == original and image.exists()
    with pytest.raises(ValueError, match="export_destination_exists"):
        export_run(source.parent, relocated)
    (relocated / row["source_img"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="export_package_mismatch"):
        verify_package(relocated)


@pytest.mark.parametrize("mode,code", [("missing", "image_missing"), ("corrupt", "image_corrupt"), ("outside", "image_outside_asset_roots")])
def test_export_refuses_bad_images_before_writing(tmp_path, mode, code):
    source, data, image = workspace(tmp_path)
    if mode == "corrupt":
        image.write_bytes(b"broken")
    else:
        replacement = tmp_path / "other.png"
        if mode == "outside":
            Image.new("RGB", (8, 8)).save(replacement)
        data.write_text(json.dumps({"benchmark": "edit", "source_img": str(replacement)}) + "\n")
    target = tmp_path / "package"
    with pytest.raises(ValueError, match=code):
        export_run(source.parent, target)
    assert not target.exists()


def test_export_rejects_conflicting_record_keys(tmp_path):
    source, data, image = workspace(tmp_path)
    row = {"record_key": "one", "benchmark": "edit", "source_img": str(image)}
    data.write_text(json.dumps(row) + "\n" + json.dumps({**row, "instruction": "changed"}) + "\n")
    with pytest.raises(ValueError, match="duplicate_record_key"):
        export_run(source.parent, tmp_path / "package")


def test_export_rewrites_unified_raw_turns_and_typed_image_content(tmp_path):
    source, data, image = workspace(tmp_path)
    row = {"benchmark": "unify.comm.c1", "image_id": "frame.png",
           "turns": [{"in": {"start_image": str(image), "image_prev": str(image), "context_images": [str(image)]},
                      "gt": {"middle_image": str(image), "interleaved": [{"type": "image", "content": str(image)},
                             {"type": "text", "content": "Explain the frame."}]}}]}
    data.write_text(json.dumps(row) + "\n")
    export_run(source.parent, tmp_path / "package")
    packaged = json.loads((tmp_path / "package" / "unify.jsonl").read_text())
    assert str(image) not in json.dumps(packaged)
    assert packaged["image_id"] == "frame.png"
    assert packaged["turns"][0]["gt"]["interleaved"][0]["content"].startswith("images/")
    assert verify_package(tmp_path / "package")["ok"]


def test_export_does_not_follow_escaping_image_symlink(tmp_path):
    source, data, image = workspace(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    linked = source / "temp" / "link.png"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlink permission unavailable")
    data.write_text(json.dumps({"benchmark": "edit", "source_img": str(linked)}) + "\n")
    with pytest.raises(ValueError, match="image_outside_asset_roots"):
        export_run(source.parent, tmp_path / "package")


def test_offline_guards_block_models_and_network_and_restore_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    with offline_guards():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        with pytest.raises(RuntimeError, match="smoke_model_import_forbidden"):
            __import__("torch")
        with pytest.raises(RuntimeError, match="smoke_network_forbidden"):
            socket.create_connection(("127.0.0.1", 12345))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


def test_public_smoke_runs_real_writer_resume_and_export(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    result = run_smoke(tmp_path / "smoke")
    assert result["ok"] and result["simulated"]
    assert result["paid_api_calls"] == result["model_calls"] == result["real_optimizer_steps"] == 0
    assert result["portable_export"]["counts"] == {"understanding": 1, "edit": 1, "unify": 1, "metadata": 1}


def test_cli_init_never_overwrites_and_diagnostics_do_not_leak(tmp_path, capsys):
    path = tmp_path / "local.yaml"
    assert main(["init", "--output", str(path)]) == 0
    original = path.read_bytes()
    assert main(["init", "--output", str(path)]) == 1
    assert path.read_bytes() == original
    path.write_text("llm:\n  api_key: private-value\nversion: 1\n")
    assert main(["config", "validate", "--config", str(path)]) == 1
    assert "private-value" not in capsys.readouterr().err


def test_paper_recipes_validate_offline_without_claiming_deployment():
    from vidueu.reproduction import run_recipe
    root = Path(__file__).resolve().parents[1]
    for name in ("train_internvl3_understanding", "train_bagel_understanding", "train_bagel_editing", "train_bagel_unified", "eval_paper"):
        report = run_recipe(root / "configs" / (name + ".yaml"))
        assert report["ok"] and not report["executed"] and not report["ready_for_execution"]
        assert "adapter" in report["missing_deployment_fields"]


def test_paper_recipes_refuse_wrong_batch_and_judge_protocol(tmp_path):
    from vidueu.reproduction import run_recipe
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "train_internvl3_understanding.yaml").read_text())
    config["deployment"].update(gpus=[1, 2], gradient_accumulation=1)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="internvl_effective_batch_must_equal_32"):
        run_recipe(path)
    config = yaml.safe_load((root / "configs" / "eval_paper.yaml").read_text())
    config["paper"]["generation_passes"] = 3
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="paper_evaluation_protocol_mismatch"):
        run_recipe(path)


@pytest.mark.skipif(os.name != "posix", reason="Linux run manager")
def test_manager_proxy_modes_are_child_only_and_state_contains_no_proxy_secret(tmp_path, monkeypatch, capsys):
    from scripts import manage_run
    path = small(tmp_path, network={"proxy_mode": "explicit", "proxy_url": "http://user:fixture-password@proxy.invalid:8080"})
    monkeypatch.setenv("https_proxy", "http://original.invalid:80")
    before = dict(os.environ)
    direct = manage_run.child_environment(os.environ, "direct")
    assert "https_proxy" not in direct
    assert manage_run.child_environment(os.environ, "inherit")["https_proxy"] == before["https_proxy"]
    assert "fixture-password" in manage_run.child_environment(os.environ, "explicit", path)["https_proxy"]
    assert main(["run", "--config", str(path), "--dry-run"]) == 0
    assert "fixture-password" not in capsys.readouterr().out
    assert dict(os.environ) == before
