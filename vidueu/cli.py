"""Unified public command line. Heavy model modules are imported only by runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from vidueu_config import load_config, redact
from vidueu.config import required_assets, template
from vidueu.diagnostics import diagnostic


def config_arguments(parser):
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output", "--output-root", dest="output")


def configured(args):
    overrides = {name: getattr(args, name) for name in ("input", "output") if getattr(args, name, None) is not None}
    # CLI paths are explicit shell paths; file paths use the configuration directory.
    return load_config(args.config, overrides={name: str(Path(value).expanduser().resolve()) for name, value in overrides.items()})


def print_doctor(report, as_json=False):
    if as_json:
        print(json.dumps(report, indent=2))
        return
    for item in report["checks"]:
        print(f"{'OK' if item['ok'] else 'FAIL'} {item['name']}")
        if not item["ok"]:
            print(f"  {item.get('code', item.get('error'))}: {item.get('hint', 'Inspect this check.')}")
        elif isinstance(item.get("detail"), dict):
            print("  " + json.dumps(item["detail"]))
    print("Ready" if report["ok"] else "Preflight failed; fix the checks above.")


def parser():
    p = argparse.ArgumentParser(prog="vidueu", description="Generate and export evidence-grounded video instruction data.")
    sub = p.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="Create a small config without overwriting files")
    init.add_argument("--preset", default="external-vlm", choices=("external-vlm",))
    init.add_argument("--output", default="config.local.yaml")
    config = sub.add_parser("config", help="Validate, inspect or print a configuration example")
    cs = config.add_subparsers(dest="config_action", required=True)
    cs.add_parser("example")
    for action in ("validate", "show"):
        c = cs.add_parser(action)
        config_arguments(c)
    doctor = sub.add_parser("doctor", help="Explain environment, inputs, assets and endpoint readiness")
    config_arguments(doctor)
    for flag in ("assets", "endpoint", "gpu", "media"):
        doctor.add_argument("--check-" + flag, action="store_true")
    doctor.add_argument("--lightweight", action="store_true", help="Only configuration, filesystem and system-command checks")
    doctor.add_argument("--json", action="store_true")
    assets = sub.add_parser("assets", help="Download or verify assets selected by the configuration")
    aa = assets.add_subparsers(dest="assets_action", required=True)
    aa.add_parser("list")
    for action in ("download", "verify"):
        a = aa.add_parser(action)
        config_arguments(a)
        a.add_argument("--only", nargs="+")
        a.add_argument("--endpoint")
        a.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run", help="Start a separately owned pipeline process")
    config_arguments(run)
    run.add_argument("--gpus", help="Explicit physical GPU indices; defaults to config hardware.gpus")
    run.add_argument("--source-name")
    run.add_argument("--tmux", action="store_true")
    run.add_argument("--dashboard", action="store_true")
    run.add_argument("--dashboard-port", type=int, default=8766)
    run.add_argument("--dry-run", action="store_true")
    for action in ("status", "stop"):
        a = sub.add_parser(action)
        a.add_argument("--output", "--run-dir", dest="output", required=True)
    service = sub.add_parser("vlm", help="Manage a local loopback vLLM service")
    service.add_argument("arguments", nargs=argparse.REMAINDER)
    export = sub.add_parser("export", help="Copy task data, metadata and referenced images to a new portable package")
    export.add_argument("--output", "--run-root", dest="output", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--asset-root", action="append", default=[])
    export.add_argument("--dry-run", action="store_true")
    paper_export = sub.add_parser("paper-export", help="Convert a portable package to the paper-schema-aligned release package")
    paper_export.add_argument("--source", required=True)
    paper_export.add_argument("--destination", required=True)
    paper_export.add_argument("--sample-limit", type=int)
    paper_export.add_argument("--dry-run", action="store_true")
    paper_export.add_argument("--allow-unresolved", action="store_true", help="Retain known gaps in an explicit candidate; publish_ready remains false")
    smoke = sub.add_parser("smoke", help="Offline CPU contract smoke; no model imports, GPU or API calls")
    smoke.add_argument("--output", required=True)
    reproduce = sub.add_parser("reproduce", help="Validate or plan paper training and evaluation recipes")
    reproduce.add_argument("operation", choices=("validate", "plan", "train", "eval"))
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--execute", action="store_true", help="Explicitly execute the printed training/evaluation command")
    return p


def dispatch(args):
    if args.action == "init":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(template(args.preset), stream, sort_keys=False)
        print(f"Created {target.resolve()}. Set input/output/model_root and the provider credentials, then run vidueu doctor.")
        return 0
    if args.action == "config":
        if args.config_action == "example":
            print(yaml.safe_dump(template(), sort_keys=False))
            return 0
        cfg = configured(args)
        print(json.dumps(redact(cfg) if args.config_action == "show" else {"ok": True, "config": str(Path(args.config).resolve())}, indent=2))
        return 0
    if args.action == "doctor":
        from scripts.doctor import inspect
        cfg = configured(args)
        mask = cfg.get("_vidueu", {}).get("gpus", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        # The doctor checks the requested mapping without exposing extra devices.
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
        try:
            report = inspect(cfg, assets=args.check_assets, endpoint=args.check_endpoint,
                             gpu=args.check_gpu, media=args.check_media, lightweight=args.lightweight)
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        print_doctor(report, args.json)
        return 0 if report["ok"] else 1
    if args.action == "assets":
        from scripts.download_models import main
        if args.assets_action == "list":
            return main(["--list"])
        cfg = configured(args)
        root = cfg.get("_vidueu", {}).get("model_root")
        if not root:
            raise ValueError("small_config_required_for_asset_selection")
        command = ["--model-root", root, "--only", *(args.only or required_assets(cfg))]
        if args.endpoint:
            command += ["--endpoint", args.endpoint]
        if args.dry_run:
            command += ["--dry-run"]
        if args.assets_action == "verify":
            command += ["--verify"]
        return main(command)
    if args.action in {"status", "stop"}:
        from scripts import manage_run
        return manage_run.main([args.action, "--run-dir", str(Path(args.output).resolve())])
    if args.action == "run":
        from scripts import manage_run
        cfg = configured(args)
        meta = cfg.get("_vidueu", {})
        command = ["start", "--config", str(Path(args.config).resolve()),
                   "--run-dir", cfg["runtime"]["output_root"], "--python", meta.get("python", sys.executable),
                   "--gpus", args.gpus if args.gpus is not None else meta.get("gpus", ""),
                   "--proxy-mode", meta.get("proxy_mode", "direct")]
        if args.input:
            command += ["--input", str(Path(args.input).resolve())]
        if args.source_name or meta.get("source_name"):
            command += ["--source-name", args.source_name or meta["source_name"]]
        if args.dashboard:
            command += ["--dashboard", "--port", str(args.dashboard_port)]
        if args.tmux:
            command += ["--tmux"]
        if args.dry_run:
            command += ["--dry-run"]
        return manage_run.main(command)
    if args.action == "vlm":
        from scripts.manage_run import main
        return main([*args.arguments, "--kind", "vllm"])
    if args.action == "export":
        from vidueu.export import export_run
        print(json.dumps(export_run(args.output, args.destination, asset_roots=args.asset_root, dry_run=args.dry_run), indent=2))
        return 0
    if args.action == "paper-export":
        from vidueu.paper_export import paper_schema_export
        print(json.dumps(paper_schema_export(args.source, args.destination,
                                             sample_limit=args.sample_limit,
                                             dry_run=args.dry_run,
                                             allow_unresolved=args.allow_unresolved), indent=2))
        return 0
    if args.action == "smoke":
        from vidueu.smoke import run_smoke
        print(json.dumps(run_smoke(args.output), indent=2))
        return 0
    if args.action == "reproduce":
        from vidueu.reproduction import run_recipe
        print(json.dumps(run_recipe(args.config, args.operation, execute=args.execute), indent=2))
        return 0
    raise ValueError("unknown_command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return dispatch(args)
    except (ValueError, OSError, KeyError, RuntimeError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"ok": False, **diagnostic(exc)}, indent=2), file=sys.stderr)
        return 1
