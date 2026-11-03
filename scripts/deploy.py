#!/usr/bin/env python3
"""Install all isolated runtimes and download the public model manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

if __package__:
    from .setup_env import PROFILES as ENV_PROFILES
else:
    from setup_env import PROFILES as ENV_PROFILES

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("pipeline", "vllm", "omniparser", "bagel", "dev")
DEFAULT_PROFILES = ("pipeline", "omniparser")
PROFILE_ASSETS = {
    "pipeline": ("clip", "whisper", "ocr_det", "ocr_rec", "yolo", "raft_code", "raft_weights", "blip2"),
    "vllm": ("qwen",),
    "omniparser": ("ocr_det", "ocr_rec", "omni_code", "omni_weights", "florence"),
    "bagel": ("bagel",),
    "dev": (),
}
# Code precedes weights where assets share a directory.
ASSETS = ("clip", "whisper", "ocr_det", "ocr_rec", "yolo", "raft_code",
          "raft_weights", "omni_code", "omni_weights", "florence", "qwen",
          "blip2", "bagel")


def discover_python(version, explicit=None):
    """Probe executables rather than assuming the SSH shell activates Conda."""
    candidates = [explicit] if explicit else [f"python{version}", sys.executable]
    if not explicit:
        for root in (Path.home() / "anaconda3", Path.home() / "miniconda3", Path("/opt/conda")):
            candidates.append(str(root / "bin/python"))
            if (root / "envs").is_dir():
                candidates.extend(str(p / "bin/python") for p in sorted((root / "envs").iterdir()))
    for candidate in candidates:
        executable = shutil.which(candidate)
        if not executable:
            continue
        try:
            actual = subprocess.check_output([executable, "-c", "import sys,venv,ensurepip;print('.'.join(map(str,sys.version_info[:2])))"], text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if actual == version:
            return executable
    raise ValueError(f"Python {version} unavailable; specify --python{version.replace('.', '')} with an executable path")


def discover_cuda():
    candidates = [shutil.which("nvcc")]
    if os.environ.get("CUDA_HOME"):
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"))
    candidates += [str(p / "bin/nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            subprocess.run([candidate, "--version"], check=True, timeout=15)
            root = str(Path(candidate).resolve().parents[1])
            os.environ["CUDA_HOME"] = root
            os.environ["PATH"] = str(Path(root) / "bin") + os.pathsep + os.environ.get("PATH", "")
            return root
    raise ValueError("CUDA compiler unavailable for BAGEL; set CUDA_HOME to a matching Toolkit")


def prepare_workspace(args):
    """Keep caches/build temporaries off the system filesystem by default."""
    workspace = args.work_dir.expanduser().absolute()
    paths = (args.env_prefix.expanduser().absolute(), args.model_root.expanduser().absolute(), workspace)
    checks = []
    for path in paths:
        if path.resolve() != path:
            raise ValueError(f"Deployment path must be canonical: {path}")
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free
        checks.append({"path": str(path), "free_gib": round(free / 2**30, 1)})
        if free < args.min_free_gib * 2**30:
            raise ValueError(f"Insufficient free space at {path}: {free / 2**30:.1f} GiB; reserve is {args.min_free_gib} GiB")
    print(json.dumps({"storage": checks, "note": "Reserve check is not a full model-size estimate"}), flush=True)
    for name in ("tmp", "cache/pip", "cache/huggingface", "cache/torch", "cache/triton"):
        path = workspace / name
        if path.resolve() != path:
            raise ValueError(f"Workspace path must not contain symlinks: {path}")
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(workspace / "tmp"), XDG_CACHE_HOME=str(workspace / "cache"),
                      PIP_CACHE_DIR=str(workspace / "cache/pip"), HF_HOME=str(workspace / "cache/huggingface"),
                      TORCH_HOME=str(workspace / "cache/torch"), TRITON_CACHE_DIR=str(workspace / "cache/triton"))


def profile_signature(profile):
    # Checks and adapter code may improve without changing installed packages.
    # Their real imports/inference are revalidated below instead of triggering
    # an unnecessary second installation of all CUDA wheels.
    files = [ROOT / "requirements" / f"{profile}.txt"]
    if profile == "bagel":
        files.append(ROOT / "vendor/bagel/requirements.txt")
    value = hashlib.sha256()
    value.update(json.dumps(ENV_PROFILES[profile], sort_keys=True).encode())
    for path in files:
        value.update(path.relative_to(ROOT).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def selected_profiles(args):
    requested = tuple(getattr(args, "profiles", None) or (() if getattr(args, "config", None) else DEFAULT_PROFILES))
    if "all" in requested:
        return PROFILES
    result = ["pipeline"] if getattr(args, "config", None) else []
    if "omniparser" in getattr(args, "config_tools", []):
        result.append("omniparser")
    for profile in requested:
        if profile not in PROFILES:
            raise ValueError(f"Unknown deployment profile: {profile}")
        if profile not in result:
            result.append(profile)
    return tuple(result)


def read_deploy_config(args):
    # Bootstrap remains independent of model dependencies. Most hosts already
    # have YAML; otherwise install only PyYAML into a new isolated bootstrap.
    try:
        import yaml
    except ImportError:
        if args.dry_run:
            if args.config:
                raise ValueError("Config dry-run requires PyYAML in the bootstrap interpreter")
            return
        bootstrap = args.work_dir.expanduser().absolute() / "bootstrap"
        if bootstrap.resolve() != bootstrap:
            raise ValueError("Bootstrap path must not contain symlinks")
        if not bootstrap.exists():
            subprocess.run([sys.executable, "-m", "venv", str(bootstrap)], check=True)
        python = bootstrap / "bin/python"
        command = [str(python), "-m", "pip", "install", "PyYAML>=6,<7"]
        if args.index_url:
            command += ["--index-url", args.index_url]
        subprocess.run(command, check=True)
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    if not args.config:
        return
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("--config requires a version: 1 pipeline configuration")
    tools = raw.get("tools", ["ocr", "asr", "yolo", "omniparser"])
    if not isinstance(tools, list) or any(t not in {"ocr", "asr", "yolo", "omniparser"} for t in tools):
        raise ValueError("Invalid tools in deployment config")
    # Validate the complete config before installation without calling an API.
    sys.path.insert(0, str(ROOT))
    from vidueu_config import load_config
    from vidueu.config import required_assets
    compiled = load_config(args.config)
    args.config_tools = compiled["runtime"]["enabled_tools"]
    args.config_assets = required_assets(compiled)
    args.config_raw = raw


def deployment_artifacts(args, profiles):
    """Generate relocatable user config and exact interpreter launchers."""
    import copy
    import yaml
    destination = args.launch_dir.expanduser().absolute()
    if destination.resolve() != destination:
        raise ValueError("Launch directory must be canonical, without symlinks")
    if args.config:
        raw = copy.deepcopy(args.config_raw)
        base = args.config.resolve().parent
    else:
        raw = yaml.safe_load((ROOT / "configs/pipeline.example.yaml").read_text())
        base = ROOT
        if "omniparser" not in profiles:
            raw["tools"] = [t for t in raw["tools"] if t != "omniparser"]
    # Keep placeholders (especially credentials) as references, never serialize
    # resolved config values, keys, or the deployment proxy environment.
    for name in ("input", "output"):
        value = raw.get(name, "./videos" if name == "input" else "./outputs/run-001")
        if "$" in str(value):
            raise ValueError(f"Use an explicit {name} path for deployment")
        raw[name] = str((base / value).resolve())
    for section in ("llm", "vlm"):
        if raw.get(section, {}).get("api_key_file"):
            raw[section]["api_key_file"] = str((base / raw[section]["api_key_file"]).resolve())
    python = str(args.env_prefix.expanduser().absolute() / "pipeline/bin/python")
    raw.update(model_root=str(args.model_root.expanduser().absolute()), python=python,
               tool_python=str(args.env_prefix.expanduser().absolute() / "omniparser/bin/python"))
    config_path = destination / "pipeline.local.yaml"
    launcher = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                + "cd -- " + shlex.quote(str(ROOT)) + "\n"
                + "exec " + shlex.quote(python) + " -m vidueu run --config "
                + shlex.quote(str(config_path)) + ' "$@"\n')
    return {"pipeline.local.yaml": yaml.safe_dump(raw, sort_keys=False), "run_pipeline.sh": launcher}


def publish_artifacts(args, profiles):
    destination = args.launch_dir.expanduser().absolute()
    content = deployment_artifacts(args, profiles)
    # Check every destination before writing any; a rerun preserves user edits.
    for name, value in content.items():
        path = destination / name
        if path.is_symlink() or (path.exists() and path.read_text() != value):
            raise ValueError(f"Refuse overwriting local deployment file: {path}; choose --launch-dir")
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        path = destination / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(value)
        if name.endswith(".sh"):
            path.chmod(0o700)
    return str(destination / "run_pipeline.sh")


def plan(args):
    profiles = selected_profiles(args)
    prefix = args.env_prefix.expanduser().absolute()
    model_root = args.model_root.expanduser().absolute()
    for path in (prefix, model_root):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Deployment path must be canonical, without symlinks: {path}")
    commands = []
    for profile in profiles:
        target = prefix / profile
        receipt = target / ".vidueu-deploy.json"
        signature = profile_signature(profile)
        if target.exists():
            if not receipt.is_file() or receipt.is_symlink():
                if not (target / ".vidueu-setup.json").is_file():
                    raise ValueError(f"Existing environment is not managed by deploy: {target}; choose --env-prefix")
                base = args.python312 if profile in {"vllm", "omniparser"} else args.python310
                command = [sys.executable, str(ROOT / "scripts/setup_env.py"), "--profile", profile,
                           "--env-prefix", str(prefix), "--python", base, "--resume"]
                if args.index_url:
                    command += ["--index-url", args.index_url]
                commands.append({"profile": profile, "argv": command, "receipt": str(receipt),
                                 "signature": signature, "reuse": False})
                continue
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved != {"profile": profile, "signature": signature} or not (target / "bin/python").is_file():
                raise ValueError(f"Environment recipe changed or incomplete: {target}; choose --env-prefix")
            # setup's receipt checks the exact package command recipe, skips
            # completed installs, and reruns current runtime and pip checks.
            base = args.python312 if profile in {"vllm", "omniparser"} else args.python310
            command = [sys.executable, str(ROOT / "scripts/setup_env.py"),
                       "--profile", profile, "--env-prefix", str(prefix), "--python", base, "--resume"]
            if args.index_url:
                command += ["--index-url", args.index_url]
        else:
            base = args.python312 if profile in {"vllm", "omniparser"} else args.python310
            command = [sys.executable, str(ROOT / "scripts/setup_env.py"),
                       "--profile", profile, "--env-prefix", str(prefix), "--python", base, "--resume"]
            if args.index_url:
                command += ["--index-url", args.index_url]
        commands.append({"profile": profile, "argv": command, "receipt": str(receipt),
                         "signature": signature, "reuse": target.exists()})
    assets = list(getattr(args, "config_assets", []))
    for profile in profiles:
        if args.config and profile == "pipeline":
            continue
        for asset in PROFILE_ASSETS[profile]:
            if asset not in assets:
                assets.append(asset)
    download_profile = next((p for p in ("pipeline", "omniparser", "bagel", "vllm", "dev") if p in profiles), profiles[0])
    python = str(prefix / f"{download_profile}/bin/python")
    download = [python, str(ROOT / "scripts/download_models.py"), "--model-root", str(model_root),
                "--only", *assets]
    if args.endpoint:
        download += ["--endpoint", args.endpoint]
    if args.download_timeout:
        download += ["--download-timeout", str(args.download_timeout)]
    verify = [python, str(ROOT / "scripts/download_models.py"), "--model-root", str(model_root),
              "--only", *assets, "--verify"]
    return commands, download, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", dest="profiles", action="append", choices=[*PROFILES, "all"],
                        help="Additional profiles with --config; otherwise explicit profiles. Default: pipeline + omniparser")
    parser.add_argument("--config", type=Path, help="Pipeline config; enabled tools determine required environments")
    parser.add_argument("--launch-dir", type=Path, help="Generated config and launcher directory; default: <work-dir>/launch")
    parser.add_argument("--download-timeout", type=int, help="Model and metadata request timeout; default 600 seconds")
    parser.add_argument("--env-prefix", type=Path, default=ROOT / "environments")
    parser.add_argument("--model-root", type=Path, default=ROOT / "model")
    parser.add_argument("--python310")
    parser.add_argument("--python312")
    parser.add_argument("--index-url")
    parser.add_argument("--endpoint", help="Optional Hugging Face mirror; Git/URL downloads keep official origins")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".deploy")
    parser.add_argument("--min-free-gib", type=float, default=20, help="Minimum free-space reserve on each target filesystem")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.launch_dir = args.launch_dir or args.work_dir / "launch"
    if args.min_free_gib < 0 or (args.download_timeout is not None and args.download_timeout <= 0):
        parser.error("Space reserve must be nonnegative and timeout positive")
    read_deploy_config(args)
    profiles = selected_profiles(args)
    if not args.dry_run:
        if any(p in profiles for p in ("pipeline", "bagel", "dev")):
            args.python310 = discover_python("3.10", args.python310)
        if any(p in profiles for p in ("vllm", "omniparser")):
            args.python312 = discover_python("3.12", args.python312)
        if "bagel" in profiles:
            args.cuda_home = discover_cuda()
        print(json.dumps({"profiles": profiles, "python310": args.python310,
                          "python312": args.python312, "cuda_home": getattr(args, "cuda_home", None)}), flush=True)
    else:
        args.python310 = args.python310 or "python3.10"
        args.python312 = args.python312 or "python3.12"
    environments, download, verify = plan(args)
    runtime_checks = []
    for profile in profiles:
        if profile in {"pipeline", "omniparser"}:
            runtime_checks.append([str(args.env_prefix.expanduser().absolute() / profile / "bin/python"),
                                   str(ROOT / "scripts/check_runtime.py"), "--profile", profile,
                                   "--model-root", str(args.model_root.expanduser().absolute())])
            if profile == "omniparser" or not args.config or "ocr" in args.config_tools:
                runtime_checks[-1].append("--inference")
        elif profile in {"vllm", "bagel", "dev"}:
            runtime_checks.append([str(args.env_prefix.expanduser().absolute() / profile / "bin/python"),
                                   str(ROOT / "scripts/check_runtime.py"), "--profile", profile])
    print(json.dumps({"environments": environments, "download": download, "verify": verify,
                      "runtime_checks": runtime_checks,
                      "launch_files": [str(args.launch_dir.expanduser().absolute() / name)
                                       for name in ("pipeline.local.yaml", "run_pipeline.sh")]
                                      if "pipeline" in profiles else []}, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be nonnegative")
    prepare_workspace(args)
    if sys.platform != "linux":
        parser.error("Deployment supports Linux GPU hosts")
    required_tools = ["git", "ffmpeg", "ffprobe"]
    if "bagel" in profiles:
        required_tools += ["nvcc", "c++"]
    for tool in required_tools:
        if not shutil.which(tool):
            parser.error(f"Install system prerequisite first: {tool}")
    required_python = []
    if any(p in profiles for p in ("pipeline", "bagel", "dev")):
        required_python.append(("3.10", args.python310))
    if any(p in profiles for p in ("vllm", "omniparser")):
        required_python.append(("3.12", args.python312))
    for version, executable in required_python:
        if not shutil.which(executable):
            parser.error(f"Interpreter not found: {executable}")
        actual = subprocess.check_output([executable, "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"], text=True).strip()
        if actual != version:
            parser.error(f"Expected Python {version}, found {actual}")
    for item in environments:
        subprocess.run(item["argv"], cwd=ROOT, check=True)
        if not item["reuse"]:
            # Publish completion only after setup and pip check succeed.
            with Path(item["receipt"]).open("x", encoding="utf-8") as stream:
                json.dump({"profile": item["profile"], "signature": item["signature"]}, stream)
    if len(download) > download.index("--only") + 1:
        subprocess.run(download, cwd=ROOT, check=True)
        subprocess.run(verify, cwd=ROOT, check=True)
    for command in runtime_checks:
        subprocess.run(command, cwd=ROOT, check=True)
    # Generate only after assets and real adapter checks pass.
    if "pipeline" in profiles:
        launcher = publish_artifacts(args, profiles)
        print("Deployment complete. Start with: bash " + shlex.quote(launcher))
    else:
        print("Deployment complete for selected optional profiles.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
