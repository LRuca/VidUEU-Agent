#!/usr/bin/env python3
"""Install or resume a managed venv without repeating completed package stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "pipeline": ("3.10", "cu124", ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]),
    "dev": ("3.10", "cu124", []),
    "vllm": ("3.12", "cu128", []),
    "omniparser": ("3.12", "cu124", ["torch==2.6.0", "torchvision==0.21.0"]),
    "bagel": ("3.10", "cu124", ["torch==2.5.1", "torchvision==0.20.1"]),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="pipeline")
    p.add_argument("--env-prefix", type=Path, default=ROOT / "environments")
    p.add_argument("--python", help="Base interpreter; must match the profile minor version")
    p.add_argument("--index-url", help="Optional pip mirror for non-Torch packages")
    p.add_argument("--wheelhouse", type=Path, help="Offline wheel directory; applies to all installs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume only this script's matching installation recipe")
    args = p.parse_args(argv)
    version, cuda, torch = PROFILES[args.profile]
    base = args.python or f"python{version}"
    prefix = args.env_prefix.expanduser().absolute()
    if prefix.is_symlink() or prefix.resolve() != prefix:
        p.error("Environment prefix must not contain symlinks")
    target = prefix / args.profile
    if target.is_symlink() or (target.exists() and not args.resume):
        p.error(f"Refusing to alter an existing environment: {target}")
    python = str(target / "bin/python")
    commands = [[base, "-m", "venv", str(target)],
                [python, "-m", "pip", "install", "pip>=24,<26", "setuptools>=70", "wheel"]]
    if torch:
        commands.append([python, "-m", "pip", "install", *torch,
                         "--index-url", f"https://download.pytorch.org/whl/{cuda}"])
    command = [python, "-m", "pip", "install", "-r", str(ROOT / "requirements" / f"{args.profile}.txt")]
    if args.index_url:
        command += ["--index-url", args.index_url]
    commands.append(command)
    if args.profile in {"pipeline", "dev"}:
        commands.append([python, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
    if args.profile == "bagel":
        commands.append([python, "-m", "pip", "install", "flash-attn==2.5.8", "--no-build-isolation"])
    install_count = len(commands)
    commands.append([python, str(ROOT / "scripts/check_runtime.py"), "--profile", args.profile])
    commands.append([python, "-m", "pip", "check"])
    for command in commands:
        if "pip" not in command or "install" not in command:
            continue
        if args.wheelhouse:
            if "--index-url" in command:
                index = command.index("--index-url")
                del command[index:index + 2]
            command += ["--no-index", "--find-links", str(args.wheelhouse.expanduser().absolute())]
        elif args.index_url and "--index-url" not in command:
            command += ["--index-url", args.index_url]
    requirements = (ROOT / "requirements" / f"{args.profile}.txt").read_bytes()

    def recipe_hash(argv, *, runtime=False):
        digest = hashlib.sha256(json.dumps(argv, sort_keys=True).encode())
        digest.update(requirements)
        if runtime:
            digest.update((ROOT / "scripts/check_runtime.py").read_bytes())
        if args.profile == "bagel":
            digest.update((ROOT / "vendor/bagel/requirements.txt").read_bytes())
        return digest.hexdigest()

    recipe = recipe_hash(commands, runtime=True)
    # Runtime probe edits must not force reinstalling unchanged package recipes.
    install_recipe = recipe_hash(commands[:install_count])
    # Receipts from before check_runtime.py was introduced can be recognized
    # exactly, using the current requirements and the original command order.
    legacy_recipe = recipe_hash(commands[:install_count] + [commands[-1]])
    state_path = target / ".vidueu-setup.json"
    state = {"recipe": recipe, "install_recipe": install_recipe, "completed": 0}
    resume_reason = "new_environment"
    if target.exists():
        if not state_path.is_file() or state_path.is_symlink():
            p.error(f"Refusing unmanaged environment: {target}")
        state = json.loads(state_path.read_text())
        if type(state.get("completed")) is not int or not 0 <= state["completed"] <= len(commands):
            p.error(f"Invalid installation receipt: {target}")
        if state.get("recipe") == recipe:
            resume_reason = "matching_recipe"
        elif state.get("install_recipe") == install_recipe:
            resume_reason = "runtime_checks_updated"
        elif state.get("recipe") == legacy_recipe and state["completed"] <= install_count + 1:
            resume_reason = "verified_legacy_recipe"
        else:
            p.error(f"Installation recipe changed: {target}; choose a new prefix")
        # Always repeat validation, even after a previously successful install.
        # A legacy receipt is upgraded only as these checks actually succeed.
        state = {**state, "recipe": recipe, "install_recipe": install_recipe,
                 "completed": min(state["completed"], install_count)}
    print(json.dumps({"profile": args.profile, "python_minor": version, "commands": commands,
                      "resume_reason": resume_reason,
                      "pending_commands": commands[state["completed"]:],
                      "system_dependencies": {x: shutil.which(x) for x in ("ffmpeg", "ffprobe", "git", "tmux")}}, indent=2))
    if args.dry_run:
        return 0
    if shutil.which(base) is None:
        p.error(f"Interpreter not found: {base}")
    actual = subprocess.check_output([base, "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"], text=True).strip()
    if actual != version:
        p.error(f"Expected Python {version}, found {actual}")
    if not all(shutil.which(x) for x in ("ffmpeg", "ffprobe", "git")):
        p.error("Install ffmpeg (including ffprobe) and git through your system package manager first")
    prefix.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.mkdir()
        with state_path.open("x") as stream:
            json.dump(state, stream)
    env = os.environ.copy()
    # Do not inherit private/global package indexes or extra package sources.
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL=args.index_url or "https://pypi.org/simple",
               PIP_EXTRA_INDEX_URL="", PIP_FIND_LINKS="", PIP_RETRIES="2", PIP_TIMEOUT="20")
    if args.wheelhouse:
        env.update(PIP_NO_INDEX="1", PIP_FIND_LINKS=str(args.wheelhouse.expanduser().absolute()))
    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(state_path)

    for index, command in enumerate(commands):
        if index < state["completed"]:
            continue
        print(f"[{args.profile}] step {index + 1}/{len(commands)}", flush=True)
        try:
            subprocess.run(command, check=True, cwd=ROOT, env=env)
        except (OSError, subprocess.CalledProcessError):
            print(f"[{args.profile}] installation stopped at step {index + 1}. "
                  "Completed stages were retained; rerun the same command with --resume.",
                  file=sys.stderr, flush=True)
            raise
        state["completed"] = index + 1
        save_state()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode or 1)
    except OSError as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
