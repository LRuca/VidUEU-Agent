#!/usr/bin/env python3
"""Audit the public source closure without reading generated data or credentials."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ("review_pipeline_candidates.py", "candidate_review.py", "review_paper_v4.py",
           "review_paper_v5.py", "review_paper_v10.py", "export_reviewed_paper.py",
           "replay_paper_prompt_alignment.py", "manage_edit_judge_proxy.sh", "run_edit_paper7_judge.sh")
SECRET = re.compile(r"(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{28,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")


def main():
    errors, files = [], []
    root_files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".py", ".json", ".sh", ".md", ".toml", ".cff", ".ini"}]
    for p in root_files:
        if SECRET.search(p.read_text(encoding="utf-8-sig")):
            errors.append({"path": p.name, "error": "credential_pattern"})
    for base in ("vidueu", "executor", "orchestrator", "dataset", "scripts", "prompts", "configs", "manifests", "tests", "docs", "examples", "vendor/bagel"):
        for p in sorted((ROOT / base).rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink():
                errors.append({"path": rel, "error": "source_symlink"})
                continue
            if p.name in PRIVATE or p.suffix in {".pt", ".pth", ".mp4", ".safetensors", ".key"} or ".local." in p.name or ".realtest." in p.name:
                errors.append({"path": rel, "error": "private_or_generated_asset_in_release"})
            if p.stat().st_size > 20 * 1024**2:
                errors.append({"path": rel, "error": "large_source_file"})
            if p.suffix not in {".py", ".json", ".yaml", ".yml", ".txt", ".md", ".toml", ".sh"}:
                continue
            data = p.read_bytes()
            text = data.decode("utf-8-sig")
            if SECRET.search(text):
                errors.append({"path": rel, "error": "credential_pattern"})
            if p.suffix == ".py":
                try:
                    tree = ast.parse(text, filename=rel)
                    if not rel.startswith("vendor/"):
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                                module = node.module
                                if module.split(".")[0] in {"vidueu", "executor", "orchestrator", "scripts", "dataset"}:
                                    candidate = ROOT.joinpath(*module.split("."))
                                    if not candidate.with_suffix(".py").exists() and not candidate.is_dir():
                                        errors.append({"path": rel, "error": "missing_internal_module", "module": module})
                except SyntaxError:
                    errors.append({"path": rel, "error": "python_syntax"})
            files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest()})
    from scripts.download_models import load_manifest
    load_manifest(ROOT / "manifests/models.yaml")
    patch_manifest = json.loads((ROOT / "manifests/vendor_patches.json").read_text())
    for change in patch_manifest["changes"]:
        path = ROOT / "vendor/bagel" / change["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != change["release_sha256"]:
            errors.append({"path": str(path.relative_to(ROOT)), "error": "vendor_patch_hash_mismatch"})
    if (ROOT / "prompts/legacy.json").read_bytes() != (ROOT / "extracted_system_prompts.json").read_bytes():
        errors.append({"path": "prompts/legacy.json", "error": "legacy_prompt_copy_changed"})
    from executor.tasks.prompt_loader import PAPER_PROFILES, load_prompt_text
    for profile in sorted(PAPER_PROFILES):
        try:
            load_prompt_text("paper", "routing", profile=profile)
        except (KeyError, ValueError, OSError):
            errors.append({"path": f"prompts/{profile}.json", "error": "missing_profile_resource"})
    if not (ROOT / "LICENSE").is_file():
        errors.append({"path": "LICENSE", "error": "missing_project_license"})
    print(json.dumps({"ok": not errors, "source_files_checked": len(files), "errors": errors,
                      "licensing": "Apache-2.0 for original code; third-party licenses remain separate; see NOTICE.md"}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
