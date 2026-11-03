from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STAGE_FILE_RE = re.compile(r"^(?P<video_base>.+)_seg(?P<segment>\d{3})_(?P<stage>T1|T2|T3|T4)\.json$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _video_id_from_path(video_path: str) -> str:
    if not video_path:
        return ""
    return os.path.splitext(os.path.basename(video_path))[0]


def _find_stage_sets(input_root: Path) -> List[Dict[str, Path]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Path]] = defaultdict(dict)
    for path in input_root.rglob("*.json"):
        match = STAGE_FILE_RE.match(path.name)
        if not match:
            continue
        stage = match.group("stage")
        segment = match.group("segment")
        video_base = match.group("video_base")
        group_key = (str(path.parent.resolve()), video_base, segment)
        grouped[group_key][stage] = path
    return [entry for entry in grouped.values() if "T4" in entry]


def _stage_output(stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    output = stage_payload.get("output", {})
    return output if isinstance(output, dict) else {}


def build_bundle_from_stage_set(stage_set: Dict[str, Path], root: Optional[Path] = None) -> Dict[str, Any]:
    t1_stage = _load_json(stage_set["T1"]) if "T1" in stage_set else {}
    t2_stage = _load_json(stage_set["T2"]) if "T2" in stage_set else {}
    t3_stage = _load_json(stage_set["T3"]) if "T3" in stage_set else {}
    t4_stage = _load_json(stage_set["T4"])

    t1_out = _stage_output(t1_stage)
    t2_out = _stage_output(t2_stage)
    t3_out = _stage_output(t3_stage)

    evidence = t4_stage.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    runtime = t4_stage.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    decision = t4_stage.get("decision", {})
    decision = decision if isinstance(decision, dict) else {}

    segment_index = int(runtime.get("segment_index", 0) or 0)
    video_path = str(runtime.get("video_path", "") or "")
    segment_path = str(runtime.get("segment_path", "") or "")
    video_id = str(evidence.get("video_id", "") or _video_id_from_path(video_path))

    frames_aligned = list(evidence.get("frames") or t3_out.get("frames") or [])
    evidence_units = list(evidence.get("evidence_units") or [])
    pair_records = list(evidence.get("aligned_pairs") or [])
    candidate_pairs = list(evidence.get("candidate_pairs") or [])
    reviewed_ok_pairs = list(evidence.get("reviewed_ok_pairs") or [])

    debug_paths = {
        stage: os.path.relpath(str(path), str(root)) if root is not None else str(path)
        for stage, path in sorted(stage_set.items())
    }

    return {
        "segment": {
            "index": segment_index,
            "video_id": video_id,
            "video_path": video_path,
            "segment_path": segment_path,
            "debug_paths": debug_paths,
        },
        "t1_out": t1_out,
        "t2_out": t2_out,
        "t3_out": t3_out,
        "frames_aligned": frames_aligned,
        "evidence": evidence,
        "evidence_units": evidence_units,
        "pair_records": pair_records,
        "candidate_pairs": candidate_pairs,
        "reviewed_ok_pairs": reviewed_ok_pairs,
        "t4_decision": decision,
        "t4_runtime": runtime,
    }


def build_manifest(bundles: Iterable[Dict[str, Any]], *, input_root: Path) -> Dict[str, Any]:
    videos: Dict[str, Dict[str, Any]] = {}
    for bundle in bundles:
        segment = bundle.get("segment", {})
        if not isinstance(segment, dict):
            continue
        video_path = str(segment.get("video_path", "") or "")
        if video_path not in videos:
            videos[video_path] = {"video_path": video_path, "video_id": str(segment.get("video_id", "") or ""), "segments": []}
        videos[video_path]["segments"].append(
            {
                "segment_index": int(segment.get("index", 0) or 0),
                "segment_path": str(segment.get("segment_path", "") or ""),
                "debug_paths": dict(segment.get("debug_paths", {}) or {}),
            }
        )
    for video in videos.values():
        video["segments"].sort(key=lambda item: int(item.get("segment_index", 0) or 0))
    return {
        "run_meta": {
            "input_root": str(input_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "video_count": len(videos),
            "segment_count": sum(len(v["segments"]) for v in videos.values()),
        },
        "videos": list(videos.values()),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble per-segment UMM bundles from run_pipeline stage debug dumps.")
    parser.add_argument("--input-root", required=True, help="Run root or any parent directory that contains logs/debug stage json files.")
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="Path to write the assembled segment bundles jsonl. Defaults to <input-root>/umm_bundles/segment_bundles.jsonl",
    )
    parser.add_argument(
        "--manifest-out",
        default="",
        help="Path to write a run-level manifest json. Defaults to <input-root>/umm_bundles/run_manifest.json",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else input_root / "umm_bundles" / "segment_bundles.jsonl"
    manifest_out = Path(args.manifest_out).resolve() if args.manifest_out else input_root / "umm_bundles" / "run_manifest.json"

    stage_sets = _find_stage_sets(input_root)
    bundles = [build_bundle_from_stage_set(stage_set, root=input_root) for stage_set in stage_sets]
    bundles.sort(
        key=lambda item: (
            str((item.get("segment", {}) or {}).get("video_path", "") or ""),
            int((item.get("segment", {}) or {}).get("index", 0) or 0),
        )
    )
    manifest = build_manifest(bundles, input_root=input_root)

    write_jsonl(output_jsonl, bundles)
    write_json(manifest_out, manifest)

    print(f"Wrote {len(bundles)} segment bundles to: {output_jsonl}")
    print(f"Wrote manifest to: {manifest_out}")


if __name__ == "__main__":
    main()
