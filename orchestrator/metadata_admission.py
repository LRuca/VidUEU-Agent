"""Retain evidence bindings while filtering signals and admitting segment metadata."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from orchestrator.planner_helpers import parse_llm_json
from executor.tasks.prompt_loader import render_prompt_text, current_prompt_profile, configured_prompt_profile, structured_evidence_profile
from orchestrator.prompt_audit import audited_text, audited_vision, paper_prompt_stage
from orchestrator.visual_state import validate_frame_observation, validate_pair_observation
from orchestrator.observation_patch import validate_patched_observation


class GateCResponseError(ValueError):
    """A retryable Gate C response-contract failure.

    This remains a ``ValueError`` for callers that historically treated an
    invalid model response as an exception, while carrying enough bounded
    provenance for the pipeline-level recovery path to hold one candidate
    instead of aborting an entire video.
    """

    def __init__(self, message: str, *, code: str, raw_response: Any,
                 metadata_id: str | None, review_binding: dict[str, Any],
                 model_identity: dict[str, Any]):
        super().__init__(message)
        raw = raw_response if isinstance(raw_response, str) else str(raw_response)
        self.code = str(code)
        self.raw_response = raw
        self.metadata_id = metadata_id
        self.review_binding = dict(review_binding or {})
        self.model_identity = dict(model_identity or {})

    def audit_payload(self) -> dict[str, Any]:
        raw = self.raw_response
        return {
            "code": self.code,
            "message": str(self),
            "metadata_id": self.metadata_id,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "raw_response_chars": len(raw),
            # Keep diagnostics bounded; full responses remain in the existing
            # chat-history/prompt audit logs when those are enabled.
            "raw_response_preview": raw[:2000],
            "review_binding": self.review_binding,
            "model_identity": self.model_identity,
        }


def paper_profile(config: dict) -> bool:
    return config.get("tasks", {}).get("T4", {}).get("task_profile", "paper") == "paper"


def filter_signals(planner: Any, t1: dict, t2: dict) -> tuple[dict, dict]:
    filtered = copy.deepcopy(t2)
    signals = []
    for pos, frame in enumerate(filtered.get("keyframes", [])):
        profile = frame.get("semantic_profile", {}) or {}
        for tool, text in (profile.get("tool_descs", {}) or {}).items():
            if str(text).strip():
                signals.append({"id": f"F{pos:02d}_TOOL_{tool.upper()}", "pos": pos, "tool": tool,
                                "signal": text, "shot_id": frame.get("shot_id", ""),
                                "caption": frame.get("caption") or frame.get("quality_profile", {}).get("caption", "")})
    if not signals:
        return filtered, {"status": "no_signals", "retained": [], "removed": []}
    profile = current_prompt_profile() if current_prompt_profile() != "legacy" else configured_prompt_profile(getattr(planner, "config", {}) or {})
    prompt = render_prompt_text("paper", "signal_filter", profile=profile,
        shots_json=json.dumps(t1.get('shot_profiles', t1.get('shots', [])), ensure_ascii=False),
        signals_json=json.dumps(signals, ensure_ascii=False))
    raw = audited_text(planner.llm, prompt, max_new_tokens=2200)
    result = parse_llm_json(raw)
    ids = {s["id"] for s in signals}
    classified = []
    if not isinstance(result, dict):
        raise ValueError("Signal filter returned invalid JSON")
    for group in ["retained", "removed"]:
        if not isinstance(result.get(group), list):
            raise ValueError("Signal filter lacks retained/removed decisions")
        for row in result[group]:
            if not isinstance(row, dict) or row.get("id") not in ids or not str(row.get("reason", "")).strip():
                raise ValueError("Signal filter returned unknown ID or missing reason")
            classified.append(row["id"])
    if len(set(classified)) != len(classified) or set(classified) != ids:
        raise ValueError("Signal filter did not classify every signal exactly once")
    keep = {s["id"] for s in result["retained"]}
    semantics = []
    by_shot = {}
    for pos, frame in enumerate(filtered.get("keyframes", [])):
        profile = frame.get("semantic_profile", {}) or {}
        descs = {tool: text for tool, text in (profile.get("tool_descs", {}) or {}).items()
                 if f"F{pos:02d}_TOOL_{tool.upper()}" in keep}
        frame["semantic_profile"] = {**profile, "tool_descs": descs, "semantic_text": " | ".join(descs.values())}
        # Only retained tools are visible downstream; raw data remains in the original T2 stage log.
        if isinstance(frame.get("tool_signals"), dict):
            frame["tool_signals"]["tools"] = {k: v for k, v in frame["tool_signals"].get("tools", {}).items() if k in descs}
        semantics.append(frame)
        shot = by_shot.setdefault(str(frame.get("shot_id", "")), {})
        for tool, text in descs.items():
            shot.setdefault(tool, {"summary": "", "signals": []})["signals"].append(text)
    for shot in by_shot.values():
        for row in shot.values():
            row["summary"] = "\n".join(dict.fromkeys(row.pop("signals")))
    filtered["semantic_profiles"] = semantics
    filtered["value_profiles"] = semantics
    filtered["tool_execution"] = {"semantic_by_shot": by_shot}
    return filtered, {"status": "filtered", "raw_response": raw, "signals": signals, **result}


def build_metadata(t1: dict, t2: dict, t3: dict, segment: Any) -> dict:
    start = float(getattr(segment, "start_sec", 0) or 0)
    source = str(getattr(segment, "video_path", "") or getattr(segment, "segment_path", ""))
    key = hashlib.sha256(f"{source}|{getattr(segment, 'index', 0)}|{start}".encode()).hexdigest()[:24]
    by_key = {(f.get("shot_id", ""), f.get("origin_index", f.get("frame_index", f.get("index", i)))): f
              for i, f in enumerate(t2.get("keyframes", []))}
    frames = []
    transitions = []
    for pos, frame in enumerate(t3.get("frames", [])):
        original = by_key.get((frame.get("shot_id", ""), frame.get("origin_index", pos)), {})
        frames.append({**copy.deepcopy(original), **copy.deepcopy(frame), "frame_id": f"F{pos:02d}",
                       "segment_sec": frame.get("sec"), "source_sec": start + float(frame.get("sec", 0) or 0)})
        if pos + 1 < len(t3.get("frames", [])):
            transitions.append({"transition_id": f"T{pos:02d}", "source": f"F{pos:02d}",
                                "target": f"F{pos+1:02d}", "text": frame.get("transition_to_next", ""),
                                "cause": frame.get("cause", "unknown"),
                                "validation": frame.get("transition_validation")})
    return {"schema_version": "vidueu.segment.v1", "metadata_id": key,
            "video": {"source_file": source, "segment_path": getattr(segment, "segment_path", ""),
                      "segment_index": getattr(segment, "index", 0), "start_sec": start,
                      "end_sec": getattr(segment, "end_sec", None), "summary_scope": "segment",
                      "summary": t3.get("segment_summary", ""), "type": t1.get("video_type", "other"),
                      "video_type": t1.get("video_type"),
                      "policy_domain": t1.get("policy_domain"),
                      "domain_source": t1.get("domain_source"),
                      "domain_confidence": t1.get("domain_confidence")},
            "chapters": t1.get("chapters", []), "shots": t1.get("shot_profiles", t1.get("shots", [])),
            "summary_hierarchy": t3.get("summary_hierarchy", {}),
            "keyframes": frames, "transitions": transitions,
            "signal_filter": t3.get("signal_filter", {})}


@paper_prompt_stage("GateC")
def admit_metadata(planner: Any, metadata: dict, retry_note: str = '') -> dict:
    incomplete = [t for t in metadata.get("transitions", []) if (t.get("validation") or {}).get("status") in {"call_failed", "invalid"}]
    if incomplete:
        raise ValueError("Metadata construction incomplete: failed/invalid transition is not no change")
    if structured_evidence_profile():
        frames = metadata.get("keyframes", [])
        if not frames or any((f.get("visual_state") or {}).get("validation_status") != "contract_valid" for f in frames):
            raise ValueError("Metadata construction incomplete: missing structured frame observations")
        if any(not (t.get("validation") or {}).get("entity_comparison") for t in metadata.get("transitions", [])):
            raise ValueError("Metadata construction incomplete: missing cross-frame entity comparison")
        states = {}
        for frame in frames:
            observed = frame["visual_state"]
            bound = validate_frame_observation(observed, frame["frame_id"], frame["img"])
            if observed.get("image_sha256") != bound["image_sha256"]:
                raise ValueError("Metadata construction incomplete: observation image hash changed")
            states[frame["frame_id"]] = bound
        for transition in metadata.get("transitions", []):
            if transition.get("source") not in states or transition.get("target") not in states:
                raise ValueError("Metadata construction incomplete: unknown transition endpoint")
            check = validate_pair_observation(transition["validation"]["raw_response"],
                                             states[transition["source"]], states[transition["target"]])
            if check["status"] not in {"valid", "no_change"} or check["text"] != transition.get("text"):
                raise ValueError("Metadata construction incomplete: entity comparison no longer agrees with transition")
            if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
                from orchestrator.independent_observation import digest
                frozen = transition['validation'].get('identity_checkpoint')
                actual = check['entity_comparison']
                if (not isinstance(frozen, dict)
                        or digest(frozen) != transition['validation'].get('identity_binding_sha256')
                        or {key: actual.get(key) for key in frozen} != frozen):
                    raise ValueError('Metadata construction incomplete: frozen identity changed')
    profile = current_prompt_profile()
    prompt, binding = metadata_review_request(metadata, profile)
    if retry_note:
        # The initial Gate C prompt is intentionally rich, but it can sit very
        # close to a local VLM's context limit.  A retry is a format repair,
        # not a second full semantic review, so send a compact claim projection
        # while keeping the same image bindings and metadata content digest.
        prompt = _compact_gate_c_retry_prompt(metadata, profile, retry_note)
        binding['final_prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    raw = audited_vision(
        planner,
        [f["img"] for f in metadata["keyframes"]],
        prompt,
        # A recovery request is deliberately concise.  The first request
        # keeps the historical budget; the retry leaves room for a compact
        # JSON object instead of allowing a long explanation to be truncated.
        max_new_tokens=(512 if retry_note else (
            3200 if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}
            else 1100
        )),
    )
    identity = dict(getattr(getattr(planner, "vlm", None), "last_attestation", {}) or {})
    result = parse_llm_json(raw)
    if not isinstance(result, dict) or result.get("decision") not in {"accept", "reject"}:
        raise GateCResponseError(
            "Gate C returned invalid JSON; API/parsing failure is not metadata rejection",
            code="invalid_json_or_decision",
            raw_response=raw,
            metadata_id=metadata.get("metadata_id"),
            review_binding=binding,
            model_identity=identity,
        )
    reasons = result.get("primary_reasons")
    # Reason-count guidance is a prompt preference; preserve every returned reason.
    if (not isinstance(reasons, list) or not reasons or
            any(not isinstance(r, str) or not r.strip() for r in reasons)):
        raise GateCResponseError(
            "Gate C requires a nonempty list of nonblank reason strings",
            code="invalid_reasons",
            raw_response=raw,
            metadata_id=metadata.get("metadata_id"),
            review_binding=binding,
            model_identity=identity,
        )
    if not isinstance(result.get("summary_consistent"), bool):
        raise GateCResponseError(
            "Gate C requires boolean summary_consistent",
            code="invalid_summary_consistency",
            raw_response=raw,
            metadata_id=metadata.get("metadata_id"),
            review_binding=binding,
            model_identity=identity,
        )
    if current_prompt_profile() in {"paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
        result["raw_model_decision"] = result["decision"]
    if not result["summary_consistent"]:
        result["decision"] = "reject"
    if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
        validate_metadata_claim_checks(result, metadata)
    return {**result, "raw_response": raw, "metadata_id": metadata["metadata_id"],
            "review_binding": {**binding, "requested_model": identity.get("requested_model"),
                               "response_model": identity.get("response_model")}}


def _compact_gate_c_retry_prompt(metadata: dict, profile: str, retry_note: str) -> str:
    """Build a context-safe contract-repair prompt for a Gate C retry."""
    video = metadata.get("video") if isinstance(metadata.get("video"), dict) else {}
    compact_frames = []
    for frame in metadata.get("keyframes", []) or []:
        if not isinstance(frame, dict):
            continue
        observations = frame.get("caption_validation", {}).get("observations", [])
        compact_frames.append({
            "frame_id": frame.get("frame_id"),
            "segment_sec": frame.get("segment_sec"),
            "caption": str(frame.get("caption") or "")[:420],
            "observations": [str(item)[:180] for item in observations[:4]],
        })
    compact_transitions = []
    for transition in metadata.get("transitions", []) or []:
        if not isinstance(transition, dict):
            continue
        compact_transitions.append({
            "transition_id": transition.get("transition_id"),
            "source": transition.get("source"),
            "target": transition.get("target"),
            "text": str(transition.get("text") or "")[:300],
        })
    payload = {
        "metadata_id": metadata.get("metadata_id"),
        "video": {
            "video_type": video.get("video_type"),
            "policy_domain": video.get("policy_domain"),
            "summary": str(video.get("summary") or "")[:700],
            "summary_scope": video.get("summary_scope"),
        },
        "keyframes": compact_frames,
        "transitions": compact_transitions,
    }
    required = "decision, primary_reasons, summary_consistent"
    if profile in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v14", "paper_v15"}:
        required += ", claim_checks"
    if profile in {"paper_v13", "paper_v14", "paper_v15"}:
        required += ", omissions"
    return (
        "You are repairing only the output format of a previous Gate C review. "
        "The images and claims are unchanged; do not reconsider them and do not "
        "turn uncertainty into acceptance. Return exactly one compact JSON object "
        "and no markdown, preamble, or trailing text. Required keys: " + required + ".\n"
        + retry_note + "\n"
        + "Current compact claim projection:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def admit_metadata_with_recovery(planner: Any, metadata: dict, *, max_attempts: int = 2) -> dict:
    """Admit metadata while isolating model response-contract failures.

    The first call uses the normal paper prompt.  A malformed or truncated
    response gets one concise contract-only retry.  If that also fails, the
    candidate is returned as ``hold`` with bounded provenance so callers can
    exclude it at distribution time without silently treating a transport or
    parser failure as a quality rejection.
    """
    attempts = max(1, int(max_attempts))
    errors: list[GateCResponseError] = []
    retry_note = ""
    for attempt in range(attempts):
        try:
            gate = admit_metadata(planner, metadata, retry_note=retry_note)
            if errors:
                gate["recovery"] = {
                    "attempts": len(errors) + 1,
                    "retried": True,
                    "initial_errors": [error.audit_payload() for error in errors],
                }
            return gate
        except GateCResponseError as exc:
            errors.append(exc)
            if attempt + 1 >= attempts:
                break
            retry_note = (
                "Output exactly one compact JSON object and nothing else. "
                "Use only the keys decision, primary_reasons, summary_consistent. "
                "decision must be accept or reject; primary_reasons must contain "
                "2-4 strings of at most 12 words each; summary_consistent must be a JSON boolean. "
                "Do not use markdown fences, explanations, or repeated metadata."
            )

    last = errors[-1]
    return {
        "decision": "hold",
        "status": "execution_pending",
        "raw_model_decision": None,
        "primary_reasons": [
            "Gate C response contract remained invalid after the configured retry; candidate held for review.",
        ],
        "summary_consistent": False,
        "metadata_id": metadata.get("metadata_id"),
        "review_binding": last.review_binding,
        "gate_c_error": {
            "attempts": len(errors),
            "errors": [error.audit_payload() for error in errors],
        },
        "recovery": {"attempts": len(errors), "retried": len(errors) > 1, "final_status": "hold"},
    }


def _canonical_claims(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical_claims(row) for row in value]
    if isinstance(value, dict):
        return {key: _canonical_claims(row) for key, row in value.items()
                if key not in {"img", "image_path", "source_file", "segment_path", "runtime"}}
    return value


def metadata_review_request(metadata: dict, profile: str) -> tuple[str, dict]:
    """Bind current claims and image bytes independently of the clip identifier."""
    if profile in {'paper_v11', 'paper_v13', 'paper_v14', 'paper_v15'}:
        from orchestrator.observation_patch import require_task_role_coverage
        for frame in metadata.get('keyframes', []):
            require_task_role_coverage(frame)
    if profile in {"paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
        for frame in metadata.get("keyframes", []):
            validate_patched_observation(frame, frame["frame_id"])
    projected = metadata_for_judging(metadata)
    images = [{"frame_id": frame["frame_id"],
               "sha256": hashlib.sha256(Path(frame["img"]).read_bytes()).hexdigest()}
              for frame in metadata.get("keyframes", [])]
    claims = {"schema_version": metadata.get("schema_version"),
              "claims": _canonical_claims(projected), "images": images}
    digest = hashlib.sha256(json.dumps(claims, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    prompt = render_prompt_text("paper", "gate_c", profile=profile,
                                metadata_json=json.dumps(projected, ensure_ascii=False))
    return prompt, {"metadata_content_sha256": digest, "image_bindings": images,
                    "prompt_profile": profile,
                    "review_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def require_metadata_review_binding(gate: dict, metadata: dict, profile: str,
                                    requested_model: str | None = None) -> None:
    _, expected = metadata_review_request(metadata, profile)
    binding = gate.get("review_binding")
    if not isinstance(binding, dict):
        raise ValueError("Gate C lacks metadata content binding; re-review current claims")
    if gate.get("metadata_id") != metadata.get("metadata_id"):
        raise ValueError("Gate C metadata identity changed")
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Gate C metadata content, image or prompt binding changed")
    if requested_model is not None and binding.get("requested_model") != requested_model:
        raise ValueError("Gate C requested model changed")


def validate_metadata_claim_checks(result: dict, metadata: dict) -> None:
    expected = [frame["frame_id"] for frame in metadata.get("keyframes", [])]
    expected += [pair["transition_id"] for pair in metadata.get("transitions", [])]
    checks = result.get("claim_checks")
    fields = ("entities_supported", "relations_supported", "direction_supported", "uncertainty_preserved")
    if current_prompt_profile() in {"paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
        fields += ("entities_complete",)
    if not isinstance(checks, list) or [row.get("id") for row in checks if isinstance(row, dict)] != expected:
        raise ValueError("Gate C lacks complete ordered frame/transition claim checks")
    for row in checks:
        if any(not isinstance(row.get(field), bool) for field in fields) or not str(row.get("reason", "")).strip():
            raise ValueError("Gate C has invalid claim check")
    if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'}:
        omissions = result.get('omissions')
        if not isinstance(omissions, list):
            raise ValueError('Gate C requires explicit visible-omission findings')
        for row in omissions:
            if (not isinstance(row, dict) or row.get('id') not in expected
                    or not all(isinstance(row.get(key), str) and row[key].strip()
                               for key in ['entity_id', 'missing_visible_change', 'reason'])):
                raise ValueError('Gate C invalid visible-omission finding')
            check = next(check for check in checks if check['id'] == row['id'])
            if check['entities_complete']:
                raise ValueError('Gate C omission conflicts with entities_complete:' + row['id'])
        if omissions:
            result['decision'] = 'reject'
    if result["decision"] == "accept" and any(not all(row[field] for field in fields) for row in checks):
        result["decision"] = "reject"


def metadata_for_judging(metadata: dict) -> dict:
    """Keep verified claims distinct from retained pre-verification hypotheses."""
    frames = []
    for frame in metadata.get("keyframes", []):
        verified = frame.get("caption_validation") or {}
        frames.append({k: frame.get(k) for k in ["frame_id", "img", "caption", "segment_sec", "source_sec"]} |
                      {"observations": verified.get("observations", []), "uncertainties": verified.get("uncertainties", []),
                       "visual_state": frame.get("visual_state"),
                       "retained_tool_signals": frame.get("semantic_profile", {}).get("tool_descs", {})})
    return {"metadata_id": metadata.get("metadata_id"), "video": metadata.get("video", {}),
            "summary_hierarchy": metadata.get("summary_hierarchy", {}), "keyframes": frames,
            "transitions": [{k: t.get(k) for k in ["transition_id", "source", "target", "text", "cause"]} |
                            {"entity_comparison": (t.get("validation") or {}).get("entity_comparison"),
                             "observed_position_differences": (t.get("validation") or {}).get("observed_position_differences", [])}
                            | ({"position_diagnostics": t["validation"]["position_diagnostics"],
                                "diagnostic_policy": "Grid/primary-anchor label changes require pixel review; they do not establish physical motion."}
                               if "position_diagnostics" in (t.get("validation") or {}) else {})
                            for t in metadata.get("transitions", [])],
            "provenance_note": "Original captions, pre-verification T1 summaries and raw tool/quality outputs are retained separately as hypotheses; they are not claims in this reviewed package."}
