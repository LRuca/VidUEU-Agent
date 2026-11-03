from types import SimpleNamespace

from orchestrator.iteration.metadata_audit import audit_domain


def _planner(attestation=None):
    return SimpleNamespace(vlm=SimpleNamespace(last_attestation=attestation or {"model": "Qwen3.5-9B"}))


def test_metadata_audit_no_images_is_explicit_and_attested():
    result = audit_domain(_planner(), [], "segment summary", "generic")
    assert result["domain_audit_status"] == "no_images"
    assert result["policy_domain"] is None
    assert result["domain_source"].endswith("_unavailable")
    assert result["domain_audit_model_identity"]["model"] == "Qwen3.5-9B"


def test_metadata_audit_unknown_preserves_reason_and_prompt_hash(monkeypatch):
    monkeypatch.setattr(
        "orchestrator.prompt_audit.audited_vision",
        lambda planner, images, prompt, max_new_tokens: '{"video_type":"other","policy_domain":"unknown","confidence":0.0,"evidence":[]}',
    )
    result = audit_domain(_planner(), ["frame.jpg"], "segment summary", "generic")
    assert result["domain_audit_status"] == "unknown"
    assert result["policy_domain"] is None
    assert result["domain_source"] == "qwen3.5_metadata_audit"
    assert result["domain_audit_prompt_sha256"]


def test_metadata_audit_invalid_json_is_not_silent(monkeypatch):
    monkeypatch.setattr(
        "orchestrator.prompt_audit.audited_vision",
        lambda planner, images, prompt, max_new_tokens: "not-json",
    )
    result = audit_domain(_planner(), ["frame.jpg"], "segment summary", "generic")
    assert result["domain_audit_status"] == "invalid_json"
    assert result["domain_source"].endswith("_invalid")
    assert result["domain_confidence"] == 0.0
