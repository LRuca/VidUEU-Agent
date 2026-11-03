from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepresentativeFrame:
    """Compact record for a representative frame inside a shot."""

    frame_index: int
    sec: float
    img_path: str
    kind: str
    quality_score: float = 0.0
    change_score: float = 0.0
    text_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShotRecord:
    """Minimal shot-level profile used by T1/T2."""

    shot_id: str
    start_sec: float
    end_sec: float
    duration: float
    frame_indices: List[int] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    detector_backend: str = ""
    boundary_confidence_in: float = 0.0
    boundary_confidence_out: float = 0.0
    rep_frames: List[RepresentativeFrame] = field(default_factory=list)
    visual_embedding: Optional[List[float]] = None
    visual_novelty: float = 0.0
    motion_score: float = 0.0
    scene_type: str = ""
    visual_quality_hint: str = ""
    informativeness_hint: str = ""
    candidate_sampling_hint: str = ""
    text_density: float = 0.0
    speech_density: float = 0.0
    shot_role: str = ""
    vlm_summary: str = ""
    ocr_summary: str = ""
    asr_summary: str = ""
    object_tags: List[str] = field(default_factory=list)
    importance_hint: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["rep_frames"] = [x.to_dict() for x in self.rep_frames]
        return out


@dataclass
class ChapterRecord:
    """Continuous chapter built from multiple adjacent shots."""

    chapter_id: str
    shot_ids: List[str]
    start_sec: float
    end_sec: float
    duration: float
    chapter_title_hint: str = ""
    chapter_summary: str = ""
    chapter_role: str = ""
    key_topics: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    importance_hint: str = "medium"
    coverage_need: str = "required"
    t2_candidate_tool_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoProfile:
    """Video-level profile consumed by T2 planners."""

    video_id: str
    video_type: str = ""
    video_summary: str = ""
    global_topics: List[str] = field(default_factory=list)
    structure_type: str = ""
    selection_objective: str = ""
    global_sampling_hint: str = ""
    global_tool_hints: List[str] = field(default_factory=list)
    global_quota_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
