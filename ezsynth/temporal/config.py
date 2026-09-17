"""Configuration for the reference-based temporal pipeline."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class TemporalConfig(BaseModel):
    enabled: bool = False
    flow_backend: Literal["legacy", "opencv", "precomputed"] = "legacy"
    precomputed_flow_dir: Optional[str] = None
    warp_device: Literal["auto", "cpu", "cuda"] = "auto"
    max_references: int = Field(3, ge=1, le=8)
    max_reference_distance: Optional[int] = Field(None, ge=1)
    scene_cuts: List[int] = Field(default_factory=list)
    detect_scene_cuts: bool = False
    scene_cut_threshold: float = Field(0.65, gt=0.0, le=1.0)
    fb_alpha: float = Field(0.01, gt=0.0)
    fb_beta: float = Field(0.5, gt=0.0)
    photo_scale: float = Field(0.15, gt=0.0)
    confidence_threshold: float = Field(0.2, ge=0.0, le=1.0)
    recommend_fraction: float = Field(0.15, ge=0.0, le=1.0)
    blend_disagreement: float = Field(0.12, ge=0.0, le=1.0)
    face_annotations: Optional[str] = None
    eye_state_tolerance: float = Field(0.25, gt=0.0, le=1.0)
    face_guide_weight: float = Field(8.0, ge=0.0)
    tracks_path: Optional[str] = None
    track_guide_weight: float = Field(5.0, ge=0.0)
    semantic_guide_dir: Optional[str] = None
    semantic_guide_weight: float = Field(2.0, ge=0.0)
    object_label_dir: Optional[str] = None
    object_guide_weight: float = Field(5.0, ge=0.0)
    missing_reference: Literal["warn", "error"] = "warn"
    seed: int = Field(0, ge=0)

    @validator("scene_cuts")
    def unique_cuts(cls, value):
        if any(i <= 0 for i in value) or len(set(value)) != len(value):
            raise ValueError("scene_cuts must contain unique positive frame indices")
        return sorted(value)

    class Config:
        extra = "forbid"
