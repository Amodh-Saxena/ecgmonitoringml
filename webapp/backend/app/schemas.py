"""Pydantic schemas matching the PRD API contract exactly."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ClassResult(BaseModel):
    code: str
    name: str
    probability: float
    threshold: float
    flagged: bool


class SignalResult(BaseModel):
    fs: int = 100
    unit: str = "mV"
    leads: List[str]
    samples: List[List[float]]


class AttentionResult(BaseModel):
    method: str = "cam"
    per_class: List[List[float]]


class PredictResponse(BaseModel):
    schema_version: str = "1"
    model_version: str
    manifest_hash: str
    status: str  # "confident" | "uncertain" | "rejected"
    reasons: List[str] = Field(default_factory=list)
    quality_score: float
    leads_used: int
    classes: List[ClassResult] = Field(default_factory=list)
    signal: Optional[SignalResult] = None
    attention: Optional[AttentionResult] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    manifest_hash: str
    classes: List[dict]
    thresholds: List[float]
    temperatures: List[float]
    test_metrics: dict
    license: str
    intended_use: str
    limitations: List[str]


class ErrorResponse(BaseModel):
    code: str
    message: str
    field: Optional[str] = None
