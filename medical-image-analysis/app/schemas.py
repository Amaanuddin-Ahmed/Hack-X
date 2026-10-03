from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class Finding(BaseModel):
    label: str = Field(description="Short, plain-language model observation.")
    confidence: Literal["low", "medium", "high"]
    note: Optional[str] = None


class RiskSignal(BaseModel):
    model_name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    risk_level: Literal["low", "moderate", "high", "unknown"]
    score: Optional[float] = Field(default=None, ge=0, le=1)
    summary: Optional[str] = Field(default=None, max_length=500)


class HealthModelSummary(BaseModel):
    source: str = Field(min_length=1, max_length=80, description="Producer of this summary, e.g. risk-model-service.")
    overall_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = Field(default=None, max_length=1200)
    risk_signals: List[RiskSignal] = Field(default_factory=list, max_length=20)

class MedicalScanAnalysis(BaseModel):
    request_id: str = Field(default="", description="Server-generated correlation ID.")
    image_type: Literal["chest_xray", "other", "unclear"]
    image_quality: Literal["insufficient", "limited", "adequate", "unclear"]
    status: Literal["requires_clinical_review", "not_a_chest_xray", "inconclusive"]
    model: str
    device: Literal["cpu", "cuda"]
    model_scores: Dict[str, float] = Field(description="Raw model outputs, not probabilities or diagnoses.")
    highest_scoring_labels: List[Finding] = Field(default_factory=list)
    observations: List[Finding] = Field(default_factory=list)
    health_model_summary: Optional[HealthModelSummary] = None
    integrated_summary: str = Field(default="No health-model summary was supplied with this image.")
    limitations: List[str] = Field(default_factory=list)
    clinical_review_required: bool = True
    disclaimer: str = "Research prototype only. This is not a diagnosis, radiology report, or medical advice. A qualified clinician must review the original study and clinical context."

class ErrorResponse(BaseModel):
    detail: str
