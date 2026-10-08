from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Finding(BaseModel):
    label: str
    score: float = Field(ge=0)
    confidence: Literal["low", "medium", "high"] = "low"
    note: str = "Model score only; not a diagnosis."


class RiskSignal(BaseModel):
    model_name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    risk_level: Literal["low", "moderate", "high", "unknown"]
    score: Optional[float] = Field(default=None, ge=0, le=1)


class HealthReport(BaseModel):
    source: str = Field(default="vitalis", min_length=1, max_length=80)
    overall_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = Field(default=None, max_length=5000)
    risk_signals: List[RiskSignal] = Field(default_factory=list, max_length=20)


class XrayRequest(BaseModel):
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    image_base64: str = Field(min_length=1, max_length=16_000_000)
    health_report: Optional[HealthReport] = None


class XrayAnalysis(BaseModel):
    image_type: Literal["chest_xray"] = "chest_xray"
    image_quality: Literal["adequate"] = "adequate"
    status: Literal["requires_clinical_review"] = "requires_clinical_review"
    model: str
    device: Literal["cpu", "cuda"]
    model_scores: Dict[str, float]
    highest_scoring_labels: List[Finding]
    integrated_summary: str
    limitations: List[str]
    clinical_review_required: bool = True
    disclaimer: str = "Research prototype only. This is not a diagnosis, radiology report, or medical advice. A qualified clinician must review the original study and clinical context."
