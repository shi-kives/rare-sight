from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    id: Optional[str] = None
    image_path: Optional[str] = None
    similarity_score: float
    class_name: str
    dataset: Optional[str] = "Reference Index"
    modality: Optional[str] = "Medical Image"
    confirmed: Optional[bool] = True


class GatesStatus(BaseModel):
    similarity_threshold_passed: bool = True
    prototype_dispersion_passed: bool = True
    ood_status_passed: bool = True


class DiagnoseResponse(BaseModel):
    prediction: str
    confidence_level: str = "LOW"
    confidence: float
    agrees: Optional[bool] = None
    modality: str
    embedding: List[float]
    gates: GatesStatus = Field(default_factory=GatesStatus)
    similar_cases: List[SimilarCase] = Field(default_factory=list)
    all_class_scores: Dict[str, float] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    query_embedding: Optional[List[float]] = None
    predicted_class: str
    correct_class: Optional[str] = None
    is_correct: bool
    notes: Optional[str] = ""


class FeedbackResponse(BaseModel):
    status: str = "success"
    message: str = "Feedback recorded successfully"


class AddClassResponse(BaseModel):
    status: str = "success"
    class_name: str
    support_images_added: int
    finetuned: bool = False
    message: str = "Class registered successfully"
    saved_to: Optional[str] = None
