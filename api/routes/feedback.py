import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger("uvicorn")
router = APIRouter(tags=["feedback"])
FEEDBACK_LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "feedback_log.jsonl"


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback_data: FeedbackRequest):
    """
    Record clinician verification or correction for a diagnosis.
    """
    try:
        predicted_class = feedback_data.predicted_class.strip()
        correct_class = feedback_data.correct_class.strip() if feedback_data.correct_class else None

        if not predicted_class:
            raise HTTPException(status_code=400, detail="Predicted class cannot be empty.")
        if not feedback_data.is_correct and not correct_class:
            raise HTTPException(status_code=422, detail="correct_class is required when is_correct is false.")
        if feedback_data.is_correct and correct_class is None:
            correct_class = predicted_class

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_class": predicted_class,
            "correct_class": correct_class,
            "is_correct": feedback_data.is_correct,
            "notes": feedback_data.notes,
            "has_embedding": feedback_data.query_embedding is not None,
        }

        # Ensure directory exists and append record
        FEEDBACK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            f"Clinician feedback recorded for predicted class '{predicted_class}' "
            f"(Verified correct: {feedback_data.is_correct})"
        )

        return FeedbackResponse(
            status="success",
            message="Feedback successfully recorded for model evaluation.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.") from e
