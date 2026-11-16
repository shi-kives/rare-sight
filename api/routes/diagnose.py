import logging
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.schemas.models import DiagnoseResponse, GatesStatus, SimilarCase
from api.utils import embed_image, preprocess_image
from src.retrieval.retriever import run_inference

logger = logging.getLogger("uvicorn")
router = APIRouter(tags=["diagnose"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024
CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.55
MIN_CONFIDENCE_GATE = 0.50
MIN_MARGIN_GATE = 0.03
MAX_PROTOTYPE_DISTANCE_GATE = 3.0


def to_static_url(file_path: str) -> str:
    if not file_path:
        return ""

    path = file_path.replace("\\", "/")
    if path.startswith(("http://", "https://", "data:")):
        return path
    if path.startswith("/static/"):
        return path
    if "support_images/" in path:
        relative_path = path.split("support_images/", 1)[1]
        return f"/static/{relative_path.lstrip('/')}"
    return f"/static/{Path(path).as_posix().lstrip('/')}"


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    request: Request,
    file: UploadFile = File(...),
    modality: Optional[str] = Form("Histopathology"),
):
    """
    Diagnose an uploaded medical image using few-shot prototype retrieval.
    """
    try:
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image uploaded.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

        app_state = request.app.state
        device = getattr(app_state, "device", torch.device("cpu"))
        model = getattr(app_state, "model", None)
        model_loaded = getattr(app_state, "model_loaded", False)
        proto_col = getattr(app_state, "prototypes_collection", None)
        support_col = getattr(app_state, "support_images_collection", None)

        if not model_loaded or model is None:
            raise HTTPException(status_code=503, detail="Embedding model is unavailable.")

        try:
            tensor = preprocess_image(image_bytes)
            emb_np = embed_image(tensor, model, device)
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        effective_modality = "Histopathology" if not modality or modality.lower() in {"auto-detect", "auto", "path"} else modality

        # 2. Run Retrieval & Inference against ChromaDB
        if proto_col is None or support_col is None:
            raise HTTPException(status_code=503, detail="ChromaDB collections are unavailable.")
        if proto_col.count() == 0:
            raise HTTPException(status_code=503, detail="ChromaDB prototype index is empty.")

        inf = run_inference(
            query_emb_np=emb_np,
            proto_collection=proto_col,
            support_collection=support_col,
            n_similar=5,
        )

        prediction = str(inf["prediction"])
        confidence = float(inf["confidence"])
        margin = float(inf["margin"])
        min_dist = float(inf["min_distance"])
        all_scores = {str(key): float(value) for key, value in inf["all_class_scores"].items()}
        agrees = inf.get("agrees")
        if agrees is not None:
            agrees = bool(agrees)

        similar_cases = []
        for item in inf.get("similar_cases", []):
            similar_cases.append(
                SimilarCase(
                    id=item.get("filename") or item.get("class"),
                    image_path=to_static_url(item.get("image_path", "")),
                    similarity_score=float(item.get("similarity", 0.0)),
                    class_name=str(item.get("class", prediction)),
                    dataset=item.get("dataset", "Reference Index"),
                    modality=effective_modality,
                    confirmed=True,
                )
            )

        if confidence >= CONFIDENCE_HIGH_THRESHOLD:
            confidence_level = "HIGH"
        elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        gates = GatesStatus(
            similarity_threshold_passed=confidence >= MIN_CONFIDENCE_GATE,
            prototype_dispersion_passed=margin >= MIN_MARGIN_GATE,
            ood_status_passed=min_dist <= MAX_PROTOTYPE_DISTANCE_GATE,
        )

        return DiagnoseResponse(
            prediction=prediction,
            confidence_level=confidence_level,
            confidence=confidence,
            agrees=agrees,
            modality=effective_modality,
            embedding=emb_np.tolist(),
            gates=gates,
            similar_cases=similar_cases,
            all_class_scores=all_scores,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during diagnose inference: {e}")
        raise HTTPException(status_code=500, detail="Inference failed. Check the API logs for details.") from e
    finally:
        await file.close()
