import io
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas.models import AddClassResponse
from src.retrieval.add_class import add_class_pure

logger = logging.getLogger("uvicorn")
router = APIRouter(tags=["add_class"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORT_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed" / "support_images"
CLASS_METADATA_FILE = Path(__file__).resolve().parents[2] / "data" / "class_metadata.json"


def save_class_description(class_name: str, description: Optional[str]):
    if not description or not description.strip():
        return
    CLASS_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if CLASS_METADATA_FILE.exists():
        with open(CLASS_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata[class_name] = {"description": description.strip(), "dataset": "UserDefined", "modality": "path"}
    temp_file = CLASS_METADATA_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    temp_file.replace(CLASS_METADATA_FILE)


def safe_class_directory(class_name: str) -> str:
    directory = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name.strip()).strip("_-")
    if not directory:
        raise HTTPException(status_code=400, detail="Class name must contain letters or numbers.")
    return directory[:80]


@router.post("/add_class", response_model=AddClassResponse)
async def add_new_class(
    request: Request,
    class_name: str = Form(...),
    description: Optional[str] = Form(None),
    modality: Optional[str] = Form("path"),
    images: List[UploadFile] = File(...),
):
    """
    Register a novel rare disease class with support reference images.
    """
    clean_name = class_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Class name cannot be empty.")

    if not images:
        raise HTTPException(status_code=400, detail="At least one reference image is required.")

    if len(images) > 50:
        raise HTTPException(status_code=400, detail="A maximum of 50 reference images can be uploaded at once.")

    modality_lower = (modality or "path").lower()
    if "path" not in modality_lower and "hist" not in modality_lower:
        raise HTTPException(status_code=400, detail="This model currently supports histopathology images only.")

    mod_tag = "path"
    dataset_tag = "UserDefined"
    directory_name = safe_class_directory(clean_name)
    target_root = SUPPORT_ROOT / mod_tag
    target_dir = target_root / directory_name

    app_state = request.app.state
    device = getattr(app_state, "device", torch.device("cpu"))
    model = getattr(app_state, "model", None)
    model_loaded = getattr(app_state, "model_loaded", False)
    proto_col = getattr(app_state, "prototypes_collection", None)
    support_col = getattr(app_state, "support_images_collection", None)
    replay_buf = getattr(app_state, "replay_buffer", None)

    if target_dir.exists():
        raise HTTPException(status_code=409, detail=f"Class '{clean_name}' already has a support-image directory.")

    target_root.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        # Stage uploads separately; add_class_pure copies them into target_dir.
        with tempfile.TemporaryDirectory(prefix="raresight_upload_") as temp_dir:
            temp_root = Path(temp_dir)

            for idx, upload in enumerate(images):
                suffix = Path(upload.filename or "image.jpg").suffix.lower() or ".jpg"
                if suffix not in ALLOWED_IMAGE_SUFFIXES:
                    raise HTTPException(status_code=400, detail=f"Unsupported image format: {suffix}")

                content = await upload.read(MAX_IMAGE_BYTES + 1)
                if not content:
                    raise HTTPException(status_code=400, detail=f"Uploaded image {idx + 1} is empty.")
                if len(content) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail=f"Each image must be 10 MB or smaller.")

                try:
                    with Image.open(io.BytesIO(content)) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    raise HTTPException(status_code=400, detail=f"Uploaded file {idx + 1} is not a valid image.") from e

                temp_path = temp_root / f"{directory_name}_{idx:04d}{suffix}"
                temp_path.write_bytes(content)
                saved_paths.append(str(temp_path))

            if model_loaded and model is not None and proto_col is not None and support_col is not None:
                if replay_buf is None or not hasattr(replay_buf, "add_class"):
                    raise HTTPException(status_code=503, detail="Replay buffer is unavailable.")

                result = add_class_pure(
                    class_name=directory_name,
                    image_paths=saved_paths,
                    dataset_tag=dataset_tag,
                    modality=mod_tag,
                    model=model,
                    proto_collection=proto_col,
                    support_collection=support_col,
                    replay_buffer=replay_buf,
                    save_dir=str(target_root),
                    device=device,
                )
                save_class_description(directory_name, description)

                return AddClassResponse(
                    status="success",
                    class_name=clean_name,
                    support_images_added=len(saved_paths),
                    finetuned=result.get("finetuned", False),
                    message=f"Class '{clean_name}' successfully added with {len(saved_paths)} support images.",
                    saved_to=str(Path("data/processed/support_images") / mod_tag / directory_name),
                )

            target_dir.mkdir(parents=True, exist_ok=False)
            for source_path in saved_paths:
                shutil.copy2(source_path, target_dir / Path(source_path).name)

            save_class_description(directory_name, description)
            logger.info(f"Class '{clean_name}' stored with {len(saved_paths)} images (standby mode).")
            return AddClassResponse(
                status="success",
                class_name=clean_name,
                support_images_added=len(saved_paths),
                finetuned=False,
                message=f"Class '{clean_name}' registered ({len(saved_paths)} reference images saved; model/index unavailable).",
                saved_to=str(Path("data/processed/support_images") / mod_tag / directory_name),
            )

    except HTTPException:
        raise
    except Exception as e:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        logger.exception(f"Failed to add class '{clean_name}': {e}")
        raise HTTPException(status_code=500, detail="Failed to register class. Check the API logs for details.") from e
    finally:
        for upload in images:
            await upload.close()
