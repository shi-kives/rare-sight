import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.utils import load_model
from api.routes import diagnose, feedback, add_class
from src.continual.replay_buffer import ReplayBuffer

logger = logging.getLogger("uvicorn")

# ---------------------------------------------------------------------
# Configuration — adjust these to match your actual project layout
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_BACKBONE = os.getenv("MODEL_BACKBONE", "resnet18")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / f"best_model_{MODEL_BACKBONE}.pth"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
FISHER_PATH = Path(os.getenv("FISHER_PATH", str(PROJECT_ROOT / "models" / "fisher_diagonal.pt")))
ANCHOR_PATH = Path(os.getenv("ANCHOR_PATH", str(PROJECT_ROOT / "models" / "anchor_weights.pth")))
REPLAY_BUFFER_PATH = Path(os.getenv("REPLAY_BUFFER_PATH", str(PROJECT_ROOT / "data" / "replay_buffer.pkl")))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))

MODEL_CONFIG = {
    "backbone": MODEL_BACKBONE,
    "embedding_dim": int(os.getenv("EMBEDDING_DIM", "128")),
}

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]

STATIC_DIR = PROJECT_ROOT / "data" / "processed" / "support_images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading backbone={MODEL_CONFIG['backbone']} from {MODEL_PATH}")

    # 1. Load the trained embedding model
    try:
        model = load_model(str(MODEL_PATH), MODEL_CONFIG, device)
        model_loaded = True
    except Exception as e:
        logger.exception(f"Failed to load model from {MODEL_PATH}: {e}")
        model = None
        model_loaded = False

    # 2. Load EWC regularization tensors (Fisher diagonal + anchor weights)
    fisher, anchor_weights, ewc_loaded = None, None, False
    try:
        fisher = torch.load(FISHER_PATH, map_location=device)
        anchor_weights = torch.load(ANCHOR_PATH, map_location=device)
        ewc_loaded = True
        logger.info("EWC Fisher diagonal and anchor weights loaded")
    except Exception as e:
        logger.warning(f"EWC assets not loaded — EWC fine-tuning will be unavailable: {e}")

    # 3. Connect to ChromaDB
    chroma_client, prototypes_collection, support_images_collection = None, None, None
    chroma_connected = False
    try:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        prototypes_collection = chroma_client.get_or_create_collection("prototypes", metadata={"hnsw:space": "l2"})
        support_images_collection = chroma_client.get_or_create_collection("support_images", metadata={"hnsw:space": "l2"})
        chroma_connected = True
        logger.info(f"Connected to ChromaDB at {CHROMA_PERSIST_DIR}")
    except Exception as e:
        logger.exception(f"Failed to connect to ChromaDB: {e}")

    # 4. Load the replay buffer (used for EWC rehearsal during fine-tuning)
    replay_buffer = ReplayBuffer()
    try:
        if REPLAY_BUFFER_PATH.exists():
            replay_buffer = ReplayBuffer.load(str(REPLAY_BUFFER_PATH))
            logger.info(f"Replay buffer loaded: {len(replay_buffer)} classes")
        else:
            logger.warning(f"No replay buffer found at {REPLAY_BUFFER_PATH}; starting empty")
    except Exception as e:
        logger.exception(f"Could not load replay buffer, starting empty: {e}")
        replay_buffer = ReplayBuffer()

    # Store everything on app.state so route handlers can access it
    app.state.device = device
    app.state.model = model
    app.state.model_loaded = model_loaded
    app.state.model_config = MODEL_CONFIG
    app.state.fisher = fisher
    app.state.anchor_weights = anchor_weights
    app.state.ewc_loaded = ewc_loaded
    app.state.chroma_client = chroma_client
    app.state.prototypes_collection = prototypes_collection
    app.state.support_images_collection = support_images_collection
    app.state.chroma_connected = chroma_connected
    app.state.replay_buffer = replay_buffer
    app.state.replay_buffer_path = str(REPLAY_BUFFER_PATH)

    yield  # ---- app serves requests here ----

    logger.info("Shutting down")


app = FastAPI(title="RareSight Diagnosis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static support images directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(diagnose.router)
app.include_router(feedback.router)
app.include_router(add_class.router)


@app.get("/health")
async def health():
    """Reports model load status and ChromaDB collection counts."""
    collections = {}
    chroma_connected = getattr(app.state, "chroma_connected", False)
    model_loaded = getattr(app.state, "model_loaded", False)
    ewc_loaded = getattr(app.state, "ewc_loaded", False)
    replay_buffer = getattr(app.state, "replay_buffer", None)
    device = getattr(app.state, "device", torch.device("cpu"))

    if chroma_connected:
        try:
            proto_col = getattr(app.state, "prototypes_collection", None)
            support_col = getattr(app.state, "support_images_collection", None)
            if proto_col is not None:
                collections["prototypes"] = proto_col.count()
            if support_col is not None:
                collections["support_images"] = support_col.count()
        except Exception as e:
            collections["error"] = str(e)

    healthy = model_loaded and chroma_connected and collections.get("prototypes", 0) > 0
    return {
        "status": "ok" if healthy else "degraded",
        "model_loaded": model_loaded,
        "backbone": MODEL_CONFIG["backbone"],
        "embedding_dim": MODEL_CONFIG["embedding_dim"],
        "ewc_loaded": ewc_loaded,
        "chroma_connected": chroma_connected,
        "chroma_collections": collections,
        "replay_buffer_classes": len(replay_buffer) if replay_buffer is not None else 0,
        "device": str(device),
    }
