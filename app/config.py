from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LOCAL_STORAGE_DIR = BASE_DIR / "local_storage"
RAW_IMAGES_DIR = LOCAL_STORAGE_DIR / "raw_images"

MODEL_PATH = BASE_DIR / "models" / "active" / "best.pth"

SERVER_BASE_URL="http://localhost:8000"

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_PROJECT_ID = 10
LABEL_STUDIO_TOKEN = "6eb6ce63c0d1e27948e73912a0816c620ff11aa6"

CLASS_NAMES = {
    0: "metal",
    1: "paper",
    2: "plastic"
}