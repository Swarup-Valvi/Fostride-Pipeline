import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import RAW_IMAGES_DIR, LOCAL_STORAGE_DIR, SERVER_BASE_URL
from app.model_predict import predict_image
from app.labelstudio_utils import create_labelstudio_task, import_task_to_labelstudio

app = FastAPI(title="Fostride Local Annotation Pipeline")

RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# CORS is useful because Label Studio runs on port 8080
# and this server runs on port 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local files through HTTP so Label Studio can load images
app.mount(
    "/media",
    StaticFiles(directory=str(LOCAL_STORAGE_DIR)),
    name="media"
)


@app.get("/")
def home():
    return {
        "message": "Fostride local server is running",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "server": "running",
        "model": "loaded",
        "storage": "local"
    }


@app.post("/upload-bin-image")
async def upload_bin_image(
    bin_id: str = Form(...),
    location: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1. Receives image from bin/local upload
    2. Saves image locally
    3. Runs pretrained model
    4. Creates Label Studio pre-annotation task
    5. Imports task into Label Studio
    """

    image_id = str(uuid.uuid4())

    original_filename = image.filename
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in [".jpg", ".jpeg", ".png"]:
        return {
            "status": "error",
            "message": "Only .jpg, .jpeg, and .png files are supported"
        }

    bin_folder = RAW_IMAGES_DIR / bin_id
    bin_folder.mkdir(parents=True, exist_ok=True)

    local_image_path = bin_folder / f"{image_id}{file_ext}"

    with open(local_image_path, "wb") as f:
        f.write(await image.read())

    # URL that Label Studio can open in browser
    image_url = f"{SERVER_BASE_URL}/media/raw_images/{bin_id}/{image_id}{file_ext}"

    predictions = predict_image(local_image_path)

    task = create_labelstudio_task(
        image_url=image_url,
        predictions=predictions,
        image_id=image_id,
        bin_id=bin_id,
        location=location,
        model_version="xmodel_v1"
    )

    labelstudio_response = import_task_to_labelstudio(task)

    return {
        "status": "success",
        "image_id": image_id,
        "bin_id": bin_id,
        "location": location,
        "saved_path": str(local_image_path),
        "image_url": image_url,
        "predictions_count": len(predictions),
        "labelstudio_response": labelstudio_response
    }


@app.get("/latest-model")
def latest_model():
    """
    Later the bin can call this endpoint to check which model to use.
    For now it returns local model info.
    """

    return {
        "model_version": "xmodel_v1",
        "model_type": "YOLO",
        "storage": "local",
        "status": "active"
    }
    
@app.post("/upload-bin-folder")
async def upload_bin_folder(
    folder_name: str = Form(...),
    location: str = Form(...)
):
    folder_path = RAW_IMAGES_DIR / folder_name

    if not folder_path.exists():
        return {
            "status": "error",
            "message": f"{folder_name} not found"
        }

    image_extensions = [".jpg", ".jpeg", ".png"]

    image_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    uploaded = []

    for image_path in image_files:

        image_id = image_path.stem

        image_url = (
    f"{SERVER_BASE_URL}/media/raw_images/"
    f"{folder_name}/{image_path.name}"
)

        predictions = predict_image(image_path)

        task = create_labelstudio_task(
            image_url=image_url,
            predictions=predictions,
            image_id=image_id,
            bin_id=folder_name,
            location=location,
            model_version="xmodel_v1"
        )

        import_task_to_labelstudio(task)

        uploaded.append(image_path.name)

    return {
        "status": "success",
        "folder": folder_name,
        "uploaded": len(uploaded),
        "files": uploaded
    }