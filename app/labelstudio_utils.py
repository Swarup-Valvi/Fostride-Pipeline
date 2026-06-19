import uuid
import requests

from app.config import (
    LABEL_STUDIO_URL,
    LABEL_STUDIO_PROJECT_ID,
    LABEL_STUDIO_TOKEN
)


def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(value, max_value))


def yolo_to_labelstudio_result(pred):
    """
    Converts YOLO normalized box:
    class_id x_center y_center width height

    into Label Studio percentage box:
    x y width height
    """

    x_center = pred["x_center"]
    y_center = pred["y_center"]
    width = pred["width"]
    height = pred["height"]

    x = (x_center - width / 2) * 100
    y = (y_center - height / 2) * 100

    return {
        "id": str(uuid.uuid4())[:8],
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "score": pred["score"],
        "original_width": pred["image_width"],
        "original_height": pred["image_height"],
        "image_rotation": 0,
        "value": {
            "x": clamp(x),
            "y": clamp(y),
            "width": clamp(width * 100),
            "height": clamp(height * 100),
            "rotation": 0,
            "rectanglelabels": [pred["label"]]
        }
    }


def create_labelstudio_task(
    image_url,
    predictions,
    image_id,
    bin_id,
    location,
    model_version="xmodel_v1"
):
    results = [yolo_to_labelstudio_result(p) for p in predictions]

    avg_score = 0
    if predictions:
        avg_score = sum(p["score"] for p in predictions) / len(predictions)

    task = {
        "data": {
            "image": image_url,
            "image_id": image_id,
            "bin_id": bin_id,
            "location": location
        },
        "predictions": [
            {
                "model_version": model_version,
                "score": avg_score,
                "result": results
            }
        ]
    }

    return task


def import_task_to_labelstudio(task):
    url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/import"

    headers = {
        "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json=[task]
    )

    response.raise_for_status()
    return response.json()