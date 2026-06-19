import json
import shutil
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_PROJECT_ID = 10
LABEL_STUDIO_TOKEN = "6eb6ce63c0d1e27948e73912a0816c620ff11aa6"

MIN_REVIEWED_IMAGES = 100

EXPORT_DIR = BASE_DIR / "labelstudio_exports"
OLD_DATASET = BASE_DIR / "datasets" / "dataset_v1"
NEW_DATASET = BASE_DIR / "datasets" / "dataset_v2"


def get_reviewed_tasks():
    url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/export"
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tasks = response.json()

    reviewed = [
        task for task in tasks
        if task.get("annotations") and len(task["annotations"]) > 0
    ]

    return reviewed


def save_export(tasks):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    export_path = EXPORT_DIR / "reviewed_latest.json"

    with open(export_path, "w") as f:
        json.dump(tasks, f, indent=2)

    return export_path


def main():
    reviewed_tasks = get_reviewed_tasks()

    print(f"Reviewed images found: {len(reviewed_tasks)}")

    if len(reviewed_tasks) < MIN_REVIEWED_IMAGES:
        print(f"Need {MIN_REVIEWED_IMAGES}. Not retraining yet.")
        return

    export_path = save_export(reviewed_tasks)
    print(f"Export saved to: {export_path}")

    print("Next step: convert Label Studio export to dataset_v2")


if __name__ == "__main__":
    main()