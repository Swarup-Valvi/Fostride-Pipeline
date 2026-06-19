import json
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

MIN_NEW_REVIEWED = 100

EXPORT_JSON = BASE_DIR / "labelstudio_exports" / "reviewed_latest.json"
HISTORY_FILE = BASE_DIR / "training_history.json"
REGISTRY_FILE = BASE_DIR / "model_registry.json"

DATASET_V1 = BASE_DIR / "datasets" / "dataset_v1"
DATASET_V2 = BASE_DIR / "datasets" / "dataset_v2"

ACTIVE_MODEL = BASE_DIR / "models" / "active" / "best.pth"
ARCHIVE_DIR = BASE_DIR / "models" / "archive"

CANDIDATE_DIR = BASE_DIR / "models" / "candidate_nmodel_v2"
CANDIDATE_MODEL = CANDIDATE_DIR / "checkpoint_best_ema.pth"
CANDIDATE_LOG = CANDIDATE_DIR / "log.txt"


def run(cmd):
    print("\nRUNNING:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_task_id(task):
    return str(task["id"])


def get_reviewed_tasks():
    with open(EXPORT_JSON, "r") as f:
        tasks = json.load(f)

    return [
        t for t in tasks
        if t.get("annotations") and len(t["annotations"]) > 0
    ]


def filter_new_tasks(tasks):
    history = load_json(HISTORY_FILE, {"processed_task_ids": []})
    processed = set(history["processed_task_ids"])

    new_tasks = [
        t for t in tasks
        if get_task_id(t) not in processed
    ]

    return new_tasks, history


def save_new_batch(tasks):
    with open(EXPORT_JSON, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Saved new batch with {len(tasks)} tasks.")


def parse_candidate_map():
    if not CANDIDATE_LOG.exists():
        print("No candidate log found. Cannot compare metrics.")
        return None

    text = CANDIDATE_LOG.read_text()

    matches = re.findall(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+([0-9.]+)",
        text
    )

    if not matches:
        print("Could not find AP50:95 in log.")
        return None

    return float(matches[-1])


def deploy_candidate(candidate_map, new_tasks):
    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(f"Candidate model not found: {CANDIDATE_MODEL}")

    registry = load_json(REGISTRY_FILE, {
        "active_map": None,
        "versions": []
    })

    active_map = registry.get("active_map")

    print(f"Active mAP: {active_map}")
    print(f"Candidate mAP: {candidate_map}")

    if active_map is not None and candidate_map <= active_map:
        print("Candidate is not better. Keeping old active model.")
        return False

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = ARCHIVE_DIR / f"active_backup_{timestamp}.pth"

    if ACTIVE_MODEL.exists():
        shutil.copy(ACTIVE_MODEL, backup_path)
        print(f"Old model archived: {backup_path}")

    shutil.copy(CANDIDATE_MODEL, ACTIVE_MODEL)
    print(f"Candidate promoted to active model: {ACTIVE_MODEL}")

    registry["active_map"] = candidate_map
    registry["versions"].append({
        "timestamp": timestamp,
        "active_model": str(ACTIVE_MODEL),
        "backup_model": str(backup_path),
        "candidate_model": str(CANDIDATE_MODEL),
        "candidate_map": candidate_map,
        "trained_task_count": len(new_tasks)
    })

    save_json(REGISTRY_FILE, registry)

    return True


def mark_tasks_processed(history, tasks):
    old_ids = set(history.get("processed_task_ids", []))
    new_ids = {get_task_id(t) for t in tasks}

    history["processed_task_ids"] = sorted(list(old_ids | new_ids))
    save_json(HISTORY_FILE, history)

    print(f"Marked {len(new_ids)} tasks as processed.")


def promote_dataset_v2_to_v1():
    if DATASET_V1.exists():
        shutil.rmtree(DATASET_V1)

    shutil.copytree(DATASET_V2, DATASET_V1)
    print("dataset_v2 promoted to dataset_v1.")


def main():
    print("STEP 1: Export reviewed annotations from Label Studio")
    run(["python", "scripts/retrain_pipeline.py"])

    reviewed_tasks = get_reviewed_tasks()
    new_tasks, history = filter_new_tasks(reviewed_tasks)

    print(f"Total reviewed tasks: {len(reviewed_tasks)}")
    print(f"New untrained reviewed tasks: {len(new_tasks)}")

    if len(new_tasks) < MIN_NEW_REVIEWED:
        print(f"Not enough new reviewed images. Need {MIN_NEW_REVIEWED}.")
        return

    selected_tasks = new_tasks[:MIN_NEW_REVIEWED]
    save_new_batch(selected_tasks)

    print("STEP 2: Convert Label Studio annotations to YOLO")
    run(["python", "scripts/ls_to_yolo_dataset.py"])

    print("STEP 3: Create dataset_v2")
    run(["python", "scripts/create_next_dataset.py"])

    print("STEP 4: Convert YOLO dataset to COCO")
    run(["python", "scripts/yolo_to_coco.py"])

    print("STEP 5: Train candidate model")
    run(["python", "scripts/train_and_replace.py"])

    print("STEP 6: Compare candidate with active model")
    candidate_map = parse_candidate_map()

    if candidate_map is None:
        print("Candidate metric missing. Not deploying.")
        return

    deployed = deploy_candidate(candidate_map, selected_tasks)

    if deployed:
        promote_dataset_v2_to_v1()

    mark_tasks_processed(history, selected_tasks)

    print("HITL automation completed.")


if __name__ == "__main__":
    main()