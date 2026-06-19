import json
import shutil
import random
from pathlib import Path
from urllib.parse import urlparse, unquote

BASE_DIR = Path(__file__).resolve().parent.parent

EXPORT_JSON = BASE_DIR / "labelstudio_exports" / "reviewed_latest.json"
OUTPUT_DATASET = BASE_DIR / "datasets" / "reviewed_batch"

CLASS_MAP = {
    "metal": 0,
    "paper": 1,
    "plastic": 2
}

SPLIT_RATIO = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1
}


def image_url_to_local_path(image_url):
    parsed = urlparse(image_url)
    path = unquote(parsed.path)

    # converts /media/raw_images/bin_001/img.jpg
    # to local_storage/raw_images/bin_001/img.jpg
    if path.startswith("/media/"):
        relative_path = path.replace("/media/", "")
        return BASE_DIR / "local_storage" / relative_path

    raise ValueError(f"Unsupported image path: {image_url}")


def ls_box_to_yolo(value):
    x = value["x"]
    y = value["y"]
    w = value["width"]
    h = value["height"]

    x_center = (x + w / 2) / 100
    y_center = (y + h / 2) / 100
    width = w / 100
    height = h / 100

    return x_center, y_center, width, height


def prepare_output_dirs():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in ["train", "valid", "test"]:
        (OUTPUT_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)


def main():
    prepare_output_dirs()

    with open(EXPORT_JSON, "r") as f:
        tasks = json.load(f)

    reviewed_items = []

    for task in tasks:
        annotations = task.get("annotations", [])

        if not annotations:
            continue

        annotation = annotations[-1]

        if annotation.get("was_cancelled"):
            continue

        results = annotation.get("result", [])

        if not results:
            continue

        image_url = task["data"]["image"]
        image_path = image_url_to_local_path(image_url)

        if not image_path.exists():
            print(f"Missing image, skipped: {image_path}")
            continue

        labels = []

        for result in results:
            if result.get("type") != "rectanglelabels":
                continue

            value = result["value"]
            label_name = value["rectanglelabels"][0]

            if label_name not in CLASS_MAP:
                print(f"Unknown class skipped: {label_name}")
                continue

            class_id = CLASS_MAP[label_name]
            x_center, y_center, width, height = ls_box_to_yolo(value)

            labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if labels:
            reviewed_items.append((image_path, labels))

    random.shuffle(reviewed_items)

    total = len(reviewed_items)
    train_end = int(total * SPLIT_RATIO["train"])
    valid_end = train_end + int(total * SPLIT_RATIO["valid"])

    split_items = {
        "train": reviewed_items[:train_end],
        "valid": reviewed_items[train_end:valid_end],
        "test": reviewed_items[valid_end:]
    }

    for split, items in split_items.items():
        for image_path, labels in items:
            image_out = OUTPUT_DATASET / split / "images" / image_path.name
            label_out = OUTPUT_DATASET / split / "labels" / f"{image_path.stem}.txt"

            shutil.copy(image_path, image_out)

            with open(label_out, "w") as f:
                f.write("\n".join(labels) + "\n")

    data_yaml = OUTPUT_DATASET / "data.yaml"
    data_yaml.write_text(
        f"""path: {OUTPUT_DATASET}
train: train/images
val: valid/images
test: test/images

names:
  0: metal
  1: paper
  2: plastic
"""
    )

    print(f"Reviewed images converted: {total}")
    print(f"Dataset created at: {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()