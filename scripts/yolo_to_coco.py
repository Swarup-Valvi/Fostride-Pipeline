import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "dataset_v2"

CLASS_NAMES = {
    0: "metal",
    1: "paper",
    2: "plastic"
}

SPLITS = ["train", "valid", "test"]


def convert_split(split):
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    output_json = DATASET_DIR / split / "_annotations.coco.json"

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "waste"}
            for class_id, name in CLASS_NAMES.items()
        ]
    }

    image_id = 1
    annotation_id = 1

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(images_dir.glob(ext))

    for image_path in sorted(image_files):
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": f"images/{image_path.name}",
            "width": img_width,
            "height": img_height
        })

        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    print(f"Skipping bad label line in {label_path}: {line}")
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                x = (x_center - box_width / 2) * img_width
                y = (y_center - box_height / 2) * img_height
                width = box_width * img_width
                height = box_height * img_height

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"{split}: saved {output_json}")
    print(f"  images: {len(coco['images'])}")
    print(f"  annotations: {len(coco['annotations'])}")


def main():
    for split in SPLITS:
        convert_split(split)


if __name__ == "__main__":
    main()