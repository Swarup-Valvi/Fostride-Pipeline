import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OLD_DATASET = BASE_DIR / "datasets" / "dataset_v1"
NEW_BATCH = BASE_DIR / "datasets" / "reviewed_batch"
NEXT_DATASET = BASE_DIR / "datasets" / "dataset_v2"

SPLITS = ["train", "valid", "test"]


def copy_folder(src, dst):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)

    for file in src.iterdir():
        if file.is_file():
            target = dst / file.name

            # avoid overwrite if same filename exists
            if target.exists():
                target = dst / f"new_{file.name}"

            shutil.copy(file, target)


def main():
    if NEXT_DATASET.exists():
        shutil.rmtree(NEXT_DATASET)

    shutil.copytree(OLD_DATASET, NEXT_DATASET)

    for split in SPLITS:
        copy_folder(
            NEW_BATCH / split / "images",
            NEXT_DATASET / split / "images"
        )
        copy_folder(
            NEW_BATCH / split / "labels",
            NEXT_DATASET / split / "labels"
        )

    data_yaml = NEXT_DATASET / "data.yaml"
    data_yaml.write_text(
        f"""path: {NEXT_DATASET}
train: train/images
val: valid/images
test: test/images

names:
  0: metal
  1: paper
  2: plastic
"""
    )

    print("dataset_v2 created successfully")
    print(NEXT_DATASET)


if __name__ == "__main__":
    main()