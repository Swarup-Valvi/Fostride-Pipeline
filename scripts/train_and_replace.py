import shutil
from pathlib import Path
from rfdetr import RFDETRSmall

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "dataset_v2"
ACTIVE_MODEL = BASE_DIR / "models" / "active" / "best.pth"
CANDIDATE_DIR = BASE_DIR / "models" / "candidate_nmodel_v2"

EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LR = 1e-4


def main():
    if CANDIDATE_DIR.exists():
        shutil.rmtree(CANDIDATE_DIR)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRSmall(pretrain_weights=str(ACTIVE_MODEL))

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        output_dir=str(CANDIDATE_DIR),
    )

    print("Candidate training completed.")
    print(f"Candidate saved at: {CANDIDATE_DIR}")


if __name__ == "__main__":
    main()