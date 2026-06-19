# Fostride HITL Pipeline

Human-in-the-loop waste detection pipeline using FastAPI, RF-DETR, Label Studio, and active retraining.

## Pipeline

```text
Bin images
→ FastAPI server
→ RF-DETR prediction
→ Label Studio pre-annotation
→ Human correction
→ Export reviewed annotations
→ Build dataset
→ Train candidate model
→ Replace active model if better
```

fostride-pipeline/
  app/
    main.py
    config.py
    model_predict.py
    labelstudio_utils.py

  scripts/
    retrain_pipeline.py
    ls_to_yolo_dataset.py
    create_next_dataset.py
    yolo_to_coco.py
    train_and_replace.py
    hitl_auto_pipeline.py

  local_storage/
    raw_images/
      bin1/
      bin2/

  datasets/
    dataset_v1/
    dataset_v2/
    reviewed_batch/

  models/
    active/
      best.pth
    archive/
    candidate_nmodel_v2/

  requirements.txt

Setup
1. Create virtual environment
cd fostride-pipeline
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Start Label Studio
label-studio

Open:
http://localhost:8080

Create a project with these labels:
metal
paper
plastic

Label Studio config:

<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <Label value="metal"/>
    <Label value="paper"/>
    <Label value="plastic"/>
  </RectangleLabels>
</View>

Get:
Project ID
Access Token

Update app/config.py:
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_PROJECT_ID = 10
LABEL_STUDIO_TOKEN = "YOUR_LABEL_STUDIO_TOKEN"

SERVER_BASE_URL = "http://localhost:8000"
MODEL_PATH = BASE_DIR / "models" / "active" / "best.pth"

Required Files

Before running, make sure active model exists:

models/active/best.pth

Raw bin images should be placed like this:

local_storage/raw_images/bin1/
local_storage/raw_images/bin2/
local_storage/raw_images/bin3/

Clean training dataset should exist at:

datasets/dataset_v1/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
Run FastAPI Server
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Check server:

http://localhost:8000
Upload One Image
curl -X POST "http://localhost:8000/upload-bin-image" \
  -F "bin_id=bin1" \
  -F "location=office" \
  -F "image=@./local_storage/raw_images/bin1/image.jpg"
Upload Full Bin Folder

Folder must exist:

local_storage/raw_images/bin2/

Run:

curl -X POST "http://localhost:8000/upload-bin-folder" \
  -F "folder_name=bin2" \
  -F "location=office"

This will:

load images from bin2
→ run RF-DETR
→ create Label Studio tasks
→ show pre-annotations in Label Studio
Human Annotation

Open Label Studio:

http://localhost:8080

For each task:

check model prediction
fix wrong boxes/classes
add missing boxes
submit annotation

Only submitted annotations are used for training.

Run Full HITL Retraining Pipeline

After at least 100 new reviewed images:

source venv/bin/activate
python scripts/hitl_auto_pipeline.py

This script will:

export reviewed Label Studio annotations
filter already-trained tasks
use 100 new reviewed tasks
convert Label Studio JSON to YOLO
create dataset_v2
convert YOLO to COCO
train candidate RF-DETR model
compare candidate with active model
replace active model only if better
archive old model
mark tasks as processed
Important Output Files
labelstudio_exports/reviewed_latest.json
datasets/reviewed_batch/
datasets/dataset_v2/
models/candidate_nmodel_v2/
models/active/best.pth
models/archive/
training_history.json
model_registry.json

## Required External Files

This GitHub repo does not include raw images, datasets, or trained model files.

Before running the project, download/copy these folders manually:

```text
models/active/best.pth

datasets/dataset_v1/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels

local_storage/raw_images/
  bin1/
  bin2/
  bin3/
