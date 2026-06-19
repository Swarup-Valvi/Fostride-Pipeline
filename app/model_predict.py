from PIL import Image
from rfdetr import RFDETRSmall

from app.config import MODEL_PATH, CLASS_NAMES

model = RFDETRSmall(pretrain_weights=str(MODEL_PATH))


def predict_image(image_path, conf=0.35):
    """
    Runs RF-DETR model on uploaded image.
    Returns predictions in normalized YOLO-style format.
    """

    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    detections = model.predict(image, threshold=conf)

    predictions = []

    for xyxy, class_id, score in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        x1, y1, x2, y2 = xyxy

        box_width = x2 - x1
        box_height = y2 - y1

        x_center = x1 + box_width / 2
        y_center = y1 + box_height / 2

        predictions.append({
            "class_id": int(class_id),
            "label": CLASS_NAMES.get(int(class_id), str(class_id)),
            "score": float(score),
            "x_center": float(x_center / img_width),
            "y_center": float(y_center / img_height),
            "width": float(box_width / img_width),
            "height": float(box_height / img_height),
            "image_width": img_width,
            "image_height": img_height
        })

    return predictions