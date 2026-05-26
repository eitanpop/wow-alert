from ultralytics import YOLO
import os
import json

# Adjust these paths
MODEL_PATH = "D:/wow_rl_addon/models/model1/runs/detect/train-3/weights/best.pt"
INPUT_DIR = "D:/wow_rl_addon/recordings/frames/triumvate"
OUTPUT_JSON = "D:/wow_rl_addon/yolo/seat_prelabels.json"
CONFIDENCE = 0.25  # lower than default to catch more cast bars

model = YOLO(MODEL_PATH)
results = model.predict(source=INPUT_DIR, conf=CONFIDENCE, save=False, verbose=False)

tasks = []
for r in results:
    image_path = r.path
    filename = os.path.basename(image_path)
    img_width = r.orig_shape[1]
    img_height = r.orig_shape[0]
    
    predictions = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_pct = (x1 / img_width) * 100
        y_pct = (y1 / img_height) * 100
        width_pct = ((x2 - x1) / img_width) * 100
        height_pct = ((y2 - y1) / img_height) * 100
        
        predictions.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": width_pct,
                "height": height_pct,
                "rectanglelabels": ["cast_bar"]
            }
        })
    
    task = {
        "data": {
            "image": f"/data/local-files/?d=triumvate/{filename}"
        },
        "predictions": [{
            "model_version": "yolo_v1",
            "result": predictions
        }]
    }
    tasks.append(task)

with open(OUTPUT_JSON, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Generated pre-labels for {len(tasks)} frames")
print(f"Saved to {OUTPUT_JSON}")