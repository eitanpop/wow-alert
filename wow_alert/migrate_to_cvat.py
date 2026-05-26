"""
Migrate Label Studio annotations to a CVAT task.

Reads Label Studio JSON export, matches annotations to CVAT frames by filename,
and pushes them to CVAT via the SDK.
"""

import json
import sys
from cvat_sdk import make_client
from cvat_sdk.api_client import models

# Config
LABEL_STUDIO_JSON = "D:/wow_rl_addon/label_studio_export.json"
CVAT_HOST = "localhost"
CVAT_PORT = 8080
CVAT_USERNAME = ""
CVAT_PASSWORD = ""
TASK_ID = 5
LABEL_NAME = "cast_bar"


def extract_filename_from_ls_task(ls_task):
    """Get the original image filename from a Label Studio task."""
    image_path = ls_task.get("data", {}).get("image", "")
    if not image_path:
        return None
    # Label Studio paths look like: /data/upload/3/01bb114c-frame_0183.jpg
    # We want just the basename
    return image_path.split("/")[-1]


def ls_annotation_to_cvat_shape(annotation, image_width, image_height, label_id):
    """Convert a Label Studio annotation (percentages) to a CVAT shape (pixels)."""
    value = annotation["value"]
    x_pct = value["x"]
    y_pct = value["y"]
    width_pct = value["width"]
    height_pct = value["height"]
    
    # Label Studio: x,y are top-left; width/height are box dimensions
    # All in percentages of original image
    x1 = (x_pct / 100.0) * image_width
    y1 = (y_pct / 100.0) * image_height
    x2 = x1 + (width_pct / 100.0) * image_width
    y2 = y1 + (height_pct / 100.0) * image_height
    
    return {
        "type": "rectangle",
        "label_id": label_id,
        "points": [x1, y1, x2, y2],
        "frame": None,  # set later
        "occluded": False,
        "outside": False,
        "attributes": [],
    }


def main():
    print(f"Loading Label Studio JSON from {LABEL_STUDIO_JSON}")
    with open(LABEL_STUDIO_JSON, "r", encoding="utf-8") as f:
        ls_tasks = json.load(f)
    print(f"  Loaded {len(ls_tasks)} Label Studio tasks")
    
    print(f"\nConnecting to CVAT at {CVAT_HOST}:{CVAT_PORT}...")
    with make_client(host=CVAT_HOST, port=CVAT_PORT, credentials=(CVAT_USERNAME, CVAT_PASSWORD)) as client:
        # Get the task
        task = client.tasks.retrieve(TASK_ID)
        print(f"  Connected to task: {task.name} (id={task.id})")
        print(f"  Task has {task.size} frames")
        
        # Get the task's labels - find the cast_bar label
        labels = task.get_labels()
        cast_bar_label = next((l for l in labels if l.name == LABEL_NAME), None)
        if not cast_bar_label:
            print(f"ERROR: Label '{LABEL_NAME}' not found in task. Available labels: {[l.name for l in labels]}")
            sys.exit(1)
        print(f"  Found label '{LABEL_NAME}' (id={cast_bar_label.id})")
        
        # Get frame info - filename to frame index map
        meta = task.get_meta()
        frame_map = {frame.name: idx for idx, frame in enumerate(meta.frames)}
        print(f"  Built frame map with {len(frame_map)} entries")
        print(f"  Sample frame names: {list(frame_map.keys())[:3]}")
        
        # Build CVAT shapes from Label Studio annotations
        shapes_to_create = []
        matched = 0
        skipped_no_frame = 0
        skipped_no_annotations = 0
        
        for ls_task in ls_tasks:
            filename = extract_filename_from_ls_task(ls_task)
            if not filename:
                skipped_no_annotations += 1
                continue
            
            # Find matching CVAT frame
            if filename not in frame_map:
                # Try without hash prefix in case CVAT stripped it
                stripped = filename.split("-", 1)[-1] if "-" in filename else filename
                if stripped in frame_map:
                    frame_idx = frame_map[stripped]
                else:
                    skipped_no_frame += 1
                    continue
            else:
                frame_idx = frame_map[filename]
            
            # Get image dimensions from CVAT
            frame_info = meta.frames[frame_idx]
            img_w = frame_info.width
            img_h = frame_info.height
            
            # Get annotations from Label Studio task
            annotations = ls_task.get("annotations", [])
            if not annotations:
                skipped_no_annotations += 1
                continue
            
            for ann_group in annotations:
                results = ann_group.get("result", [])
                for result in results:
                    if result.get("type") != "rectanglelabels":
                        continue
                    shape = ls_annotation_to_cvat_shape(result, img_w, img_h, cast_bar_label.id)
                    shape["frame"] = frame_idx
                    shapes_to_create.append(shape)
                    matched += 1
        
        print(f"\nMatched {matched} annotations across {len(ls_tasks)} tasks")
        print(f"Skipped {skipped_no_frame} tasks (no matching frame in CVAT)")
        print(f"Skipped {skipped_no_annotations} tasks (no annotations)")
        
        if not shapes_to_create:
            print("Nothing to upload.")
            return
        
        # Push to CVAT
        print(f"\nUploading {len(shapes_to_create)} annotations to CVAT...")
        
        # Build the annotations request
        labeled_data = models.LabeledDataRequest(
            shapes=[
                models.LabeledShapeRequest(
                    type=s["type"],
                    label_id=s["label_id"],
                    points=s["points"],
                    frame=s["frame"],
                    occluded=s["occluded"],
                    outside=s["outside"],
                    attributes=s["attributes"],
                )
                for s in shapes_to_create
            ],
            tags=[],
            tracks=[],
        )
        
        task.set_annotations(labeled_data)
        print("Done.")


if __name__ == "__main__":
    main()