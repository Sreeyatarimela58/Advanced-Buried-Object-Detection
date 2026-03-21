# Multi-Class YOLO Buried Object Detection Pipeline

Build a complete pipeline: data preparation → YOLOv8 multi-class training → Streamlit evaluation UI. The model detects GPR hyperbolas and classifies them as **Utility** (class 0) or **Cavity** (class 1).

## Proposed Changes

### Data Preparation

#### [NEW] [prepare_dataset.py](file:///d:/Advanced-Buried-Object-Detection/prepare_dataset.py)

Script that converts the current `GPR_data/` layout into YOLO's required directory structure with an 80/20 train/val split.

**Key logic:**
- Copies augmented utility images → `yolo_dataset/images/train|val/`, labels from `YOLO_format/` → `yolo_dataset/labels/train|val/` with **class 0 preserved**
- Copies augmented cavity images → same dirs, but **relabels class 0 → class 1** in label files
- Intact images are **excluded** (no annotations; including them as background negatives would need empty [.txt](file:///d:/Advanced-Buried-Object-Detection/requirements.txt) files — we can add this later if mAP is low)
- Prefixes filenames with `util_` / `cav_` to avoid name collisions
- Prints summary statistics (image/label counts per split)
- Uses `random.seed(42)` for reproducibility

---

#### [NEW] [data.yaml](file:///d:/Advanced-Buried-Object-Detection/data.yaml)

```yaml
path: ./yolo_dataset
train: images/train
val: images/val
nc: 2
names: ['utility', 'cavity']
```

---

### Training

#### [NEW] [train.py](file:///d:/Advanced-Buried-Object-Detection/train.py)

Configurable training script using `ultralytics` Python API.

**Features:**
- Parses CLI arguments: `--model` (default `yolov8s.pt`), `--epochs` (default 150), `--batch` (default 16), `--imgsz` (default 224), `--name` (default `gpr_multiclass`)
- Trains with `patience=50` for early stopping
- After training, runs validation and prints mAP50, mAP50-95, per-class precision/recall
- Saves training results to `runs/detect/<name>/`

---

### Streamlit Evaluation Interface

#### [NEW] [app.py](file:///d:/Advanced-Buried-Object-Detection/app.py)

A Streamlit web app for interactive model evaluation.

**UI Design:**
- **Sidebar**: Model path selector (file uploader or text input defaulting to `runs/detect/gpr_multiclass/weights/best.pt`), confidence threshold slider (0.0–1.0, default 0.25)
- **Main area**:
  - Image uploader (accepts JPG/PNG)
  - Side-by-side display: original image | annotated image with bounding boxes
  - Detection results table: class name, confidence, bounding box coordinates
  - Color coding: **Utility = blue boxes**, **Cavity = red boxes**
  - Summary stats: total detections, count per class
- Uses `ultralytics.YOLO` for inference, OpenCV for drawing boxes, Streamlit columns for layout

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///d:/Advanced-Buried-Object-Detection/requirements.txt)

Add:
```
ultralytics
streamlit
Pillow
```

---

## Verification Plan

### Automated Checks

1. **Data preparation** — Run `python prepare_dataset.py` and verify:
   ```bash
   # Check directory structure exists
   ls yolo_dataset/images/train yolo_dataset/images/val yolo_dataset/labels/train yolo_dataset/labels/val
   # Check image/label counts match
   # Check a cavity label file contains class 1 (not class 0)
   ```

2. **Training sanity check** — Run 2-epoch training to verify pipeline doesn't crash:
   ```bash
   python train.py --epochs 2 --batch 8 --name sanity_check
   ```

3. **Streamlit launch test** — Start the app and verify it loads:
   ```bash
   streamlit run app.py
   ```
   Then use the browser to upload a sample GPR image and confirm detections render.

### Manual Verification
- After full training (~150 epochs), validate mAP50 > 0.5 on the val set
- User tests Streamlit UI with sample GPR images to confirm bounding boxes and class labels are correct
