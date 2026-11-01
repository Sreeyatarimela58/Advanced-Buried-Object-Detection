# PAPER_CONTEXT.md: facts about the paper and the current code

## Paper
Title: "GPR-Domain-Informed Augmentation and Lightweight YOLOv8 Nano for Underground Object Detection" (IEEEtran conference format, SRM University AP).

Goal: detect buried objects (utilities vs cavities) in 224x224 GPR B-scan patches with a lightweight detector (YOLOv8 Nano) trained on a small dataset using GPR-motivated augmentation.

## Existing code in the repo
`crop.py` (patch extraction), `augmentation.py`, `prepare_dataset.py`, `train.py`, `app.py` (Streamlit). Trained weights: `/models/best.pt`.

## Data
- Source dataset: GPR B-scans from infrastructure surveys in Morocco (2019-2024), 200 MHz and 400 MHz antennas (Mojahid et al., Data in Brief 2024; DOI 10.17632/ww7fd9t325.1).
- 285 original images: 79 cavity, 131 utility, 75 intact/background (intact has no boxes).
- Labels: YOLO normalized format. Class 0 = utility, class 1 = cavity.
- Annotation tool used: Make Sense (makesense.ai).

## Six offline augmentations (parameters)
| Technique | Parameter | Notes |
|---|---|---|
| Gaussian noise | std 0.15 on normalized intensities | |
| Time shift | ±20 px, circular (wrap-around) | box x-center shifts with wrap |
| Rotation | uniform in [-15°, +15°], reflect boundary | boxes need re-computation |
| Horizontal flip | p = 0.5 | x_center -> 1 - x_center |
| Elastic deformation | alpha = 34, sigma = 4 | Gaussian-smoothed random displacement field (Simard 2003). The paper wrongly says "thin-plate spline": verify in `augmentation.py` and correct the wording. |
| FFT spectral shift | roll by 100 bins in 2D FFT, take real part of inverse | check which axis(es) the code shifts |

Augmentation produced 2,239 images (553 cavity, 786 utility, 900 intact); 1,339 object-containing images were used for YOLO.

## Training config (keep identical unless told)
YOLOv8n, 50 epochs, imgsz 224, batch 8, AdamW, lr0 0.001, lrf 0.01, early-stopping patience 20, CPU (8 hours). YOLOv8 online augmentation settings were not logged: log them this time.

## Current results (LEAKY, do not present as final)
Augmentation was done BEFORE the 80/20 split (1,070 train / 269 val), so augmented copies of the same source image appear in both. Treat these as "augmented-split" numbers only.

Overall: mAP50 0.9595, mAP50-95 0.6068, P 0.952, R 0.939, F1 0.9456.
Class-wise mAP50: utility 0.9463, cavity 0.9726.

Leave-one-out (old split), mAP50 / delta:
Full 0.9595 | no noise 0.9812 (+0.0217) | no time shift 0.9583 (-0.0012) | no rotation 0.9513 (-0.0082) | no flip 0.9651 (+0.0056) | no elastic 0.9487 (-0.0108) | no FFT 0.9371 (-0.0224)

CPU latency (old): total mean 49.62 ms, std 51.59 ms (suspicious: std > mean), median 36.85 ms, P95 107.56 ms, 20.2 FPS. Model: 3.01 M params, ~6.2 MB FP32. Needs a clean re-run.

## Known problems to fix
1. Augmentation-before-split leakage (main issue).
2. Elastic deformation described as thin-plate spline but implemented as a Gaussian-smoothed displacement field.
3. Paper contains internal-note paragraphs: "Note on baseline mAP50 precision", the confusion-matrix "not available in this session" paragraph, and "author-supplied measurements" wording. Replace with real results.
4. Latency std larger than mean.
5. Only one dataset, one region, two antenna frequencies; no baseline comparison; no confusion matrix.

## Mentor's revision list (8 items)
1. More contributions + equations in methodology
2. Literature review table of recent studies (IEEE refs), YOLOv8 Nano significance, neater implementation section
3. Neat methodology diagrams (YOLOv8n architecture + project pipeline)
4. Test on another dataset (testing and comparison)
5. Compare against other methods (YOLOv4 vs YOLOv8 vs others)
6. More qualitative and quantitative evaluation
7. Plagiarism check
8. Collect GPR datasets and papers for buried-object detection

The agent's job covers items 4, 5, 6 (experiments) and the diagrams in item 3 (optional section in TASK.md). Items 1, 2, 7, 8 are written by co-authors.
