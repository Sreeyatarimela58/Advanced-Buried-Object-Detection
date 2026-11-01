# PAPER_UPDATES.md: Recommended Paper Corrections and Revisions

This document tracks all required textual, numerical, and methodological updates for the research paper revision.

---

## 1. Methodology & Dataset Splitting

### Spatial Grouping Proxy
> **Old Text / Assumption:** Patches were treated as independent and randomly split across partitions.
>
> **Recommended Update:**  
> *"Due to the horizontal sliding-window patch extraction across continuous GPR B-scans, adjacent patches can exhibit spatial correlation. Because survey scan-line identifiers and spatial coordinates were not preserved in the dataset metadata, the 131 utility patches were partitioned using a conservative proxy based on consecutive patch indices: patches separated by a gap of $\le 3$ missing indices were merged into 11 distinct survey groups, with groups kept whole within each fold to prevent cross-partition leakage."*

### Use of Offline Augmented Imagery
> **Old Text / Assumption:** The original evaluation applied offline augmentations prior to dataset splitting, leaking augmented copies of test images into the training split.
>
> **Recommended Update:**  
> *"The revised experimental evaluation utilizes a grouped 5-fold cross-validation protocol. To preserve authentic ground truth bounding box annotations, the pre-computed offline augmented images and verified labels from the author's survey repository are employed for training folds only, rather than dynamically re-generating them with `augmentation.py` (which lacks bounding box coordinate transformation logic for rotation and elastic deformation). In each fold, validation and test sets strictly comprise 100% un-augmented original images. Every augmented image maps to exactly one source image, guaranteeing zero source-image or transect leakage into validation or test sets."*

### Limitations
> *"Cavities (79 profiles) and Intact background profiles (75 profiles) are partitioned as independent samples across the 5 cross-validation folds because continuous survey transect indices are absent from their file naming and metadata."*

---

## 2. Experimental Results & Performance Comparisons

| Evaluation Setup | mAP50 | mAP50-95 | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Augmented-split (Old, Leaky)** | 0.9595 | 0.6068 | 0.9520 | 0.9390 | 0.9456 |
| **Grouped 5-Fold (New, Honest)** | **0.7390 ± 0.0746** | **0.3309 ± 0.0348** | **0.7407 ± 0.0573** | **0.7054 ± 0.0561** | **0.7225 ± 0.0560** |

### Per-Fold Breakdown (YOLOv8n Baseline)
| Fold | mAP50 | mAP50-95 | Precision | Recall | F1-Score | Utility mAP50 | Cavity mAP50 |
|---|---|---|---|---|---|---|---|
| **Fold 1** | 0.6792 | 0.3273 | 0.7111 | 0.6771 | 0.6937 | 0.8353 | 0.5869 |
| **Fold 2** | 0.7962 | 0.3225 | 0.7412 | 0.7172 | 0.7290 | 0.7524 | 0.7301 |
| **Fold 3** | 0.8220 | 0.3552 | 0.8316 | 0.7983 | 0.8146 | 0.7550 | 0.9083 |
| **Fold 4** | 0.7504 | 0.3700 | 0.7418 | 0.6754 | 0.7070 | 0.6343 | 0.8492 |
| **Fold 5** | 0.6473 | 0.2795 | 0.6777 | 0.6592 | 0.6683 | 0.4863 | 0.8690 |
| **Mean ± Std** | **0.7390 ± 0.0746** | **0.3309 ± 0.0348** | **0.7407 ± 0.0573** | **0.7054 ± 0.0561** | **0.7225 ± 0.0560** | **0.6927 ± 0.1348** | **0.7887 ± 0.1309** |

*All per-fold metrics, evaluation plots, and pooled confusion matrices are persisted under `results/baseline_new_split/`.*

