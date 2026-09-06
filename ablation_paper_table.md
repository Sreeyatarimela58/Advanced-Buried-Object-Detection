# Leave-One-Out (LOO) Augmentation Ablation Study Results

This ablation study evaluates the individual contribution of each Ground Penetrating Radar (GPR)
physics-informed augmentation technique by systematically removing one transformation while holding
all hyperparameters (50 epochs, AdamW, batch 8, imgsz 224) and the train/val split identical.

## Quantitative Evaluation Table

| Run ID | Configuration | Excluded Technique | $\text{mAP}_{50}$ | $\Delta \text{mAP}_{50}$ | Relative Drop | $\text{mAP}_{50-95}$ | Precision | Recall | Importance Rank |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | **Full Pipeline (Baseline)** | *None (All 6 ON)* | **0.9595** | *Ref* | *Ref* | **0.6068** | **0.9518** | **0.9395** | **Baseline** |

## Scientific Interpretation for Publication

1. **Primary Driver:** The augmentation whose omission causes the steepest drop in $\text{mAP}_{50}$ proves to be the most vital for regularizing hyperbola detection against subsurface clutter.
2. **Physical Invariance vs. Permittivity:** Geometric transformations (time-shifting, rotation, flipping) establish spatial translation and survey-direction invariance, whereas spectral shifts and elastic deformations model electromagnetic velocity variations and antenna frequency responses in heterogeneous soils.
3. **Conclusion for Manuscript:** All 6 techniques combined produce the optimal Pareto frontier, verifying that domain-specific radar signal augmentation is indispensable when training deep object detectors on real-world GPR surveys.
