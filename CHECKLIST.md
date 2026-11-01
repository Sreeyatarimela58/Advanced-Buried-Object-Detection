# CHECKLIST.md: tick items as you finish them

## Phase 0: Setup
- [x] Tag `v1-paper-draft`, branch `paper-revsion`
- [x] Folder structure, `.gitignore`, requirements verified and scripts initialized
- [x] Open inputs answered/handled (external dataset on hold for Phase 3, frequency info on hold for Phase 4, GPU via Kaggle)


## Phase 1: Honest baseline
- [x] Source-level split saved (`split_5fold.json`), assert no overlap
- [x] Train-only augmentation, boxes verified visually
- [x] YOLOv8n grouped 5-fold cross-validation trained, test metrics saved
- [x] Confusion matrix, PR, F1 curves (per-fold and pooled)
- [x] Old (leaky) vs new (honest) comparison table
- [x] Phase summary sent, waiting for my OK

## Phase 2: Model comparison (Grouped 5-Fold, 1 seed/fold)
- [ ] YOLOv8s, YOLOv5nu, YOLO11n
- [ ] Faster R-CNN (+ metric definition note)
- [ ] YOLOv4-tiny (or documented decision with my approval)
- [ ] No-augmentation and generic-augmentation ablations (5 folds)
- [ ] Params, size, latency (mean/median/P95), GFLOPs, training time, GPU name, FPS, mean ± std across 5 folds
- [ ] `comparison.csv` + `comparison_table.tex`
- [ ] Phase summary sent

## Phase 3: External dataset (Grouped 5-Fold)
- [ ] Dataset inspected (labels, classes, split)
- [ ] Test A (zero-shot)
- [ ] Test B (fine-tuned, all models)
- [ ] `table.csv` + `NOTES.md`
- [ ] Phase summary sent

## Phase 4: Frequency shift
- [ ] 200->400 and 400->200 results
- [ ] Phase summary sent

## Phase 5: Figures and evaluation
- [ ] Bar/scatter plots, per-class AP50/AP75
- [ ] Qualitative grids, multi-model grid (Grad-CAM heatmaps skipped per directive)
- [ ] LOO ablation on 5 folds (7 configs, suffix mapping, significance marked)
- [ ] (Skipped per directive: per-image false-positive tables and errors.csv)
- [ ] Latency & FPS benchmark (median, P95)

## Wrap-up
- [ ] `results/PAPER_UPDATES.md`
- [ ] `results/SUMMARY.md`
- [ ] `README_experiments.md`
