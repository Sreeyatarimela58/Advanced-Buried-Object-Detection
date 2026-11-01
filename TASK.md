# TASK.md: experiments for the paper revision

Read `AGENTS.md` (rules, including ask-before-acting) and `PAPER_CONTEXT.md` (facts) first.
Work phase by phase. After each phase: stop, summarize (what was done, files created, key numbers), and ask before the next phase.

## Open inputs (ask me for any that are missing, do not guess)
- External dataset path (needs bounding-box labels): `<ASK ME>`
- Whether the 285 images carry antenna frequency (200 or 400 MHz) in file names or metadata: `<ASK ME>`
- GPU available? (auto-detect; if none, tell me the estimated time and propose a smaller plan)

---

## Active Protocol Override (User Directives - Overrides conflicting items below)
- **Grouping & Validation Strategy:** For Phases 2, 3, and 5: use the same grouped 5-fold split as Phase 1 (`results/split_new/split_5fold.json`), 1 seed per fold, same pretrained start and same settings for every model, with mean ± std over folds and per-fold values. Do not use the old 70/15/15 split or 3 seeds anywhere.
- **Phase 2 Comparison Metrics:** For every model in the comparison table, record: mAP50, mAP50-95, P, R, F1, params (M), size (MB), CPU latency (mean/median/P95 ms), GFLOPs, training time, GPU name, and FPS (derived from median latency: 1000 / median_ms).
- **Phase 5 Figures & Ablations:**
  - Skip Grad-CAM heatmaps, per-image false-positive tables, and `errors.csv`.
  - For the Leave-One-Out (LOO) ablation, use the same 5 folds and the same augmentation suffix mapping (util: aug_1..aug_6, cavity: aug_7..aug_12).
  - Keep strictly to the metrics already listed, nothing extra.

---

## Phase 0: Setup
- `git tag v1-paper-draft`, then `git checkout -b paper-revision`.
- Create:
```
experiments/
  config.py                    # single source of truth (starter provided)
  utils.py
  01_split_source_level.py
  02_train_all_models.py
  03_eval_external.py
  04_frequency_domain_shift.py
  05_make_figures.py
results/
  split_new/ baseline_new_split/ model_comparison/ external_dataset/ freq_shift/ figures/ tables/
datasets/                      # gitignored
```
- Add `datasets/`, `runs/`, `*.pt` (except `/models/best.pt`) to `.gitignore`.
- Install from `requirements-experiments.txt`; after a working install, record exact versions with `pip freeze > requirements-experiments.lock.txt`. Old code used Ultralytics 8.0; YOLO11 needs a newer version. Ask me before upgrading if anything conflicts.

## Phase 1: Source-level split + honest baseline (`01_train_5fold_baseline.py`) [COMPLETED]
1. Read the 285 ORIGINAL (non-augmented) images and labels. Stratify by class (utility / cavity / intact).
2. Grouped 5-fold split saved to `results/split_new/split_5fold.json`.
3. Apply existing augmentation pipeline to TRAIN images only for each fold; val and test sets are 100% un-augmented.
4. Train YOLOv8n across all 5 folds. Evaluate on each test fold.
5. Save: mAP50, mAP50-95, P, R, F1, per-class AP50, confusion matrices, PR and F1 curves.
6. Output `results/baseline_new_split/summary.csv`, `.json`, `pooled_confusion_matrix.png`, and `results/tables/baseline_table.tex`.

## Phase 2: Model comparison (`02_train_all_models.py`)
All models on the Phase 1 grouped 5-fold split (`split_5fold.json`), same augmented train sets, 224 px, 50 epochs, 1 seed per fold (seed 0), same pretrained start and same settings.
- Must: YOLOv8n (reused from Phase 1), YOLOv8s, YOLOv5nu, YOLO11n (Ultralytics).
- Must: Faster R-CNN (torchvision MobileNetV3-FPN or ResNet50-FPN, COCO-pretrained, fine-tuned). Write a YOLO-to-torchvision label converter. Compute mAP50 and mAP50-95 so they are comparable with Ultralytics.
- Should: YOLOv4-tiny. If infeasible, ask before skipping and write reason in `results/model_comparison/NOTES.md`.
- Two YOLOv8n ablations across 5 folds: (a) no augmentation, (b) generic augmentation only (flip + rotation + brightness, no GPR-specific ones).
- Per model record: mAP50, mAP50-95, P, R, F1, params (M), size (MB), CPU latency (ms mean/median/P95; 224x224, batch 1, 20 warm-up + 100 timed runs, fixed thread count), GFLOPs, training time, GPU name, and FPS (from median latency). All reported as mean ± std over folds plus per-fold values.
- Output `results/model_comparison/comparison.csv` and `results/tables/comparison_table.tex` (IEEEtran style).

## Phase 3: Another dataset (`03_eval_external.py`)
First inspect the dataset and print: image count, whether bounding-box labels exist, label format, classes, official train/valid/test split. If there are NO box labels, STOP and ask me.
- Use grouped 5-fold split (`split_5fold.json`), 1 seed per fold.
- Class mapping: if external dataset has single class, merge utility+cavity into one class for this evaluation only.
- Test A (zero-shot): run Phase 1 YOLOv8n models on external test split with NO training. Same metrics.
- Test B (fine-tuned): fine-tune on external train split, evaluate on external test split.
- Output `results/external_dataset/table.csv` and `NOTES.md`.

## Phase 4: Frequency domain-shift test (`04_frequency_domain_shift.py`)
- Use the 200/400 MHz information (ask me if it is not available).
- Train on 200 MHz, test on 400 MHz, and reverse with YOLOv8n.
- Report mAP50 and per-class results next to in-domain result.

## Phase 5: Evaluation and figures (`05_make_figures.py`)
Save 300-dpi PNG and PDF in `results/figures/`.
- Confusion matrix (raw and normalized), PR and F1-confidence curves.
- Bar chart of mAP50 and mAP50-95 for all models with error bars; params-vs-mAP50 and latency-vs-mAP50 scatter.
- Per-class AP50 and AP75 table.
- Qualitative grid (ground truth vs prediction): 6 correct, 4 false positives, 4 misses.
- Same 6 images predicted by YOLOv8n, YOLOv5nu, YOLOv8s, Faster R-CNN side by side.
- Leave-one-out (LOO) ablation re-run on the 5 folds: 7 configurations across the same 5 folds using the existing augmentation suffix mapping (`util: aug_1..aug_6`, `cavity: aug_7..aug_12`), mean ± std over folds, delta vs baseline. Mark deltas smaller than one std as "not significant".
- (Skipped per user directive: Grad-CAM heatmaps, per-image false-positive tables, and `errors.csv`).
- Keep to the metrics already listed, nothing extra.

## Paper-side outputs (suggestions only, never edit the paper)
- LaTeX tables in `results/tables/` after each phase (IEEEtran style).
- `results/PAPER_UPDATES.md`: exact sentences/numbers in the paper that must change (abstract, contributions, Section V, conclusion), old value vs new value.
- Suggested replacement text for the internal-note paragraphs (see PAPER_CONTEXT.md, known problem 3).
- Check `augmentation.py` and propose corrected wording for the elastic deformation description (known problem 2).

## Optional (only if I ask)
- Diagrams: see `DIAGRAMS.md`; verify YOLOv8n layers against `model.info()` and `yolov8n.yaml` before finalizing.
- Literature table and GPR dataset/paper list: include only papers and datasets you can verify exist, with full citation and link. If you cannot verify one, leave it out.

## Definition of done
- Each script runs from a clean clone with one command; `README_experiments.md` explains the order.
- Every number in the new tables traces to a file in `results/`.
- `results/SUMMARY.md` contains: new honest baseline, model comparison table, external dataset table (Test A and B), frequency-shift result, LOO ablation on the new split, and a plain list of remaining limitations.
