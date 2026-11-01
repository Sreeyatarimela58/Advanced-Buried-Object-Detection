"""experiments/config.py: single source of truth. Edit paths here only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ORIGINAL = ROOT / "datasets" / "original"      # 285 un-augmented images + YOLO labels (images/, labels/)
DATA_SPLIT = ROOT / "datasets" / "split_new"        # created by 01_split_source_level.py
DATA_EXTERNAL = None                                # set after the user provides the path
RESULTS = ROOT / "results"

SEEDS = [0, 1, 2]
SPLIT_RATIOS = (0.70, 0.15, 0.15)                   # train / val / test
SPLIT_SEED = 42

CLASS_NAMES = {0: "utility", 1: "cavity"}           # intact images have empty label files

TRAIN = dict(                                       # same for ALL models unless told otherwise
    imgsz=224, epochs=50, batch=8,
    optimizer="AdamW", lr0=0.001, lrf=0.01, patience=20,
)
FINETUNE = dict(epochs=25, lr0=0.0003)              # external-dataset Test B (confirm with user)

LATENCY = dict(batch=1, warmup=20, runs=100, threads=None)  # record the thread count actually used

AUGMENTATIONS = {                                   # parameters as in the paper
    "gaussian_noise": dict(std=0.15),
    "time_shift": dict(max_px=20),
    "rotation": dict(max_deg=15),
    "h_flip": dict(p=0.5),
    "elastic": dict(alpha=34, sigma=4),
    "fft_shift": dict(bins=100),
}
