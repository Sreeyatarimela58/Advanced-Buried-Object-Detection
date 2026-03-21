# -*- coding: utf-8 -*-
"""
Train YOLOv8 for Multi-Class GPR Buried Object Detection.

Usage:
    # Default (recommended for CPU):
    python train.py

    # Custom settings:
    python train.py --model yolov8n.pt --epochs 50 --batch 4 --imgsz 224

    # Resume training from a checkpoint:
    python train.py --resume runs/detect/gpr_multiclass/weights/last.pt
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for GPR buried object detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model to start from (default: yolov8n.pt — fastest, best for CPU)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to data.yaml config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50 for CPU)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8 — safe for CPU with 16GB RAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size for training (default: 224 — matches dataset)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="gpr_multiclass",
        help="Name for this training run",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20 epochs)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2 for CPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading pretrained model: {args.model}")
        model = YOLO(args.model)

    # Train
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:     CPU")
    print(f"  Patience:   {args.patience}")
    print("=" * 50 + "\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        device="cpu",
        workers=args.workers,
        # Augmentation settings (YOLO's built-in augmentations)
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation
        hsv_v=0.4,       # Value augmentation
        degrees=10.0,    # Rotation ±10°
        translate=0.1,   # Translation ±10%
        scale=0.3,       # Scale ±30%
        fliplr=0.5,      # Horizontal flip 50%
        flipud=0.0,      # No vertical flip (GPR images are orientation-sensitive)
        mosaic=0.5,      # Mosaic augmentation (reduced — dataset is small)
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,       # Slightly lower LR for fine-tuning
        lrf=0.01,        # Final LR = lr0 * lrf
        weight_decay=0.0005,
        # Save settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        verbose=True,
    )

    # Validation
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    best_model_path = f"runs/detect/{args.name}/weights/best.pt"
    best_model = YOLO(best_model_path)
    metrics = best_model.val(data=args.data, device="cpu")

    print(f"\n  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")

    # Per-class results
    class_names = ["utility", "cavity"]
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
    print("  " + "-" * 44)
    for i, name in enumerate(class_names):
        if i < len(metrics.box.p):
            print(
                f"  {name:<12} {metrics.box.p[i]:>10.4f} "
                f"{metrics.box.r[i]:>10.4f} {metrics.box.ap50[i]:>10.4f}"
            )

    print(f"\n  Best model saved to: {best_model_path}")
    print(f"  Training plots at:   runs/detect/{args.name}/")
    print(f"\n  To run the Streamlit app:")
    print(f"    streamlit run app.py")


if __name__ == "__main__":
    main()
