# DIAGRAMS.md: Mermaid sources (optional task)
Paste into draw.io (Arrange > Insert > Advanced > Mermaid) or mermaid.live, export as SVG/PDF for the paper.
Verify the YOLOv8n layer list against `model.info()` and `yolov8n.yaml` before finalizing.

## Project pipeline
```mermaid
flowchart LR
  A[GPR B-scan survey<br/>200/400 MHz] --> B[Patch extraction<br/>224x224]
  B --> C[Annotation<br/>YOLO format]
  C --> S[Source-level split<br/>train / val / test]
  S -->|train only| AUG
  subgraph AUG[Domain-informed augmentation - train only]
    direction LR
    D1[Gaussian noise] --> D2[Time shift] --> D3[Rotation] --> D4[H-flip] --> D5[Elastic] --> D6[FFT shift]
  end
  AUG --> T[YOLOv8n training]
  S -->|val / test, not augmented| E
  T --> W[best.pt]
  W --> E[Evaluation: mAP, P, R, F1,<br/>ablation, comparison, external test]
  W --> APP[Streamlit app]
```

## YOLOv8n architecture (width 0.25, depth 0.33; head scales at 224 input: 28, 14, 7)
```mermaid
flowchart TB
  IN[Input 224x224x3] --> B1[Conv s2, 16] --> B2[Conv s2, 32] --> B3[C2f, 32]
  B3 --> B4[Conv s2, 64] --> B5[C2f x2, 64 - P3]
  B5 --> B6[Conv s2, 128] --> B7[C2f x2, 128 - P4]
  B7 --> B8[Conv s2, 256] --> B9[C2f, 256] --> B10[SPPF, 256 - P5]
  B10 --> N1[Upsample + Concat P4 + C2f]
  B7 --> N1
  N1 --> N2[Upsample + Concat P3 + C2f - out P3]
  B5 --> N2
  N2 --> N3[Conv s2 + Concat + C2f - out P4]
  N1 --> N3
  N3 --> N4[Conv s2 + Concat + C2f - out P5]
  B10 --> N4
  N2 --> H1[Decoupled head 28x28<br/>box DFL + cls]
  N3 --> H2[Decoupled head 14x14]
  N4 --> H3[Decoupled head 7x7]
  H1 --> OUT[NMS: utility / cavity]
  H2 --> OUT
  H3 --> OUT
```
