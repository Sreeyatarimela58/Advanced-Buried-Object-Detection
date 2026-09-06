# Model Computational Complexity & Latency Benchmark

| Metric | Measured Value | Academic Description |
| :--- | :---: | :--- |
| **Model Size** | `3.01 M` | Total trainable parameters in YOLOv8 Nano |
| **Input Resolution** | `224 × 224` | Patch size (compatible with CNN receptive fields) |
| **Pre-process Time** | `1.66 ms` | Resizing, color normalization, tensor casting |
| **Inference Time** | `46.60 ms` | Forward neural network backbone & head computation |
| **Post-process Time (NMS)** | `0.83 ms` | Non-Maximum Suppression and coordinate re-scaling |
| **Total Pipeline Latency** | **`49.62 ms`** | End-to-end latency per single GPR patch |
| **Throughput (FPS)** | **`20.2 FPS`** | Real-time processing speed on standard CPU |

### Publication Paragraph for Results Section:
> *"The proposed YOLOv8 Nano architecture consists of 3.01 million parameters. When evaluated on standard multi-core CPU hardware with 224×224 input patches, the model achieves an average end-to-end latency of 49.62 ms per B-scan patch (consisting of 1.66 ms pre-processing, 46.60 ms forward inference, and 0.83 ms post-processing). This corresponds to a real-time throughput of 20.2 frames per second (FPS), comfortably exceeding typical field cart survey speeds (1–5 FPS), thereby demonstrating feasibility for real-time edge deployment."*
