# Multi-Class GPR Buried Object Detection: Complete Project Report

This document provides a comprehensive, end-to-end overview of the development, training, and evaluation of a deep learning model for detecting and classifying buried utilities and cavities from Ground Penetrating Radar (GPR) imagery.

---

## 1. Introduction & Raw Data

Ground Penetrating Radar (GPR) is a non-destructive method that uses electromagnetic radiation to image the subsurface. When the radar wave encounters a buried object such as a pipe or a void, it reflects back to the surface, forming a characteristic **hyperbolic signature** (an inverted "U" shape) in the resulting B-scan image.

### The Raw Dataset
The original data acquisition process utilized 400 MHz and 200 MHz antennas configurations. The initial datasets were huge, full-length survey scans that were subsequently cropped using `crop.py` into **224×224 pixel patches**. This size is highly optimized for ingestion into Convolutional Neural Networks (CNNs).

The raw, un-augmented dataset was critically small:
*   **Original Cavities:** 79 images
*   **Original Utilities:** 131 images
*   **Original Intact (Background):** 75 images
*   **Total:** 285 images

Training an object detection model like YOLO on just 210 positive examples would invariably lead to severe overfitting, where the model memorizes the training data but fails to detect utilities or cavities in new, unseen environments. 

---

## 2. Data Augmentation Pipeline: Why It Is Essential

Deep learning object detectors like YOLO are notoriously "data-hungry." They learn by identifying complex pixel relationships (edges, gradients, textures) across thousands of examples. When presented with only 210 positive examples, a CNN will simply memorize the exact background noise and exact pixel locations of those 210 training images (overfitting), rendering it useless when tested on a new, unseen GPR scan.

To solve this, we cannot just capture more data (which is expensive and labor-intensive). Instead, we intentionally apply a pipeline of mathematical transformations directly tailored to the physics of Ground Penetrating Radar. By synthesizing new, physically plausible variations of the original images, we force the model to focus **strictly on the geometric shape of the hyperbola**, ignoring irrelevant background noise. This drastically increases the model's detection efficiency and real-world robustness.

### How Each Augmentation Stage Increases Detection Efficiency:

1.  **Gaussian Noise (`add_noise`, std=0.15):** 
    *   **Why it's needed:** Real-world GPR soil is never perfectly uniform. It contains rocks, moisture variations, and sensor electromagnetic interference (clutter) that pepper the B-scan with static.
    *   **How it helps efficiency:** By artificially injecting snow-like pixel noise into clean training images, the model learns that "static" is not a determining feature. It forces the neural network's convolutional filters to look past the grain and latch onto the high-contrast, smooth curves of the hyperbola itself.
2.  **Horizontal Time Shift (`time_shift`, shift=20px):** 
    *   **Why it's needed:** A utility pipe could be buried at the exact start of a survey line or right in the middle. The resulting hyperbola might appear on the far left edge or dead center of the 224x224 patch.
    *   **How it helps efficiency:** Translating the image horizontally ensures the model becomes *translation invariant*. It learns that the absolute coordinate of the object doesn't matter; the structural geometry does. This means the model won't fail just because a pipe is slightly off-center in a real-time scan.
3.  **Rotation (`rotate_image`, angle=15°):** 
    *   **Why it's needed:** GPR operators push survey carts over uneven terrain (hills, curbs, debris). If the antenna hits a bump, the radar waves enter the ground at a slight angle, causing the resulting hyperbola in the B-scan to appear slightly lopsided or tilted in the image space.
    *   **How it helps efficiency:** Training the model on ±15° rotated images mathematically prepares the network to recognize these tilted "U" shapes. The model no longer requires a perfectly horizontal ground parallel to detect a pipe.
4.  **Horizontal Flip (`flip_image`, mode=1):** 
    *   **Why it's needed:** A GPR cart can be pushed forward (West to East) or pulled backward (East to West) over the exact same buried object. 
    *   **How it helps efficiency:** Flipping the image left-to-right perfectly simulates scanning the object from the opposite direction. This instantly doubles the dataset with 100% physically accurate realistic representations, improving the model's viewing angles without collecting new data.
5.  **Elastic Deformation (`elastic_transform`, alpha=34, sigma=4):** 
    *   **Why it's needed:** Soil density is highly heterogeneous. Subsurface radar waves travel faster through dry sand and slower through wet clay. If a pipe is buried in mixed soil, the returning radar wave is distorted, causing the hyperbola to look "wavy," asymmetrical, or visually squished.
    *   **How it helps efficiency:** Elastic deformation applies a localized, smooth warping to the pixels in the image. This teaches the model that a hyperbola isn't always a perfect textbook arc. By exposing it to warped signatures during training, the model efficiently detects distorted anomalies in complex, multi-layered soils.
6.  **Spectral Shift (`spectral_shift`, shift=100):** 
    *   **Why it's needed:** Different target depths require different GPR antenna frequencies (e.g., 200 MHz vs 400 MHz). Lower frequencies penetrate deeper but produce "thicker," lower-resolution hyperbolas. Higher frequencies produce crisp, thin hyperbolas but lack depth.
    *   **How it helps efficiency:** Applying a 2D Fast Fourier Transform (FFT) to shift the image frequencies mathematically simulates swapping out the physical antenna on the GPR cart. The model learns to detect the generalized shape of a utility/cavity regardless of whether the radar pulse was high or low frequency, creating an exceptionally robust, hardware-agnostic detector.s.

**Post-Augmentation Dataset Size:**
*   `augmented_cavities`: 553 images
*   `augmented_utilities`: 786 images
*   **Total Processed Images:** 1,339 images

---

## 3. Dataset Normalization (`prepare_dataset.py`)

The annotations for both the utility and cavity augmented datasets were created using the external `makesense.ai` tool. However, the raw output exported from that tool labeled *both* utilities and cavities as "Class 0" in the YOLO `.txt` format.

To allow our model to perform **Multi-Class Detection** (differentiating between a utility pipe and a void/cavity), the dataset had to be programmatically restructured.

**What `prepare_dataset.py` accomplished:**
1.  **YOLO Directory Structure:** Automatically built the strict `images/train`, `images/val`, `labels/train`, and `labels/val` folder hierarchy required by Ultralytics YOLOv8 into a new directory named `yolo_dataset/`.
2.  **Train/Validation Split:** Randomized the augmented dataset and mathematically partitioned it into an 80% Training set and a 20% Validation set.
3.  **Class ID Remapping:** 
    *   Copied the Utility text labels directly representing **Class 0**.
    *   Iterated through every Cavity text label, dynamically rewriting the file to remap the first integer from `0` to **Class 1**.
4.  **Prefixing:** Prefixed image and label filenames with `util_` and `cav_` to absolutely ensure no filename collisions occurred when merging the separated raw folders into the unified YOLO directories.

The final curated dataset consisted of **1,070 Training images** and **269 Validation images**.

---

## 4. Model Architecture & Training (`train.py`)

With the dataset correctly formatted via `data.yaml` (designating `nc: 2` classes), we utilized the **YOLOv8** framework.

**Model Selection:** We instantiated `yolov8n.pt` (YOLOv8 Nano). The Nano architecture is highly optimized, containing fewer parameters making it exceptionally fast to compute. This allowed the entire 50-epoch training cycle to be performed efficiently on a standard CPU without necessitating a dedicated CUDA GPU.

**Training Configuration:**
*   **Epochs:** 50 (allowing the model enough passes to refine feature weights)
*   **Image Size:** 224x224 (perfectly matching the `crop.py` baseline output)
*   **Batch Size:** 8 (preventing memory bottlenecks)
*   **Optimizer:** AdamW (Adam with Weight Decay, ensuring smoother convergence and less overfitting compared to standard SGD)
*   **Learning Rate:** Initialized at 0.001

Furthermore, YOLOv8's internal dataloader applied subtle, dynamic online augmentations during the training loop (such as ±10° rotations, ±10% scale translations, and HSV color space slight adjustments) to ensure the model never saw the exact identical pixel array twice across the 50 epochs.

---

## 5. Validation and Final Metrics

At the conclusion of the 50th epoch, the model evaluated its final learned weights (`best.pt`) against the 269 unseen images in the validation split.

The results were phenomenally successful, proving that our offline geometric and spectral augmentations paired seamlessly with YOLOv8's spatial feature extraction.

### Overall Performance Overview
*   **Overall mAP50 (Mean Average Precision at 0.5 IoU threshold):** `0.9594`
*   **mAP50-95 (Strict Evaluation over continuous IoU scaling):** `0.6068`
*   **Precision (P):** `0.952`
*   **Recall (R):** `0.939`

### Specific Multi-Class Breakdown
The model exhibited an extraordinary ability to differentiate between the structural signatures of solid utilities and air/fluid-filled cavities.

| Class Identifier | Class Name | Instances in Val Split | Precision | Recall | mAP50 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | **Utility** | 204 | 0.9370 | 0.9118 | **0.9463** |
| `1` | **Cavity** | 122 | 0.9668 | 0.9672 | **0.9726** |

**Metric Analysis:**
A Mean Average Precision (mAP50) of nearly 96% is considered highly robust for specialized Computer Vision tasks. 

Notably, **Cavities (0.9726)** were detected with slightly higher confidence than **Utilities (0.9463)**. This is a fascinating validation of GPR physics: underground voids represent a severe, sudden transition in relative dielectric permittivity (e.g., from dense soil $ε_r ≈ 15$ to empty air $ε_r = 1$), resulting in an incredibly sharp, high-amplitude radar reflection that the CNN easily targets. Utilities, depending on their material (PVC vs. metal), may fade slightly more into noisy background terrain, yet still achieved a spectacular ~95% classification accuracy.

---

## 6. Evaluation & Deployment (`app.py`)

A model's structural metrics must be easily verifiable by human operators. To accomplish this, `app.py` was developed utilizing the `Streamlit` open-source app framework.

**Interactive Web Application Features:**
1.  **Direct Image Processing:** Users can upload any `.jpg` or `.png` GPR image directly through their browser, immediately piping it to the trained PyTorch inference engine.
2.  **Side-by-Side Visualization:** The raw uploaded scan is displayed adjacent to the model's output, offering immediate visual confirmation.
3.  **Intelligent Bounding Boxes:** Using OpenCV, the UI automatically color-codes the overlay rects depending on the model's identified class:
    *   **🔵 Blue Boxes:** Utility detected.
    *   **🔴 Red Boxes:** Cavity detected.
4.  **Adjustable Confidence Thresholding:** A sidebar slider allows the operator to dynamically lower or raise the minimum requirement (0.00 to 1.00) for a detected object to be drawn on screen. If set to `0.25`, the model will only draw boxes it is at least 25% sure contain a hyperbola.
5.  **Dynamic Telemetry Table:** The UI renders a structured readout indicating the exact Class Name, the percentage Confidence, and the exact `(X1, Y1) → (X2, Y2)` pixel coordinates of the bounding box.

The Streamlit interface successfully bridges the gap between deep machine learning architecture and actionable human insight, completing the automated pipeline locally.

### Cloud Deployment Strategy (Streamlit Community Cloud)
Deploying the `app.py` interface to Streamlit Community Cloud required overcoming key platform-specific infrastructure challenges:
1. **GitHub Tracking:** Ultralytics YOLO purposefully `gitignores` the `runs/` directory where model weights are saved. To allow Streamlit Cloud access to the `best.pt` weights, the file was strategically extracted into a dedicated, tracked `models/` directory.
2. **Linux GUI Dependency Hell:** Standard `opencv-python` (forced by the YOLOv8 package) fundamentally requires system-level GUI drivers (`libGL.so.1`, `libgthread-2.0.so.0`). On Streamlit's minimal containerized Debian instances, installing these via `apt` (`packages.txt`) frequently fails due to unresolved dependencies like `libffi7` on their mixed repositories.
3. **Self-Healing Python Patch:** Instead of relying on brittle system-level `apt-get` commands, a robust **self-healing script** was injected directly at the top of `app.py`. If the script detects an `ImportError` when loading `cv2`, it uses `subprocess.check_call()` to dynamically `pip uninstall` the broken standard OpenCV installation mid-boot and forcibly recompiles `opencv-python-headless`. This guarantees an error-free boot sequence without native GUI library requirements.

---

## 7. Conclusion

This project successfully traversed the entire machine learning lifecycle inside anomalous Ground Penetrating Radar detection.

We started with a dataset of just 210 annotated images—far too small for generalized neural networks. By designing a highly-targeted subset of signal augmentations (most notably the FFT spectral shift), we algorithmically simulated a much larger dataset of 1,339 unique physical scenarios.

We automated the restructuring of annotation formats, pivoting from a single-class structure to a true multi-class binary problem, ensuring YOLO directories were accurately synced. 

Finally, training a YOLOv8 Nano model upon this prepared split yielded a highly accurate multi-class anomaly detector (mAP50: **0.9594**), completely solving the problem of accurately identifying distinguishing subsurface properties from electromagnetic scan imagery efficiently on local hardware.
