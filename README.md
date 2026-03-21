# Advanced Buried Object Detection (GPR)

This project provides an end-to-end deep learning pipeline for detecting and classifying buried objects from Ground Penetrating Radar (GPR) imagery. It features a complete workflow from raw dataset processing and data augmentation to **YOLOv8** multi-class model training and an interactive **Streamlit** evaluation web interface.

## 🚀 Features

*   **Multi-Class Detection:** Automatically distinguishes between buried **Utilities** (pipes/cables) and underground **Cavities** (voids) by analyzing their hyperbolic radar signatures.
*   **Domain-Specific Augmentation:** Expands a critically small raw dataset using signal-aware augmentations (like FFT spectral shifting and elastic deformation) to prevent overfitting and simulate real-world subsurface variability.
*   **Automated Dataset Prep:** A robust script (`prepare_dataset.py`) converts makesense.ai output into YOLO-ready directories with correct class mappings.
*   **Optimized Training:** A ready-to-use YOLOv8 training script configured to run efficiently on CPUs without requiring dedicated CUDA graphics.
*   **Interactive UI:** A dark-themed Streamlit frontend (`app.py`) for uploading raw B-scans to instantly view side-by-side color-coded detection bounding boxes.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Advanced-Buried-Object-Detection.git
   cd Advanced-Buried-Object-Detection
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Running the Pipeline (End-to-End)

### Step 1: Prepare the YOLO Dataset
The raw annotations label both utilities and cavities as "Class 0". We need to separate them into Class 0 (Utilities) and Class 1 (Cavities) and format them for YOLO.
```bash
python prepare_dataset.py
```
*This processes the `GPR_data/` folder and creates an 80/20 train/validation split inside a new `yolo_dataset/` folder.*

### Step 2: Train the YOLO Model
Start the training script. By default, it uses the lightweight `yolov8n.pt` model, trains for 50 epochs, and is optimized for CPU usage.
```bash
python train.py
```
*Trained weights will be saved automatically to `runs/detect/gpr_multiclass/weights/best.pt`.*

### Step 3: Evaluate via Streamlit Interface
Launch the real-time inference web application to test your model on unseen images.
```bash
streamlit run app.py
```
*Open your browser to `http://localhost:8501`. Upload any GPR patch (like those in `GPR_data/intact` or unseen examples) to see the detections and bounding boxes.*

---

## 📁 Dataset Description

The dataset consists of high-resolution Ground Penetrating Radar (GPR) data, specifically A-scan and B-scan profiles, obtained using antennas operating at 400 MHz and 200 MHz across various infrastructure projects in Morocco (2019-2024).

The baseline patch dataset (pre-augmentation) includes:
- **79** original profiles with underground cavities
- **131** original profiles with underground utilities
- **75** original intact profiles (background)

All annotations were generated using [makesense.ai](https://www.makesense.ai/) and are exported in both YOLO and VOC XML formats. 
The dataset can be downloaded [here](https://data.mendeley.com/datasets/ww7fd9t325/1).

### Data Augmentation
To expand the dataset for deep learning, `augmentation.py` applies geometric transformations (flips, rotations, time-shifting) alongside advanced techniques like **spectral shifting** (frequency domain manipulation) and **elastic deformations**. This expands the available training pool to **1,339 annotated images**.

---

## 📂 Project Structure

```
├── GPR_data/               # Original & augmented dataset (downloaded)
├── yolo_dataset/           # Generated YOLO-ready dataset (via prepare_dataset.py)
├── runs/                   # Training checkpoints and validation plots (auto-generated)
├── app.py                  # Streamlit evaluation web interface
├── augmentation.py         # Advanced GPR data augmentation strategies
├── crop.py                 # Slices wide GPR profiles into 224x224 CNN patches
├── data.yaml               # YOLO class and path configuration
├── prepare_dataset.py      # Cleans, remaps, and splits annotations for YOLO
├── Project_Report.md       # Comprehensive breakdown of AI pipeline and physics
├── requirements.txt        # Required Python packages
└── train.py                # CPU-Optimized YOLOv8 training routine
```

---

## 📜 Citing the Original Dataset

If you use the foundational GPR dataset, please cite the following publication:

_A. MOJAHID, D. EL OUAI, K. El Amraoui, K. EL-HAMI, H. AITBENAMER, (2024). Intelligent Recognition of Subsurface Utilities and Voids: A Ground Penetrating Radar Dataset for Deep Learning Applications. Data in Brief. 10.17632/ww7fd9t325.1_

```bibtex
@article{mojahid2024Intelligent,
    title={Intelligent Recognition of Subsurface Utilities and Voids: A Ground Penetrating Radar Dataset for Deep Learning Applications.},
    author={Abdelaziz MOJAHID, Driss EL OUAI, Khalid EL AMRAOUI, Khalil EL HAMI, Hamou AITBENAMER},
    journal={Data in Brief},
    doi={10.17632/ww7fd9t325.1},
    publisher={Elsevier}
}
```
