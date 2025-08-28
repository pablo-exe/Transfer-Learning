# Transfer Learning Portfolio — Image Classification & Safety Applications

This repository collects three end-to-end projects that demonstrate my ability to apply **transfer learning** to real problems. Each notebook is self-contained and shows the full workflow: data ingestion, augmentation, model building, training, evaluation, and saving artifacts.


---

## Repository Structure

- `Transfer_Learning.ipynb` — Binary image classification (Cats vs Dogs) with **VGG16** feature extraction + custom classifier.
- `Transfer Learning exercise.ipynb` — A practice notebook comparing **two approaches**: a small CNN **from scratch** vs **transfer learning** with VGG16 on a directory-structured dataset.
- `Social distancing exercise.ipynb` — **Mask & social-distancing** pipeline: train a mask classifier (VGG19) and run inference with **OpenCV Haar cascades** + pairwise distance checks to flag violations.

---

## Notebook Summaries & What to Notice

### 1) `Transfer_Learning.ipynb` — Cats vs Dogs (Transfer Learning with VGG16)
**Problem:** Build a robust binary classifier for dog/cat images.  
**Data:** Folder-based dataset (`datasets/dogscats/train`, `datasets/dogscats/validation`).  
**Approach:**
- Load **VGG16 (weights='imagenet', include_top=False)** to use as a **frozen convolutional feature extractor**.
- Add a small **Dense head** and train only the top classifier.
- Use **ImageDataGenerator** with rescaling and augmentation.
- Train for **~20 epochs** at **150×150** resolution and **save the model** (`with_vgg16_cats_dogs.h5`).
- Plot training curves with **Matplotlib** to monitor overfitting.

**Key skills shown:**
- Practical **transfer learning** (freezing backbone, training a new head).
- **Data augmentation** to improve generalization.
- **Keras/TensorFlow** model assembly, training, and checkpointing.
- Interpreting **learning curves** and adjusting training accordingly.

---

### 2) `Transfer Learning exercise.ipynb` — From-Scratch CNN vs VGG16 Transfer
**Problem:** Compare a **from-scratch CNN** against **VGG16 transfer learning** on a folder-organized dataset (the notebook includes paths such as an “intel” dataset folder and Cats vs Dogs helpers; the pipeline itself is generic for **binary classification**).

**Approach:**
- **Baseline:** Build a small **Sequential CNN** (Conv2D → MaxPool → Dense) and train end-to-end.
- **Transfer:** Load **VGG16 (include_top=False, weights='imagenet')**, precompute **bottleneck features**, and train a compact **Dense classifier** on top.
- Use **ImageDataGenerator** with rotation/shift/zoom/flip/brightness.
- Conduct experiments with **batch size (20–32)** and **~20 epochs**; optimizer **RMSprop**.
- Compare convergence speed, stability, and generalization between both approaches.

**Key skills shown:**
- Building and training **custom CNNs** vs **feature-extraction** workflows.
- Efficient pipelines with **precomputed activations (bottlenecks)**.
- **Experimentation & tuning** (augmentations, batch sizes, simple optimizer choices).
- Clean separation of **data loaders**, **feature extraction**, and **classifier training**.

---

### 3) `Social distancing exercise.ipynb` — Mask Classification + Distance Violations
**Problem:** Detect whether faces wear a mask and highlight **social-distancing** violations in images.

**Approach:**
- **Mask classifier:** Train a binary classifier (MASK / NO MASK) using **VGG19 (include_top=False, weights='imagenet')** at **128×128**, with augmentation (flip/zoom/shear). Optimizer **Adam**, **batch size 32**, **~20 epochs**. Save the model (e.g., `masknet.h5`).
- **Face detection:** Use **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`) to detect faces in images/frames.
- **Distance logic:** Compute face-centroid **Euclidean distances** (via `scipy.spatial`) and flag pairs below a threshold.  
  - Draw **green boxes** for compliant faces, **red boxes** for violations.  
  - Overlay **MASK / NO MASK** label per face using the trained classifier.

**Key skills shown:**
- Combining **deep learning** (Keras/TensorFlow) with **classical CV** (OpenCV).
- Designing an **end-to-end inference pipeline** that stitches detection, classification, and business logic.
- Practical handling of **real-world constraints** (thresholding, visualization, per-frame processing).

---

## Skills Matrix

| Skill / Topic | Cats vs Dogs (VGG16) | From-Scratch vs VGG16 | Mask & Distancing (VGG19 + OpenCV) |
|---|:--:|:--:|:--:|
| Transfer learning (feature extraction) | ✅ | ✅ | ✅ |
| Fine-tuning strategy (frozen backbone + new head) | ✅ | ✅ | ✅ |
| Custom CNN from scratch |  | ✅ |  |
| Data pipelines with `ImageDataGenerator` | ✅ | ✅ | ✅ |
| Augmentation (rotation/shift/zoom/flip/brightness/shear) | ✅ | ✅ | ✅ |
| Keras/TensorFlow modeling & training | ✅ | ✅ | ✅ |
| OpenCV detection (Haar cascades) |  |  | ✅ |
| Distance-based post-processing (pairwise Euclidean) |  |  | ✅ |
| Experimentation & comparison |  | ✅ | ✅ |
| Visualization of metrics & results | ✅ | ✅ | ✅ |
| Model saving & reuse | ✅ | ✅ | ✅ |

---

## Tech Stack

- **Python**, **Jupyter/Colab**
- **TensorFlow / Keras**
- **NumPy**, **Matplotlib**, **Pandas**, **SciPy**
- **OpenCV** (Haar cascades for face detection)

---

## How to Run (local or Colab)

> The notebooks use **directory-structured datasets** (`train/`, `validation/`, and optionally `test/`). Replace paths in the notebooks with your local folders.

**Environment (example with `pip`):**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install tensorflow keras opencv-python numpy matplotlib scipy pandas
Run: jupyter notebook

Open a notebook, update dataset paths, and run cells top-to-bottom.

Notes:

For GPU acceleration, use Google Colab or install local CUDA-enabled TensorFlow.
Some code references Colab paths (e.g., Google Drive). Update them for your environment.
The Haar cascade XML must be available (e.g., haarcascade_frontalface_default.xml).
