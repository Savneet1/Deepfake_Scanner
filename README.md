# Deepfake Scanner

Deepfake Scanner is a lightweight deepfake detection pipeline built for low‑resource machines (CPU, ~4 GB RAM). It uses a MobileNetV2‑based classifier trained on a small subset of the FaceForensics++ dataset to classify video frames as real or fake.

The final model (`best_model.pt`) achieves approximately **97.96% frame‑level accuracy** on 2,160 frames extracted from 72 FaceForensics++ videos (37 real, 35 fake).

## Owner

This repository and code are maintained by **Savneet Singh** (GitHub: [Savneet1](https://github.com/Savneet1)).

## Dataset

This project uses a subset of the **FaceForensics / FaceForensics++** dataset for training and evaluation. The dataset itself is **NOT** included in this repository. To reproduce the experiments, you must obtain FaceForensics(++) from the original authors under their terms and conditions.

Please refer to the [original FaceForensics++ paper/project](https://github.com/ondyari/FaceForensics) for proper citation and usage details.

## Project structure

- `data/`
  - `real/`, `fake/` – input videos from FaceForensics++ (not in repo)
- `extracted_frames/`
  - `real/`, `fake/` – extracted training/eval frames (not in repo)
- `models/checkpoints/`
  - `best_model.pt` – trained MobileNetV2 weights
- `logs/`
  - `training_history.json` – training metrics log
- `scripts/`
  - `1_extract_frames.py` – extract frames from dataset videos
  - `train_model.py` – train MobileNetV2 classifier
  - `3_evaluate_model.py` – evaluate best_model.pt on validation frames
  - `4_predict_folder.py` – run predictions on any folder of images

Folders like `data/`, `extracted_frames/`, `test/`, and `test_frames/` are created locally and are ignored in the repository.

## Installation

1. **Clone the repository:**

git clone https://github.com/Savneet1/Deepfake_Scanner.git
cd Deepfake_Scanner

text

2. **Create and activate a virtual environment (recommended):**

python3 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

text

3. **Install Python dependencies:**

pip install -r requirements.txt

text

4. **Prepare the dataset:**
- Download FaceForensics / FaceForensics++ following the official instructions.
- Place a small subset of videos into:
  - `data/real/`
  - `data/fake/`

## How to use

### 1. Extract frames

From project root:

python3 scripts/1_extract_frames.py

text
This reads videos from `data/real` and `data/fake` and writes frames into `extracted_frames/`.

### 2. Train the model

python3 scripts/train_model.py

text
This trains a MobileNetV2 classifier on the extracted frames and saves checkpoints into `models/checkpoints/`, including `best_model.pt`.

### 3. Evaluate the model

python3 scripts/3_evaluate_model.py

text
This loads `best_model.pt`, evaluates it on the validation split of `extracted_frames`, and writes metrics (e.g., accuracy, confusion matrix) into `logs/evaluation_results.json`.

### 4. Run predictions on a folder

You can classify any folder of images (e.g., frames from your own videos):

python3 scripts/4_predict_folder.py
--model_path models/checkpoints/best_model.pt
--input_dir path/to/frames

text
The script prints per‑image predictions (Real/Fake) and basic statistics for the folder.

## Important notes

- The dataset is not included here; only code and configuration are provided.
- This project was designed for experimentation and education on a constrained CPU‑only environment.
- **Disclaimer:** Do not use this model as a sole decision‑maker for high‑stakes scenarios; always combine automated detection with human review and additional verification.
