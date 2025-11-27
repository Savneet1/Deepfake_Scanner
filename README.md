# Deepfake Scanner

This project implements a deepfake detection pipeline using a small subset of the FaceForensics++ dataset. It extracts frames from real and fake videos, trains a MobileNetV2-based classifier on CPU under tight resource constraints, and evaluates frame-level classification performance.

The final model (`best_model.pt`) achieves approximately 97.96% frame-level accuracy on 2,160 frames extracted from 72 FaceForensics++ videos (37 real, 35 fake).

## Owner

This repository and code are maintained by **Savneet Singh** (`GitHub: Savneet1`).

## Dataset

This project uses a subset of the **FaceForensics** / FaceForensics++ dataset. The dataset itself is NOT included in this repository. To reproduce the experiments, you must obtain FaceForensics(++) from the original authors under their terms and conditions.

Please cite the dataset as:

> A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner,  
> "FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces,"  
> arXiv, 2018.

BibTeX:

@article{roessler2018faceforensics,
author = {Andreas R"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
title = {FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces},
journal = {arXiv},
year = {2018}
}

text

## Important Notes

- The original FaceForensics / FaceForensics++ videos are not distributed here.
- All training was performed on CPU in a resource-constrained virtual machine.
- The main artifacts are:
  - `scripts/1_extract_frames.py`
  - `scripts/train_model.py`
  - `scripts/3_evaluate_model.py`
  - `models/checkpoints/best_model.pt`
  - `logs/evaluation_results.json`
