# 🧠 YOLOv11-Powered Brain Tumor Diagnostic Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18898503.svg)](https://doi.org/10.5281/zenodo.18898503)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📄 Official Publication
This project is officially published on **Zenodo**. You can read the full research paper here:
👉 **[Read the Paper (DOI: 10.5281/zenodo.18898503)](https://doi.org/10.5281/zenodo.18898503)**

## 🚀 Project Overview
An end-to-end medical AI solution for the classification and pixel-level segmentation of brain tumors from MRI scans.

### Key Features:
* **Architecture:** YOLOv11 (Latest Generation).
* **Accuracy:** **98.7%** on a balanced dataset of Glioma, Meningioma, Pituitary, and Healthy scans.
* **Gatekeeper Pipeline:** A dual-pipeline logic that ensures high-confidence results before clinical visualization.
* **Deployment:** Real-time web interface built with **Streamlit**.

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/Shabana2002/Brain-Tumor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the App: `streamlit run app.py`

## 📊 Results
The model provides high-precision masks and classification labels, reducing the time required for manual radiological review.
