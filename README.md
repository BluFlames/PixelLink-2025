# Text Regeneration App

A complete, runnable web-based application for text regeneration using OCR and deep learning models.
PixelLink for Text Detection.
PaddleOCR for Text Recognition.
Google ByT5 for Missing Text Regeneration.
This repository is intended to be **cloned, executed locally, studied, and improved** by others.

---

## What This Project Does

- Accepts image input
- Performs text recognition (OCR)
- Regenerates / processes recognized text using trained models
- Provides a simple web interface to interact with the pipeline

This project is shared as a **baseline implementation**, not a polished product.

---

## Tech Stack

- Python
- Flask
- PP-OCR (PaddleOCR inference model)
- TensorFlow Lite
- HTML, CSS, JavaScript

---

## Repository Structure
.

├── app.py # Main application entry point

├── debug.py # Debug / testing utilities

├── requirements.txt # Python dependencies

├── models/ # Quantized / trained models

├── en_PP-OCRv4_rec_slim_infer/ # OCR inference model files

├── templates/ # HTML templates

├── static/ # CSS and JavaScript assets

├── .gitignore

└── README.md

---

## System Requirements

- Python 3.8+
- pip
- Virtual environment support

---

## Setup Instructions

##
**1. Clone the repository**

git clone https://github.com/BluFlames/PixelLink-2025.git
cd PixelLink-2025

## 
**2. Create and activate a virtual environment**

Windows

python -m venv venv
venv\Scripts\activate

Linux / macOS

python3 -m venv venv
source venv/bin/activate

## 
**3. Install dependencies**
pip install -r requirements.txt

Running the Application
python app.py

Open your browser and go to:
http://127.0.0.1:5000/

<img width="1920" height="1020" alt="Screenshot 2026-01-22 150823" src="https://github.com/user-attachments/assets/cffbafa9-a49d-4113-8576-823de0ad6316" />
UI of the Web application

![silence without c](https://github.com/user-attachments/assets/322b9a8e-4623-424a-a5f4-9d51cccca573)
Test Image

<img width="949" height="1280" alt="download" src="https://github.com/user-attachments/assets/7e0c7a7c-6dc2-44c1-b036-ef84122e5ed8" />
Output Image after Detection, Recoignition and Regeneration


<img width="1920" height="1020" alt="Screenshot 2026-01-22 151017" src="https://github.com/user-attachments/assets/fbd14538-0a10-4941-9a0a-3391a1bdd92e" />
Textual Output of the Application


## Models used

**1. PixelLink (Custom trained on 1.5K images)**
This model was trained using ISTD OC 2021 aand ICDAR 2015 datasets.
Prominently trained on the 90% occlusion dataset of the ISTD OC 2021 dataset.
It also uses the 1K images available in the standard ICDAR 2015 Dataset. 

**2. PaddleOCR (Quantized Model)**
Pretrained model but with quantization

**3. Google ByT5**
For correcting the recognized text, incase the text requires any corrections.


## Important Notes

No API keys or secrets are required.
All execution is local.
Tested on a local development environment only.
This is an academic project.

**The aim of this project was to custom train a Deep Learning Model (here, PixelLink), 
create a pipeline that detects, recognizes and regenerates missing characters 
from an image with text. English is the only language that this project works.**

## Contributions are welcome
It is highly encouraged.
