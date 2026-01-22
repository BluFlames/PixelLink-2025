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

**1. Clone the repository**

```bash
git clone https://github.com/BluFlames/PixelLink-2025.git
cd PixelLink-2025

**2. Create and activate a virtual environment**

Windows

python -m venv venv
venv\Scripts\activate

Linux / macOS

python3 -m venv venv
source venv/bin/activate

**3. Install dependencies**
pip install -r requirements.txt

Running the Application
python app.py

Open your browser and go to:
http://127.0.0.1:5000/
