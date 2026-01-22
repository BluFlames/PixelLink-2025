import os
import cv2
import numpy as np
import gc
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ML libs
import tensorflow as tf
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # <-- Changed imports
import torch
import re
from spellchecker import SpellChecker

# -------------------------
# SERVER CONFIG
# -------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------
# MODEL PATHS & HYPERPARAMS
# -------------------------
TFLITE_MODEL_PATH = "models/quantized_model_20ep.tflite"
IMG_SIZE = 512
TEXT_THRESHOLD = 0.8
LINK_THRESHOLD = 0.8
MIN_AREA_PIXELS = 20

# -------------------------
# GLOBAL MODELS
# -------------------------
tflite_interpreter, input_details, output_details = None, None, None
paddle_ocr = None
correction_model, correction_tokenizer = None, None # <-- Renamed for clarity
spell = None

# -------------------------
# TEXT QUALITY CHECK
# -------------------------
def is_text_corrupted(text: str) -> bool:
    if not text or len(text) < 3: return False
    non_alnum_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if len(text) > 0 and non_alnum_count / len(text) > 0.4: return True
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', text, re.IGNORECASE): return True
    words = text.split()
    if any(len(w) > 2 and not w.isupper() and not w.islower() and not w.istitle() for w in words): return True
    words_to_check = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    if not words_to_check: return False
    misspelled = spell.unknown(words_to_check)
    if misspelled:
        print(f"Spell check flagged as corrupted. Misspelled: {misspelled}")
        return True
    return False

# -------------------------
# LOAD MODELS
# -------------------------
def load_models():
    global tflite_interpreter, input_details, output_details, paddle_ocr, correction_model, correction_tokenizer, spell

    print("Loading models...")
    
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    print("✅ TFLite PixelLink detector loaded")

    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', det=False, use_gpu=False, show_log=False, enable_mkldnn=True, cpu_threads=4)
    print("✅ PaddleOCR recognizer loaded")

    # --- USING ByT5 FOR OCR CORRECTION ---
    print("Loading Yelp ByT5 OCR corrector...")
    MODEL_NAME = "yelpfeast/byt5-base-english-ocr-correction"
    correction_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    correction_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")  # force CPU
    correction_model = torch.quantization.quantize_dynamic(correction_model, {torch.nn.Linear}, dtype=torch.qint8)
    correction_model.eval()
    print(f"✅ ByT5 ({MODEL_NAME}) corrector loaded")
    # --- END OF CHANGE ---

    spell = SpellChecker()
    print("✅ Spell checker loaded")

# -------------------------
# IMAGE & TEXT PROCESSING
# -------------------------
def preprocess_image_for_detection(img_bgr):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = IMG_SIZE / max(h, w)
    rh, rw = int(h * scale), int(w * scale)
    pad_h, pad_w = (IMG_SIZE - rh) // 2, (IMG_SIZE - rw) // 2
    img_resized = cv2.resize(img_rgb, (rw, rh))
    img_padded = np.full((IMG_SIZE, IMG_SIZE, 3), 128, np.uint8)
    img_padded[pad_h:pad_h+rh, pad_w:pad_w+rw] = img_resized
    return np.expand_dims(preprocess_input(img_padded.astype(np.float32)), 0), (pad_h, pad_w, rh, rw)

def run_detection(img_input):
    tflite_interpreter.set_tensor(input_details[0]['index'], img_input)
    tflite_interpreter.invoke()
    out0 = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
    out1 = tflite_interpreter.get_tensor(output_details[1]['index'])[0]
    return (out0, out1) if out0.shape[-1] == 1 else (out1, out0)

def extract_text_regions(text_map, orig_img, padding_info):
    pad_h, pad_w, rh, rw = padding_info
    orig_h, orig_w = orig_img.shape[:2]
    text_pixels = (text_map[:, :, 0] > TEXT_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(text_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, rois = [], []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA_PIXELS: continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        map_h, map_w = text_map.shape[:2]
        box[:, 0] *= (IMG_SIZE / map_w)
        box[:, 1] *= (IMG_SIZE / map_h)
        box[:, 0] -= pad_w
        box[:, 1] -= pad_h
        box[:, 0] *= (orig_w / rw)
        box[:, 1] *= (orig_h / rh)
        box = np.clip(box, [0, 0], [orig_w - 1, orig_h - 1])
        box_int = box.astype(np.int32)
        x_min, y_min = np.min(box_int, axis=0)
        x_max, y_max = np.max(box_int, axis=0)
        if x_max - x_min < 5 or y_max - y_min < 5: continue
        roi = orig_img[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            boxes.append(box_int)
            rois.append(roi)
    return boxes, rois

def recognize_text_batch(rois):
    texts, confs = [], []
    for roi in rois:
        try:
            result = paddle_ocr.ocr(roi, cls=False, det=False)
            if result and result[0]:
                line = result[0][0]
                texts.append(line[0])
                confs.append(line[1])
            else:
                texts.append(""); confs.append(0.0)
        except:
            texts.append(""); confs.append(0.0)
    return texts, confs

# --- UPDATED REGENERATION FUNCTION FOR BYT5 ---
def regenerate_text(raw_text, confidence):
    if not raw_text.strip():
        return ""
    
    if confidence > 0.95 and not is_text_corrupted(raw_text):
        return raw_text

    if is_text_corrupted(raw_text) or confidence < 0.7:
        try:
            # ByT5 is fine-tuned for OCR correction and doesn't need a specific prompt format
            input_ids = correction_tokenizer(raw_text, return_tensors="pt").input_ids.to("cpu")
            
            with torch.no_grad():
                outputs = correction_model.generate(
                    input_ids,
                    max_length=100, # Increased max_length for potentially longer corrections
                    num_beams=5,
                    early_stopping=True
                )
            
            regenerated = correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return regenerated if regenerated else raw_text

        except Exception as e:
            print(f"ByT5 correction error: {e}")
            return raw_text
    
    return raw_text
# --- END OF UPDATE ---

# -------------------------
# MAIN PROCESSING PIPELINE
# -------------------------
def process_image(img_bgr):
    img_input, padding_info = preprocess_image_for_detection(img_bgr)
    text_map, link_map = run_detection(img_input)
    boxes, rois = extract_text_regions(text_map, img_bgr, padding_info)
    raw_texts, confidences = recognize_text_batch(rois)
    regenerated_texts = [regenerate_text(txt, conf) for txt, conf in zip(raw_texts, confidences)]
    
    vis_img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
    for box, text in zip(boxes, regenerated_texts):
        cv2.polylines(vis_img, [box], True, (0, 255, 0), 2)
        top_left = tuple(box[np.argmin(np.sum(box, axis=1))])
        cv2.putText(vis_img, text, (top_left[0], max(15, top_left[1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    pil_img = Image.fromarray(vis_img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    gc.collect()
    
    return {
        "image": img_b64,
        "detections": [{
            "box": box.tolist(), "raw_text": raw, "regenerated_text": regen,
            "confidence": float(conf), "was_corrected": raw != regen
        } for box, raw, regen, conf in zip(boxes, raw_texts, regenerated_texts, confidences)]
    }

# -------------------------
# FLASK ROUTES
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files.get('image')
        if not file: return jsonify({'error': 'No image uploaded'}), 400
        
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'error': 'Invalid image file'}), 400
        
        result = process_image(img)
        return jsonify(result)
        
    except Exception as e:
        print(f"SERVER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred'}), 500

# -------------------------
# START SERVER
# -------------------------
if __name__ == "__main__":
    load_models()
    print("\n" + "="*50)
    print("🚀 Server ready! Access at http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)