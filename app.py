import os
import sys
import torch
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import requests

# Setup Flask app
app = Flask(__name__)
@app.route('/')
def index():
    return '✅ Helmet Detection API is live'

# Add yolov5 to path
sys.path.append('./yolov5')

# Import YOLOv5 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

# Load model
device = torch.device('cpu')
model = attempt_load('best.pt', device)
model.eval()

# Telegram Bot Config
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        try:
            requests.post(url, data=payload)
        except Exception as e:
            print("Failed to send Telegram alert:", e)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        orig_img = np.array(img)
    except Exception as e:
        return jsonify({'error': 'Invalid image'}), 400

    # Preprocess
    img_resized = letterbox(orig_img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(device).float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Post-process
    detections = []
    no_helmet_detected = False
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], orig_img.shape).round()
            for *xyxy, conf, cls in det:
                label = model.names[int(cls)]
                confidence = float(conf)
                bbox = [float(x.item()) for x in xyxy]
                detections.append({'label': label, 'confidence': confidence, 'bbox': bbox})
                if label == 'no_helmet':
                    no_helmet_detected = True

    # Telegram Alert
    if no_helmet_detected:
        send_telegram_alert("🚨 Alert: No helmet detected!")

    return jsonify({'detections': detections})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # For Render
    app.run(host="0.0.0.0", port=port)



