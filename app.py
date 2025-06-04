import os
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import numpy as np
# Optional: for telegram alerts
import requests

app = Flask(__name__)

# Load your YOLOv5 model once
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

# Telegram Bot Config (fill your token and chat id or leave empty)
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
    except Exception as e:
        return jsonify({'error': 'Invalid image'}), 400

    # Run detection
    results = model(img)
    preds = results.pandas().xyxy[0]  # Pandas DataFrame of predictions

    # Extract relevant info
    detections = []
    no_helmet_detected = False
    for _, row in preds.iterrows():
        label = row['name']
        conf = float(row['confidence'])
        bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
        detections.append({'label': label, 'confidence': conf, 'bbox': bbox})
        if label == 'no_helmet':
            no_helmet_detected = True

    # Send telegram alert if no helmet detected
    if no_helmet_detected:
        send_telegram_alert("Alert: No helmet detected!")

    return jsonify({'detections': detections})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
