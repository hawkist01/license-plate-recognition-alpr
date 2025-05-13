# license-plate-recognition-alpr

# License Plate Recognition System with ALPR Dashboard Integration 🚗📸

This project implements an advanced License Plate Recognition (LPR) pipeline using a combination of:
- YOLOv8 object detection for vehicle localization,
- OCR engines (PaddleOCR + Tesseract) for plate reading,
- Machine learning–based correction for noisy OCR,
- DeepSORT vehicle tracking, and
- Optional integration with an ALPR Web Dashboard.

---

## 🔍 Features

✅ Real-time vehicle detection and tracking  
✅ License plate detection and enhancement  
✅ Multi-OCR inference with ML-based post-processing  
✅ Consensus-based final plate prediction  
✅ ALPR Dashboard integration (optional)  
✅ Output annotated video and plate logs  
✅ Saves per-vehicle images for further analysis

---

## 📦 Dependencies

```bash
pip install -q opencv-python-headless ultralytics deep_sort_realtime paddleocr pytesseract scikit-learn joblib

Also install Tesseract if not available:
```bash
 sudo apt-get install -y tesseract-ocr

##📁 File Structure
```bash
├── license_plate_recognition.py      # Main pipeline script
├── yolov8n.pt                        # YOLOv8 model (downloaded automatically)
├── results.txt                       # Output file with recognized plates
├── vehicle_images/                   # Cropped images of detected vehicles
├── output_*.mp4                      # Annotated output video

##▶️ How to Run
###Option 1: From Command Line
```bash
python license_plate_recognition.py --video path/to/your/video.mp4 --output results.txt --use_dashboard
###Option 2: From Colab
```bash
args = types.SimpleNamespace(
    video="/content/your_video.mp4",
    output="results.txt",
    dashboard_url="http://localhost:8000/alpr_app/default/call/json/message",
    use_dashboard=False
)
And run:
```bash
%run license_plate_recognition.py

##🧠 Core Modules
YOLOv8 (via Ultralytics): Vehicle detection

DeepSORT: Multi-object tracking

PaddleOCR & Tesseract: OCR with fallback

Naive Bayes model: ML-based OCR correction

Custom heuristic rules: Error correction for region-specific plates

OpenALPR (optional): Additional OCR backend

##📸 Output

results.txt: License plates with timestamp and confidence

output_*.mp4: Annotated video with tracking overlays

vehicle_images/: Cropped snapshots of tracked vehicles

