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


vehicle_images/: Cropped snapshots of tracked vehicles

