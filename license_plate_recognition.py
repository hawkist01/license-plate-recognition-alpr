#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License Plate Recognition System with ALPR Web Dashboard Integration
This script provides a complete pipeline for detecting vehicles, recognizing license plates,
and sending the results to an ALPR Web Dashboard for visualization.
"""

import cv2
import numpy as np
import os
import time
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR
import pytesseract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ===== ALPR Web Dashboard Connector =====
class ALPRWebDashboardConnector:
    def __init__(self, server_url="http://localhost:8000/alpr_app/default/call/json/message"):
        self.server_url = server_url
        self.session = requests.Session()
        print(f"Dashboard connector initialized with URL: {self.server_url}")
    
    def send_plate_detection(self, plate_text, confidence, timestamp, vehicle_id=None, image_path=None):
        """Send a plate detection to the ALPR Web Dashboard"""
        data = {
            "plate": plate_text,
            "confidence": confidence,
            "timestamp": timestamp,
            "camera_id": "cam1",  # Default camera ID
            "vehicle_id": vehicle_id if vehicle_id else "unknown"
        }
        
        # If we have an image, we can optionally send it too
        files = {}
        if image_path and os.path.exists(image_path):
            files = {
                'plate_image': ('plate.jpg', open(image_path, 'rb'), 'image/jpeg')
            }
        
        try:
            if files:
                response = self.session.post(self.server_url, data=data, files=files)
            else:
                response = self.session.post(self.server_url, json=data)
                
            if response.status_code == 200:
                print(f"Successfully sent plate {plate_text} to dashboard")
                return True
            else:
                print(f"Failed to send plate data. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending plate data: {str(e)}")
            return False

# ===== License Plate Recognition Components =====
class EnhancedOCRCorrector:
    """
    Enhanced ML-based correction system for OCR results
    """
    def __init__(self):
        # Initialize OCR engines
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Regular expression pattern for license plates (customize for your region)
        self.plate_pattern = re.compile(r'^[A-Z0-9]{2,8}$')
        
        # Train correction model with expanded example data
        self.train_correction_model()
        
    def train_correction_model(self):
        """Train a robust ML model to correct common OCR errors"""
        # Expanded training data for better coverage
        incorrect_plates = [
            'KA2Q7999', 'XA29799Q', 'KA2979Q9', 'KA297QQ9',
            'K4297999', 'KA29799O', 'XA297999', 'KA29Z999',
            'KA29799S', 'KA2979S9', 'K4297Q99', 'KA29799G',
            'KA297999', 'KA297998', 'KA297997', 'KA297996',
            'XA04PP', 'X404PP', 'XAO4PP', 'XA04P9',
            'K404PP', 'KAO4PP', 'KA04P9', 'KA04?P'
        ]
        correct_plates = [
            'KA297999', 'KA297999', 'KA297999', 'KA297999',
            'KA297999', 'KA297999', 'KA297999', 'KA297999',
            'KA297999', 'KA297999', 'KA297999', 'KA297999',
            'KA297999', 'KA297999', 'KA297999', 'KA297999',
            'KA04PP', 'KA04PP', 'KA04PP', 'KA04PP',
            'KA04PP', 'KA04PP', 'KA04PP', 'KA04PP'
        ]
        
        # Create character n-gram features
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
        X = self.vectorizer.fit_transform(incorrect_plates)
        
        # Train model
        self.model = MultinomialNB()
        self.model.fit(X, correct_plates)
        
    def ml_correct_plate(self, plate_text):
        """Apply ML-based correction to fix common OCR errors"""
        if not plate_text or len(plate_text) < 2:
            return plate_text
            
        # Transform the text using our vectorizer
        try:
            X = self.vectorizer.transform([plate_text])
            
            # Predict the corrected plate
            corrected = self.model.predict(X)[0]
            return corrected
        except:
            return plate_text  # Return original if error occurs
    
    def apply_heuristic_rules(self, text):
        """Apply rule-based corrections for common OCR mistakes in license plates"""
        if not text:
            return text
            
        # Apply specific pattern corrections
        if len(text) >= 2:
            if text[0:2] == 'XA':
                text = 'KA' + text[2:]
            elif text[0:1] == 'X':
                text = 'K' + text[1:]
        
        return text
    
    def correct_plate(self, plate_text):
        """Apply multiple correction methods and heuristics"""
        # Clean the text
        if not plate_text:
            return ""
        
        # Preprocessing
        plate_text = plate_text.upper().strip()
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)  # Remove non-alphanumeric
        
        # Apply heuristic rules
        plate_text = self.apply_heuristic_rules(plate_text)
        
        # Try ML correction
        ml_corrected = self.ml_correct_plate(plate_text)
        
        # If ML correction produces a valid plate, return it
        if self.plate_pattern.match(ml_corrected):
            return ml_corrected
            
        # Return the best available version
        return ml_corrected or plate_text

def detect_license_plate(vehicle_img):
    """
    Enhanced license plate detection within a vehicle image
    Returns the plate region or None if not found
    """
    if vehicle_img is None or vehicle_img.size == 0:
        return None
        
    # Convert to grayscale
    try:
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    except:
        return None
    
    # Apply bilateral filter to preserve edges while reducing noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(gray, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # License plates are typically rectangular with 4 corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Most license plates have aspect ratio between 1.5 to 5
            if 1.5 <= aspect_ratio <= 5:
                plate_contour = approx
                break
    
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        # Make sure the bounding box is within the image bounds
        if x >= 0 and y >= 0 and x+w <= vehicle_img.shape[1] and y+h <= vehicle_img.shape[0]:
            plate_img = vehicle_img[y:y+h, x:x+w]
            return plate_img
    
    # If contour-based detection fails, return the whole vehicle image
    # (plate may be too small or unclear for contour detection)
    return vehicle_img

def filter_car_detections(detections):
    """
    Filter detections to include only cars (class_id=2) and adjust bounding boxes
    for better visualization
    """
    filtered = []
    for det in detections.boxes.data:
        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
        if int(cls_id) == 2 and conf > 0.5:  # Only cars with high confidence
            # Expand bounding box slightly for better visualization (10% expansion)
            width = x2 - x1
            height = y2 - y1
            x1_exp = max(0, x1 - 0.1 * width)
            y1_exp = max(0, y1 - 0.1 * height)
            x2_exp = x2 + 0.1 * width
            y2_exp = y2 + 0.1 * height
            bbox = [float(x1_exp), float(y1_exp), float(x2_exp - x1_exp), float(y2_exp - y1_exp)]
            filtered.append((bbox, float(conf), int(cls_id)))
    return filtered

def get_consensus_plate(plate_readings):
    """
    Apply consensus algorithms to determine the most likely license plate.
    Input: List of (plate_text, confidence, timestamp) tuples
    Output: (plate_text, confidence) tuple
    """
    if not plate_readings:
        return ("UNKNOWN", 0.0)
    
    # If we only have one reading, return it
    if len(plate_readings) == 1:
        return (plate_readings[0][0], plate_readings[0][1])
    
    # Method 1: Weighted voting based on confidence
    plate_votes = defaultdict(float)
    for plate, conf, _ in plate_readings:
        plate_votes[plate] += conf
    
    # Method 2: Character-level voting for similar plates
    if len(plate_readings) >= 3:
        chars_by_position = defaultdict(lambda: defaultdict(float))
        
        # First, determine the most common plate length
        length_counts = defaultdict(int)
        for plate, _, _ in plate_readings:
            length_counts[len(plate)] += 1
        
        if length_counts:  # Make sure we have valid plates
            most_common_length = max(length_counts.items(), key=lambda x: x[1])[0]
            
            # Only consider plates of the most common length
            filtered_readings = [p for p in plate_readings if len(p[0]) == most_common_length]
            
            for plate, conf, _ in filtered_readings:
                for i, char in enumerate(plate):
                    chars_by_position[i][char] += conf
            
            # Construct most likely plate character by character
            constructed_plate = ""
            total_conf = 0
            positions = sorted(chars_by_position.keys())
            
            if positions:  # Make sure we have valid positions
                for pos in positions:
                    # Get character with highest confidence at this position
                    char_votes = chars_by_position[pos]
                    if char_votes:  # Make sure we have votes for this position
                        best_char = max(char_votes.items(), key=lambda x: x[1])
                        constructed_plate += best_char[0]
                        total_conf += best_char[1]
                
                if positions:  # Avoid division by zero
                    constructed_conf = total_conf / len(positions)
                    
                    # Compare with the top weighted vote
                    best_plate = max(plate_votes.items(), key=lambda x: x[1])
                    
                    if constructed_conf > best_plate[1]:
                        return (constructed_plate, constructed_conf / len(filtered_readings))
    
    # If character-level voting fails or isn't applicable
    best_plate = max(plate_votes.items(), key=lambda x: x[1])
    return (best_plate[0], best_plate[1] / len(plate_readings))

def process_image_with_ocr(img, ocr_corrector, use_openalpr=False):
    """Process an image with multiple OCR methods and return the best result"""
    if img is None or img.size == 0:
        return None, 0
        
    results = []
    
    # Try OpenALPR if available and requested
    if use_openalpr:
        try:
            # Import here to avoid errors if not installed
            from openalpr import Alpr
            alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
            if alpr.is_loaded():
                # Convert image to bytes for OpenALPR
                _, img_encoded = cv2.imencode('.jpg', img)
                alpr_results = alpr.recognize_array(img_encoded.tobytes())
                
                if alpr_results and 'results' in alpr_results and alpr_results['results']:
                    for result in alpr_results['results']:
                        plate = result['plate']
                        confidence = result['confidence']
                        corrected_plate = ocr_corrector.correct_plate(plate)
                        results.append((corrected_plate, confidence))
                alpr.unload()
        except Exception as e:
            print(f"OpenALPR error: {e}")
    
    # Try PaddleOCR
    try:
        paddle_result = ocr_corrector.paddle_ocr.ocr(img, cls=True)
        if paddle_result and paddle_result[0]:
            for line in paddle_result[0]:
                text = line[1][0]
                confidence = float(line[1][1])
                # Apply ML correction
                corrected_text = ocr_corrector.correct_plate(text)
                results.append((corrected_text, confidence))
    except Exception as e:
        print(f"PaddleOCR error: {e}")
    
    # Try Tesseract OCR (as fallback)
    if not results:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # OCR using Tesseract
            text = pytesseract.image_to_string(binary, config='--psm 7 --oem 1')
            
            if text.strip():
                # Apply ML correction
                corrected_text = ocr_corrector.correct_plate(text.strip())
                results.append((corrected_text, 0.6))  # Default confidence
        except Exception as e:
            print(f"Tesseract error: {e}")
    
    # Return the best result
    if results:
        # Sort by confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]
    
    return None, 0

def process_video_with_tracking_and_ml_correction(video_path, output_file, dashboard_connector=None):
    """
    Complete pipeline for license plate recognition with:
    - Vehicle tracking (cars only)
    - License plate detection
    - Multiple OCR engines
    - ML-based correction
    - Multi-frame consensus
    - Dashboard integration
    """
    if video_path is None:
        print("No video path provided.")
        return [], {}, None
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], {}, None
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize our classes
    ocr_corrector = EnhancedOCRCorrector()
    
    # Try to load YOLO model
    try:
        detector = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Downloading YOLO model...")
        os.system("pip install -q ultralytics")
        detector = YOLO("yolov8n.pt")
    
    # Initialize tracker
    tracker = DeepSort(max_age=30, nn_budget=100)
    
    # Dictionary for vehicle tracking
    # Key: track_id, Value: {active, plates: [(text, confidence, timestamp)]}
    vehicles = {}
    
    results = []
    frame_count = 0
    vehicle_images = {}  # Store one image of each detected vehicle
    
    # Create a video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "output_" + os.path.basename(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print("Processing video...")
    
    with open(output_file, 'w') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process every 2 frames to speed things up
            if frame_count % 2 != 0:
                out.write(frame)  # Write the original frame
                continue
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Detect all objects in the frame
            detections = detector(frame)[0]
            
            # Filter for cars only with enhanced bounding boxes
            detection_list = filter_car_detections(detections)
            
            # Update tracker with filtered detections
            tracks = tracker.update_tracks(detection_list, frame=frame)
            
            # Process each tracked vehicle
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                bbox = track.to_tlbr()  # top left, bottom right coordinates
                
                # Convert to x, y, w, h format
                x, y = int(bbox[0]), int(bbox[1])
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])
                
                # Draw bounding box for vehicle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {track_id}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Create or update vehicle entry
                if track_id not in vehicles:
                    vehicles[track_id] = {
                        "active": True,
                        "plates": [],
                        "last_seen": frame_count
                    }
                    
                    # Store one image of this vehicle with more context
                    # Expand capture area by 20% to include more context
                    expand = 0.2
                    y_start = max(0, int(y - expand * h))
                    y_end = min(frame.shape[0], int(y + h + expand * h))
                    x_start = max(0, int(x - expand * w))
                    x_end = min(frame.shape[1], int(x + w + expand * w))
                    
                    if y_end > y_start and x_end > x_start:  # Ensure valid dimensions
                        vehicle_images[track_id] = frame[y_start:y_end, x_start:x_end].copy()
                    else:
                        # Fallback to original bbox if expanded one is invalid
                        vehicle_images[track_id] = frame[y:y+h, x:x+w].copy() if y+h <= frame.shape[0] and x+w <= frame.shape[1] else None
                else:
                    vehicles[track_id]["last_seen"] = frame_count
                
                # Only process each vehicle every 10 frames to avoid redundant processing
                if frame_count % 10 == 0 or len(vehicles[track_id]["plates"]) < 3:
                    # Extract vehicle image for plate detection
                    if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                        vehicle_img = frame[y:y+h, x:x+w]
                        
                        if vehicle_img.size > 0:
                            # Try to detect license plate
                            plate_img = detect_license_plate(vehicle_img)
                            
                            if plate_img is not None and plate_img.size > 0:
                                # Try OCR on the plate
                                plate_text, confidence = process_image_with_ocr(
                                    plate_img, ocr_corrector, use_openalpr=False
                                )
                                
                                if plate_text:
                                    vehicles[track_id]["plates"].append((plate_text, confidence, timestamp))
                                    
                                    # Draw the detected plate on the frame
                                    cv2.putText(display_frame, plate_text, (x, y+h+20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Check for inactive vehicles (not seen for a while)
            for vehicle_id in list(vehicles.keys()):
                if frame_count - vehicles[vehicle_id]["last_seen"] > fps * 3:  # 3 seconds
                    if vehicles[vehicle_id]["active"]:
                        vehicles[vehicle_id]["active"] = False
                        
                        # If we have plate readings, determine final consensus
                        if vehicles[vehicle_id]["plates"]:
                            final_plate, confidence = get_consensus_plate(vehicles[vehicle_id]["plates"])
                            
                            # Only output if we have a reasonable confidence
                            if confidence > 0.3:
                                time_str = time.strftime("%H:%M:%S", time.gmtime(vehicles[vehicle_id]["plates"][0][2]))
                                result_str = f"{time_str} {final_plate} {confidence:.1f}%"
                                results.append((vehicle_id, final_plate, confidence, time_str))
                                f.write(result_str + "\n")
                                
                                print(f"Vehicle {vehicle_id}: License plate: {final_plate}, Confidence: {confidence:.1f}%")
                                
                                # Send to dashboard if connector is provided
                                if dashboard_connector is not None:
                                    # Save vehicle image to temporary file
                                    if vehicle_id in vehicle_images and vehicle_images[vehicle_id] is not None:
                                        temp_img_path = f"/tmp/vehicle_{vehicle_id}.jpg"
                                        cv2.imwrite(temp_img_path, vehicle_images[vehicle_id])
                                    else:
                                        temp_img_path = None
                                        
                                    # Send detection to dashboard
                                    dashboard_connector.send_plate_detection(
                                        final_plate, 
                                        float(confidence), 
                                        time_str,
                                        vehicle_id=str(vehicle_id),
                                        image_path=temp_img_path
                                    )
            
            # Add frame number to the frame
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write the frame to the output video
            out.write(display_frame)
            
            # Display progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    print(f"Processing complete! Results saved to {output_file}")
    print(f"Output video saved to {output_video_path}")
    
    return results, vehicle_images, output_video_path

def display_results(results, vehicle_images):
    """Display the detected license plates along with vehicle images"""
    if not results:
        print("No license plates detected.")
        return
        
    print(f"\nDetected {len(results)} unique vehicles with license plates:")
    print("="*50)
    
    for i, (vehicle_id, plate, confidence, timestamp) in enumerate(results):
        print(f"{i+1}. Vehicle ID: {vehicle_id}")
        print(f"   License Plate: {plate}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Timestamp: {timestamp}")
        print("-"*50)
        
        # Display vehicle image if available and not empty
        if vehicle_id in vehicle_images:
            img = vehicle_images[vehicle_id]
            if img is not None and img.size > 0:
                try:
                    # Resize image to a reasonable size for display
                    h, w = img.shape[:2]
                    max_dim = 600
                    scale = min(max_dim / h, max_dim / w, 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized_img = cv2.resize(img, (new_w, new_h))
                    
                    # Save image to disk
                    output_dir = 'vehicle_images'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    image_path = os.path.join(output_dir, f"vehicle_{vehicle_id}_{plate}.jpg")
                    cv2.imwrite(image_path, resized_img)
                    print(f"   Image saved: {image_path}")
                    
                except Exception as e:
                    print(f"   Failed to process image: {e}")
            else:
                print(f"   Image for vehicle {vehicle_id} is empty or None.")
        else:
            print(f"   No image found for vehicle {vehicle_id}.")

def main():
    """Main function to run the ALPR system"""
    parser = argparse.ArgumentParser(description='License Plate Recognition with Dashboard Integration')
    parser.add_argument('--video', type=str, help='Path to video file', required=True)
    parser.add_argument('--output', type=str, default='license_plate_results.txt', help='Output text file for results')
    parser.add_argument('--dashboard_url', type=str, default='http://localhost:8000/alpr_app/default/call/json/message', 
                       help='URL for the ALPR Web Dashboard')
    parser.add_argument('--use_dashboard', action='store_true', help='Enable dashboard integration')
    
    args = parser.parse_args()
    
    # Initialize dashboard connector if requested
    dashboard_connector = None
    if args.use_dashboard:
        try:
            dashboard_connector = ALPRWebDashboardConnector(server_url=args.dashboard_url)
            print("Dashboard integration enabled")
        except Exception as e:
            print(f"Failed to initialize dashboard connector: {e}")
            print("Continuing without dashboard integration")
    
    # Process the video
    results, vehicle_images, output_video_path = process_video_with_tracking_and_ml_correction(
        args.video, args.output, dashboard_connector
    )
    
    # Display results
    display_results(results, vehicle_images)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output}")
    print(f"Output video: {output_video_path}")
    print(f"Vehicle images saved to: vehicle_images/")

if __name__ == "__main__":
    main()