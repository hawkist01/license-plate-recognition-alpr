#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified License Plate Recognition System for Raspberry Pi 4
This is a lightweight version of the original script with fewer dependencies.
"""

import cv2
import numpy as np
import os
import time
import re
import json
from datetime import datetime
import requests
import argparse
import pytesseract
from collections import defaultdict

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

# ===== Simplified OCR Corrector =====
class SimpleOCRCorrector:
    """
    Basic rule-based correction for OCR results
    """
    def __init__(self):
        # Regular expression pattern for license plates (customize for your region)
        self.plate_pattern = re.compile(r'^[A-Z0-9]{2,8}$')
        
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
            
        # Replace common OCR mistakes
        text = text.replace('O', '0')
        text = text.replace('Q', '0')
        text = text.replace('I', '1')
        text = text.replace('S', '5')
        text = text.replace('B', '8')
        
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
            
        return plate_text

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

def simple_object_detection(frame, target_size=(640, 640)):
    """
    A simpler vehicle detection based on background subtraction and contours
    """
    # Create a copy of the original frame
    output_frame = frame.copy()
    
    # Resize for faster processing
    height, width = frame.shape[:2]
    scale = min(target_size[0] / width, target_size[1] / height)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
    else:
        resized_frame = frame.copy()
        new_height, new_width = height, width
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find edges
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = (new_width * new_height) * 0.005  # Minimum 0.5% of frame
    max_area = (new_width * new_height) * 0.5   # Maximum 50% of frame
    
    potential_vehicles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Additional filter: reasonable aspect ratio for vehicles
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.5:  # Most vehicles are between 0.5 and 2.5 aspect ratio
                # Scale back to original frame size
                if scale < 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                
                # Add some margin
                x = max(0, x - int(0.1 * w))
                y = max(0, y - int(0.1 * h))
                w = min(width - x, int(1.2 * w))
                h = min(height - y, int(1.2 * h))
                
                potential_vehicles.append((x, y, w, h))
                
                # Draw rectangle on the original frame
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return potential_vehicles, output_frame

def process_image_with_tesseract(img, ocr_corrector):
    """Process an image with Tesseract OCR and return the best result"""
    if img is None or img.size == 0:
        return None, 0
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None, 0
        
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Try multiple configurations and select best result
    configs = [
        '--psm 7 --oem 1',  # Single line of text
        '--psm 8 --oem 1',  # Single word
        '--psm 6 --oem 1'   # Assume a single uniform block of text
    ]
    
    best_text = ""
    best_conf = 0
    
    for config in configs:
        try:
            # OCR using Tesseract
            text = pytesseract.image_to_string(binary, config=config)
            
            if text.strip():
                # Use Tesseract confidence if available
                conf_data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
                if conf_data["conf"] and len(conf_data["conf"]) > 0:
                    conf = float(max([c for c in conf_data["conf"] if c != -1], default=0))
                else:
                    conf = 50.0  # Default confidence
                
                # Apply correction
                corrected_text = ocr_corrector.correct_plate(text.strip())
                
                if conf > best_conf:
                    best_text = corrected_text
                    best_conf = conf
        except Exception as e:
            print(f"Tesseract error: {e}")
            continue
    
    # Normalize confidence to 0-1 range
    normalized_conf = min(best_conf / 100.0, 1.0)
    
    return best_text, normalized_conf

def process_video_for_license_plates(video_path, output_file, dashboard_connector=None, skip_frames=3):
    """
    Simplified pipeline for license plate recognition on Raspberry Pi
    """
    if video_path is None:
        print("No video path provided.")
        return [], {}
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return [], {}
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize OCR corrector
    ocr_corrector = SimpleOCRCorrector()
    
    # Dictionary to store detected vehicles and their license plates
    vehicles = {}
    vehicle_counter = 0
    
    results = []
    frame_count = 0
    vehicle_images = {}  # Store one image of each detected vehicle
    
    # Create a video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for better compatibility
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
            
            # Process every N frames to speed things up
            if frame_count % skip_frames != 0:
                out.write(frame)  # Write the original frame
                continue
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Detect potential vehicles
            vehicle_boxes, detection_frame = simple_object_detection(frame)
            
            # Process each detected vehicle
            for i, (x, y, w, h) in enumerate(vehicle_boxes):
                # Extract vehicle image
                vehicle_img = frame[y:y+h, x:x+w]
                
                # Give each detection a unique ID
                vehicle_id = f"v_{frame_count}_{i}"
                vehicle_counter += 1
                
                # Try to detect license plate
                plate_img = detect_license_plate(vehicle_img)
                
                if plate_img is not None and plate_img.size > 0:
                    # Try OCR on the plate
                    plate_text, confidence = process_image_with_tesseract(plate_img, ocr_corrector)
                    
                    if plate_text and len(plate_text) >= 2 and confidence > 0.3:
                        time_str = time.strftime("%H:%M:%S", time.gmtime(timestamp))
                        
                        # Store result
                        result_tuple = (vehicle_id, plate_text, confidence*100, time_str)
                        if result_tuple not in results:
                            results.append(result_tuple)
                            
                            # Save to file
                            result_str = f"{time_str} {plate_text} {confidence*100:.1f}%"
                            f.write(result_str + "\n")
                            
                            print(f"Vehicle {vehicle_id}: License plate: {plate_text}, Confidence: {confidence*100:.1f}%")
                            
                            # Store the vehicle image
                            vehicle_images[vehicle_id] = vehicle_img.copy()
                            
                            # Save plate image
                            os.makedirs("plate_images", exist_ok=True)
                            plate_img_path = f"plate_images/plate_{vehicle_id}.jpg"
                            cv2.imwrite(plate_img_path, plate_img)
                            
                            # Send to dashboard if connector is provided
                            if dashboard_connector is not None:
                                # Save vehicle image to temporary file
                                temp_img_path = f"/tmp/vehicle_{vehicle_id}.jpg"
                                cv2.imwrite(temp_img_path, vehicle_img)
                                    
                                # Send detection to dashboard
                                dashboard_connector.send_plate_detection(
                                    plate_text, 
                                    float(confidence), 
                                    time_str,
                                    vehicle_id=vehicle_id,
                                    image_path=temp_img_path
                                )
                        
                        # Draw the detected plate on the frame
                        cv2.putText(display_frame, plate_text, (x, y+h+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw bounding box for vehicle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {vehicle_id}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
    
    return results, vehicle_images

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
    """Main function to run the simplified ALPR system"""
    parser = argparse.ArgumentParser(description='Simplified License Plate Recognition for Raspberry Pi')
    parser.add_argument('--video', type=str, help='Path to video file', required=True)
    parser.add_argument('--output', type=str, default='license_plate_results.txt', help='Output text file for results')
    parser.add_argument('--dashboard_url', type=str, default='http://localhost:8000/alpr_app/default/call/json/message', 
                       help='URL for the ALPR Web Dashboard')
    parser.add_argument('--use_dashboard', action='store_true', help='Enable dashboard integration')
    parser.add_argument('--skip_frames', type=int, default=3, help='Process every Nth frame (default: 3)')
    
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
    results, vehicle_images = process_video_for_license_plates(
        args.video, args.output, dashboard_connector, args.skip_frames
    )
    
    # Display results
    display_results(results, vehicle_images)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output}")
    print(f"Vehicle images saved to: vehicle_images/")
    print(f"Plate images saved to: plate_images/")

if __name__ == "__main__":
    main()
