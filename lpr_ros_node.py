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
# import argparse # Removed
import pytesseract
from collections import defaultdict

from collections import defaultdict # Added
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String # Will be String from std_msgs
from cv_bridge import CvBridge, CvBridgeError

# Custom imports for SORT
from sort_tracker.sort import Sort # Added

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# ===== ALPR Web Dashboard Connector =====
# Note: This class can remain mostly as-is
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
        # This can be made a ROS parameter if desired, but not required by the task
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

class LPRNode:
    def __init__(self):
        rospy.init_node('lpr_ros_node', anonymous=True)
        rospy.loginfo("LPR ROS Node started.")

        self.bridge = CvBridge()
        self.ocr_corrector = SimpleOCRCorrector()
        self.frame_count = 0
        # self.vehicle_counter = 0 # Replaced by track_id from SORT

        # --- Get ROS Parameters ---
        # Image topics
        self.input_image_topic = rospy.get_param('~input_image_topic', '/camera/image_raw')
        self.annotated_image_topic = rospy.get_param('~annotated_image_topic', '/lpr/annotated_image')
        # License plate data topic (Note: publishing logic will change)
        self.license_plate_data_topic = rospy.get_param('~license_plate_data_topic', '/lpr/license_plate_data') # Keep for now, but might be removed if not used directly
        
        # Frame processing
        self.skip_frames = rospy.get_param('~skip_frames', 1) # Default to 1 to process more frames for tracking

        # YOLOv8 Detector parameters
        self.yolo_model_path = rospy.get_param('~yolo_model_path', 'yolov8n.onnx')
        self.yolo_conf_threshold = rospy.get_param('~yolo_conf_threshold', 0.4)
        self.yolo_nms_threshold = rospy.get_param('~yolo_nms_threshold', 0.5)
        self.yolo_input_width = rospy.get_param('~yolo_input_width', 640)
        self.yolo_input_height = rospy.get_param('~yolo_input_height', 640)
        self.yolo_input_shape = (self.yolo_input_width, self.yolo_input_height)
        # COCO class names. 'car' is typically ID 2.
        self.yolo_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        } # yapf: disable
        try:
            self.yolo_detector = cv2.dnn.readNetFromONNX(self.yolo_model_path)
            rospy.loginfo(f"YOLOv8 ONNX model loaded successfully from {self.yolo_model_path}")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLOv8 ONNX model from {self.yolo_model_path}: {e}")
            # Depending on desired behavior, could raise rospy.ROSInterruptException or exit
            return 

        # SORT Tracker parameters
        self.sort_max_age = rospy.get_param('~sort_max_age', 20)
        self.sort_min_hits = rospy.get_param('~sort_min_hits', 3)
        self.sort_iou_threshold = rospy.get_param('~sort_iou_threshold', 0.3)
        self.mot_tracker = Sort(max_age=self.sort_max_age, min_hits=self.sort_min_hits, iou_threshold=self.sort_iou_threshold)

        # Data storage for tracking, consensus and caching
        self.tracked_plates_data = defaultdict(list) # Stores (plate_text, ocr_confidence, timestamp) for each track_id
        self.vehicle_image_cache = {} # Optional: could store recent vehicle crops for dashboard
        self.consensus_plates = {} # Stores the final consensus plate for a track_id: {track_id: (plate_text, confidence, timestamp)}
        self.last_seen_frames = {} # Stores the frame_count when a track_id was last seen: {track_id: frame_count}

        # Consensus Logic Parameters
        self.consensus_min_readings = rospy.get_param('~consensus_min_readings', 3)
        self.consensus_max_track_staleness_frames = rospy.get_param('~consensus_max_track_staleness_frames', 10)

        # Dashboard (if used, might need to be adapted for tracked data)
        self.dashboard_url = rospy.get_param('~dashboard_url', 'http://localhost:8000/alpr_app/default/call/json/message')
        self.use_dashboard = rospy.get_param('~use_dashboard', False)
        self.dashboard_connector = None
        if self.use_dashboard:
            try:
                self.dashboard_connector = ALPRWebDashboardConnector(server_url=self.dashboard_url)
                rospy.loginfo(f"Dashboard integration enabled for URL: {self.dashboard_url}")
            except Exception as e:
                rospy.logerr(f"Failed to initialize dashboard connector: {e}")
        
        # --- Publishers ---
        self.annotated_image_pub = rospy.Publisher(self.annotated_image_topic, Image, queue_size=10)
        # Ensure the license plate data publisher is correctly initialized
        self.plate_data_publisher = rospy.Publisher(self.license_plate_data_topic, String, queue_size=10) 

        # --- Subscriber ---
        self.image_sub = rospy.Subscriber(self.input_image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"LPR Node Initialized. Listening to {self.input_image_topic}")
        rospy.loginfo(f"YOLO Model: {self.yolo_model_path}, Conf: {self.yolo_conf_threshold}, NMS: {self.yolo_nms_threshold}")
        rospy.loginfo(f"SORT Tracker: MaxAge: {self.sort_max_age}, MinHits: {self.sort_min_hits}, IOU: {self.sort_iou_threshold}")
        rospy.loginfo(f"Consensus Params: MinReadings: {self.consensus_min_readings}, MaxStaleness: {self.consensus_max_track_staleness_frames}")

    def _apply_consensus_logic(self, track_id):
        if track_id not in self.tracked_plates_data:
            return None

        readings = self.tracked_plates_data[track_id]
        if len(readings) < self.consensus_min_readings:
            rospy.logdebug(f"Track {track_id}: Not enough readings ({len(readings)}) for consensus, need {self.consensus_min_readings}.")
            return None

        corrected_plate_texts = [self.ocr_corrector.correct_plate(r[0]) for r in readings]
        
        if not corrected_plate_texts:
            return None

        # Count frequencies
        plate_counts = defaultdict(int)
        for plate_text in corrected_plate_texts:
            if plate_text: # Ensure non-empty strings
                plate_counts[plate_text] += 1
        
        if not plate_counts:
            rospy.logdebug(f"Track {track_id}: No valid corrected plate texts found for consensus.")
            return None

        # Determine best plate text (highest frequency)
        # Simple tie-breaking: sorted by plate text alphabetically if frequencies are equal
        sorted_plates_by_freq = sorted(plate_counts.items(), key=lambda item: (-item[1], item[0]))
        best_plate_text = sorted_plates_by_freq[0][0]
        
        # Calculate aggregated confidence for the best plate text
        total_confidence = 0
        count_for_confidence_avg = 0
        for i, original_reading in enumerate(readings):
            if corrected_plate_texts[i] == best_plate_text:
                total_confidence += original_reading[1] # original_reading[1] is ocr_confidence
                count_for_confidence_avg += 1
        
        aggregated_confidence = total_confidence / count_for_confidence_avg if count_for_confidence_avg > 0 else 0

        rospy.loginfo(f"Consensus for track {track_id}: {best_plate_text} with aggregated confidence {aggregated_confidence:.2f} from {count_for_confidence_avg} readings.")
        
        # Store this consensus result with current timestamp
        self.consensus_plates[track_id] = (best_plate_text, aggregated_confidence, rospy.Time.now())
        
        return (best_plate_text, aggregated_confidence)

    def _extract_yolo_detections(self, yolo_outputs, frame_width, frame_height):
        # YOLOv8 output shape is (1, 84, num_proposals). Transpose to (num_proposals, 84).
        # 84 = 4 (box: cx, cy, w, h) + 80 (class scores for COCO)
        
        # Check if yolo_outputs is a list (common if multiple output layers are technically present even if only one is used)
        if isinstance(yolo_outputs, list):
            outputs = yolo_outputs[0] # Assuming the first output layer is the one we need
        else:
            outputs = yolo_outputs

        proposals = outputs[0].T # Transpose from (84, num_proposals) to (num_proposals, 84)
        
        boxes_xywh = [] # For NMS: [x, y, w, h] format
        scores = []     # For NMS: corresponding scores
        class_ids = []  # For NMS (optional, but good to keep)

        # Scale factors
        x_scale = frame_width / self.yolo_input_width
        y_scale = frame_height / self.yolo_input_height

        for proposal in proposals:
            box = proposal[0:4] # cx, cy, w, h (normalized to yolo_input_shape)
            class_scores = proposal[4:]
            
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # Filter by confidence and class (e.g., 'car' which is ID 2 in COCO)
            if confidence > self.yolo_conf_threshold and self.yolo_classes.get(class_id) == 'car':
                # Convert cx, cy, w, h (model input scale) to x, y, w, h (original frame scale)
                cx, cy, w, h = box
                
                # Center to top-left
                x = (cx - w / 2) * x_scale
                y = (cy - h / 2) * y_scale
                scaled_w = w * x_scale
                scaled_h = h * y_scale
                
                boxes_xywh.append([int(x), int(y), int(scaled_w), int(scaled_h)])
                scores.append(float(confidence))
                class_ids.append(class_id) # Keep track if needed later

        if not boxes_xywh:
            return np.empty((0, 5))

        # Apply Non-Maximum Suppression
        # NMSBoxes returns indices of the boxes to keep
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.yolo_conf_threshold, self.yolo_nms_threshold)

        detections_for_sort = []
        if len(indices) > 0:
            # In case indices is a nested list/tuple, flatten it
            if isinstance(indices, (list, tuple)) and len(indices) > 0 and isinstance(indices[0], (list, tuple)):
                indices = indices[0]

            for i in indices:
                x, y, w, h = boxes_xywh[i]
                score = scores[i]
                # Format for SORT: [x1, y1, x2, y2, score]
                detections_for_sort.append([x, y, x + w, y + h, score])
        
        return np.array(detections_for_sort)

    def detect_license_plate(self, vehicle_img):
        """
        Enhanced license plate detection within a vehicle image
        Returns the plate region or None if not found
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None
            
        # Convert to grayscale
        try:
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            rospy.logwarn(f"Could not convert vehicle image to gray: {e}")
            return None
        
        # Apply bilateral filter to preserve edges while reducing noise
        # Parameters for bilateralFilter: d, sigmaColor, sigmaSpace
        # Using values that are common, but might need tuning.
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75) # d=9, sigmaColor=75, sigmaSpace=75
        
        # Edge detection
        edged = cv2.Canny(gray_filtered, 50, 200) # Adjusted Canny thresholds slightly
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Changed to RETR_LIST
        
        if not contours:
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contour = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 1.5 <= aspect_ratio <= 5: # Aspect ratio for license plates
                    plate_contour = approx
                    break
        
        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            if x >= 0 and y >= 0 and x+w <= vehicle_img.shape[1] and y+h <= vehicle_img.shape[0]:
                plate_img = vehicle_img[y:y+h, x:x+w]
                return plate_img
        
        return vehicle_img # Return whole vehicle image if specific plate contour not found


    # This method is no longer used as primary vehicle detection
    # def simple_object_detection(self, frame):
    #     ... (previous code)

    def process_image_with_tesseract(self, img): # Parameter ocr_corrector removed, use self.ocr_corrector
        """Process an image with Tesseract OCR and return the best result"""
        if img is None or img.size == 0:
            return None, 0
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            rospy.logwarn(f"Could not convert plate image to gray for Tesseract: {e}")
            return None, 0
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        configs = [
            '--psm 7 --oem 1',  # Single line of text
            '--psm 8 --oem 1',  # Single word
            '--psm 6 --oem 1'   # Assume a single uniform block of text
        ]
        
        best_text = ""
        best_conf = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(binary, config=config)
                if text.strip():
                    conf_data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
                    current_conf_values = [c for c in conf_data["conf"] if c != -1 and c != '-1'] # Filter out -1 string
                    if current_conf_values:
                        conf = float(max(current_conf_values, default=0))
                    else:
                        conf = 50.0 
                    
                    corrected_text = self.ocr_corrector.correct_plate(text.strip()) # Use self.ocr_corrector
                    
                    if conf > best_conf: # Ensure conf is float for comparison
                        best_text = corrected_text
                        best_conf = float(conf)
            except Exception as e:
                rospy.logwarn(f"Tesseract error with config {config}: {e}")
                continue
        
        normalized_conf = min(best_conf / 100.0, 1.0)
        return best_text, normalized_conf

    def image_callback(self, ros_image_msg):
        self.frame_count += 1
        
        # Skip frame processing if needed
        if self.frame_count % self.skip_frames != 0:
            # As per instructions, do not publish if skipped.
            # If a consistent frame rate is needed on annotated topic,
            # we could publish the original image here.
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Timestamp for ROS messages
        # Use the image message header's timestamp if available, otherwise use current ROS time
        if ros_image_msg.header.stamp and ros_image_msg.header.stamp.secs > 0:
            current_ros_time = ros_image_msg.header.stamp
        else:
            current_ros_time = rospy.Time.now()

        display_frame = cv_image.copy() # For drawing annotations
        frame_height, frame_width = cv_image.shape[:2]

        # --- Object Detection (YOLOv8n) ---
        blob = cv2.dnn.blobFromImage(cv_image, 1/255.0, self.yolo_input_shape, swapRB=True, crop=False)
        self.yolo_detector.setInput(blob)
        
        try:
            # Forward pass through YOLO. Output name might be needed if model has multiple output layers.
            # For many ONNX models, getUnconnectedOutLayersNames() helps, or direct output if single.
            # The specific output layer name might be 'output0' or similar.
            # If yolo_detector.getUnconnectedOutLayersNames() is empty or causes issues,
            # try yolo_outputs = self.yolo_detector.forward() and inspect its structure.
            output_layer_names = self.yolo_detector.getUnconnectedOutLayersNames()
            if output_layer_names:
                 yolo_outputs = self.yolo_detector.forward(output_layer_names)
            else: # Fallback if no unconnected layers are found (e.g. some optimized models)
                 yolo_outputs = [self.yolo_detector.forward()] # Wrap in a list to match expected structure
        except Exception as e:
            rospy.logerr(f"Error during YOLOv8 forward pass: {e}")
            # Publish the original image if detection fails, to keep the annotated topic flowing
            if self.annotated_image_pub.get_num_connections() > 0:
                try:
                    error_ros_image = self.bridge.cv2_to_imgmsg(display_frame, "bgr8")
                    error_ros_image.header.stamp = current_ros_time
                    self.annotated_image_pub.publish(error_ros_image)
                except CvBridgeError as cve:
                    rospy.logerr(f"CvBridge Error publishing error image: {cve}")
            return

        detections_for_sort = self._extract_yolo_detections(yolo_outputs, frame_width, frame_height)

        # --- SORT Tracking ---
        if detections_for_sort.ndim == 2 and detections_for_sort.shape[1] == 5: # Ensure it's a 2D array with 5 columns
            tracked_objects = self.mot_tracker.update(detections_for_sort)
        else:
            # Pass empty array with correct shape if no valid detections
            tracked_objects = self.mot_tracker.update(np.empty((0, 5))) 

        # --- Process Tracked Objects ---
        current_active_track_ids = set()
        if tracked_objects.size > 0: # Ensure tracked_objects is not empty
            current_active_track_ids = {int(t[-1]) for t in tracked_objects}

        for track_id in current_active_track_ids:
            self.last_seen_frames[track_id] = self.frame_count

        for track in tracked_objects: # track is [x1, y1, x2, y2, track_id]
            x1, y1, x2, y2, track_id_float = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Convert coords to int
            track_id = int(track_id_float) # Convert track_id to int
            
            # Store latest OCR for this frame for potential drawing
            latest_ocr_in_frame = ""

            # Ensure coordinates are within frame bounds (already done by _extract_yolo_detections for initial box, but good for safety)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)

            if x1 >= x2 or y1 >= y2: # Skip if box is invalid
                continue

            vehicle_crop = cv_image[y1:y2, x1:x2]

            if vehicle_crop.size > 0:
                plate_img = self.detect_license_plate(vehicle_crop)
                if plate_img is not None and plate_img.size > 0:
                    # Pass self.ocr_corrector explicitly if method doesn't use self.
                    plate_text, ocr_confidence = self.process_image_with_tesseract(plate_img) 
                    
                    if plate_text and len(plate_text) >= 2 and ocr_confidence > 0.1: # Lowered threshold for initial capture
                        # Store data associated with track_id
                        # Using image header stamp for consistency if available
                        self.tracked_plates_data[track_id].append((plate_text, ocr_confidence, current_ros_time))
                        latest_ocr_in_frame = plate_text # Store for drawing
                        
                        # Example: Dashboard integration for *each valid OCR*
                        if self.dashboard_connector:
                            dt_object = datetime.fromtimestamp(current_ros_time.to_sec())
                            dashboard_ts_str = dt_object.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            self.dashboard_connector.send_plate_detection(
                                plate_text, float(ocr_confidence), dashboard_ts_str,
                                vehicle_id=f"track_{track_id}" 
                                # image_path=temp_plate_img_path # if saving plate images
                            )

            # Draw vehicle bounding box and track ID
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for vehicle box
            
            display_text_for_plate = ""
            if track_id in self.consensus_plates:
                # Display final consensus plate if available
                display_text_for_plate = f"Final LP: {self.consensus_plates[track_id][0]}"
            elif latest_ocr_in_frame: # Display raw OCR if available for this frame
                display_text_for_plate = f"Raw LP: {latest_ocr_in_frame}"
            # else: no plate text to display for this vehicle in this frame

            cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if display_text_for_plate:
                cv2.putText(display_frame, display_text_for_plate, (x1, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # Magenta for plate text


        # --- Publish Annotated Image ---
        cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.annotated_image_pub.get_num_connections() > 0:
            try:
                annotated_ros_image = self.bridge.cv2_to_imgmsg(display_frame, "bgr8")
                annotated_ros_image.header.stamp = current_ros_time
                self.annotated_image_pub.publish(annotated_ros_image)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error publishing annotated image: {e}")
            except Exception as e:
                rospy.logerr(f"Error publishing annotated image: {e}")
        
        # --- Consensus Triggering and Publishing for Stale/Lost Tracks ---
        stale_tracks_to_process = []
        # Iterate over a copy of keys if modifying dict during iteration
        for tid in list(self.tracked_plates_data.keys()): 
            is_stale = (self.frame_count - self.last_seen_frames.get(tid, self.frame_count)) > self.consensus_max_track_staleness_frames
            if tid not in current_active_track_ids or is_stale:
                stale_tracks_to_process.append(tid)
        
        for track_id_to_process in stale_tracks_to_process:
            rospy.logdebug(f"Track {track_id_to_process} identified as stale or lost. Attempting consensus.")
            consensus_result = self._apply_consensus_logic(track_id_to_process) # This stores result in self.consensus_plates
            
            if consensus_result is not None:
                best_plate_text, aggregated_confidence = consensus_result
                # Retrieve the timestamp stored by _apply_consensus_logic
                # self.consensus_plates[track_id] = (best_plate_text, aggregated_confidence, rospy.Time.now())
                consensus_timestamp = self.consensus_plates[track_id_to_process][2]

                plate_event_data = {
                    "vehicle_id": str(track_id_to_process),
                    "plate_text": best_plate_text,
                    "confidence": float(aggregated_confidence), # Ensure it's float
                    "timestamp": consensus_timestamp.to_sec() # ROS time to seconds
                }
                json_data = json.dumps(plate_event_data)
                
                try:
                    self.plate_data_publisher.publish(String(json_data)) # Ensure msg type is correct
                    rospy.loginfo(f"Published consensus for track {track_id_to_process}: {json_data}")
                except Exception as e:
                    rospy.logerr(f"Failed to publish consensus data for track {track_id_to_process}: {e}")

                # Cleanup raw data after successful consensus and publishing attempt
                # If the track reappears, it will start collecting new data in tracked_plates_data.
                if track_id_to_process in self.tracked_plates_data:
                    del self.tracked_plates_data[track_id_to_process]
                if track_id_to_process in self.last_seen_frames:
                    del self.last_seen_frames[track_id_to_process]
            else: # No consensus reached
                # If track is truly lost (not just stale but active), clear its data
                if track_id_to_process not in current_active_track_ids:
                    rospy.logdebug(f"Clearing data for lost track {track_id_to_process} as no consensus was reached.")
                    if track_id_to_process in self.tracked_plates_data:
                        del self.tracked_plates_data[track_id_to_process]
                    if track_id_to_process in self.last_seen_frames:
                        del self.last_seen_frames[track_id_to_process]
                    # Optionally, also clear from self.consensus_plates if it contained an old/stale consensus for this ID
                    # if track_id_to_process in self.consensus_plates:
                    #     del self.consensus_plates[track_id_to_process]


if __name__ == "__main__":
    try:
        lpr_node = LPRNode()
        # Check if initialization failed (e.g. YOLO model load)
        if hasattr(lpr_node, 'yolo_detector') and lpr_node.yolo_detector is not None:
            rospy.spin()
        else:
            rospy.logerr("LPRNode initialization failed. Shutting down.")
    except rospy.ROSInterruptException:
        rospy.loginfo("LPR ROS node shutting down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in LPR ROS Node: {e}", exc_info=True) # Add exc_info for traceback
