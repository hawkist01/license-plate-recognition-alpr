#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask-based Web Server for LPR Web Application Backend
Subscribes to ROS topics for license plate data and annotated images,
and provides API endpoints to access the latest data.
"""

import flask
from flask import Flask, jsonify
import rospy
from std_msgs.msg import String as RosString # Renamed to avoid conflict with Python's String
from sensor_msgs.msg import Image as RosImage # Renamed to avoid conflict
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
import json # For parsing the string data if needed, or for default empty JSON

# --- Global Variables & Threading Lock ---
latest_plate_data_json_str = "{}" # Store as JSON string directly
latest_annotated_image_cv = None # Store as OpenCV image
data_lock = threading.Lock() # To ensure thread-safe access to global variables

# --- Flask App Initialization ---
app = Flask(__name__)

# --- CvBridge Initialization ---
bridge = CvBridge()

# --- ROS Callback Functions ---
def license_plate_data_callback(ros_string_msg):
    global latest_plate_data_json_str
    rospy.loginfo(f"Received license plate data: {ros_string_msg.data[:100]}...") # Log snippet
    with data_lock:
        latest_plate_data_json_str = ros_string_msg.data

def annotated_image_callback(ros_image_msg):
    global latest_annotated_image_cv
    rospy.loginfo("Received annotated image.")
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding="bgr8")
        with data_lock:
            latest_annotated_image_cv = cv_image
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error converting image: {e}")
    except Exception as e:
        rospy.logerr(f"Error processing image callback: {e}")

# --- API Endpoints ---
@app.route('/api/latest_plate', methods=['GET'])
def get_latest_plate():
    global latest_plate_data_json_str
    with data_lock:
        # Return the string directly, as it's already a JSON string
        # To ensure it's valid JSON if it were an object:
        try:
            # Attempt to parse to ensure it's valid JSON before sending
            # For this app, we trust the source or send as is.
            # json.loads(latest_plate_data_json_str) # Optional validation
            response_data = latest_plate_data_json_str
        except json.JSONDecodeError:
            rospy.logwarn("latest_plate_data_json_str is not valid JSON. Returning empty JSON.")
            response_data = "{}"
    
    # Flask's jsonify is for dicts/lists. For an already formatted JSON string,
    # create a Response object directly.
    return flask.Response(response_data, mimetype='application/json')

@app.route('/api/annotated_image', methods=['GET'])
def get_annotated_image():
    global latest_annotated_image_cv
    with data_lock:
        if latest_annotated_image_cv is None:
            rospy.logwarn("Annotated image requested but not available yet.")
            return "No image available yet.", 404
        
        img_copy = latest_annotated_image_cv.copy() # Work with a copy

    if img_copy is None or img_copy.size == 0: # Should be caught by the first check, but good practice
        return "No image available yet.", 404

    try:
        # Encode the image to JPEG format
        ret, buffer = cv2.imencode('.jpg', img_copy)
        if not ret:
            rospy.logerr("Failed to encode image to JPEG.")
            return "Failed to encode image", 500
        
        image_bytes = buffer.tobytes()
        
        return flask.Response(image_bytes, mimetype='image/jpeg')
        
    except Exception as e:
        rospy.logerr(f"Error encoding or sending image: {e}")
        return "Server error while processing image", 500

# --- ROS Spinning Thread ---
def ros_spin_thread_func():
    rospy.loginfo("ROS spin thread started.")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt. ROS spin thread shutting down.")
    except Exception as e:
        rospy.logerr(f"Exception in ROS spin thread: {e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        rospy.init_node('lpr_web_subscriber', anonymous=True)
        rospy.loginfo("LPR Web Backend ROS node initialized.")

        # Get ROS Parameters
        license_plate_topic = rospy.get_param('~license_plate_data_topic', '/lpr/license_plate_data')
        annotated_image_topic = rospy.get_param('~annotated_image_topic', '/lpr/annotated_image')
        flask_host = rospy.get_param('~flask_host', '0.0.0.0')
        flask_port = rospy.get_param('~flask_port', 5000)

        rospy.loginfo(f"Subscribing to license plate data on: {license_plate_topic}")
        rospy.loginfo(f"Subscribing to annotated images on: {annotated_image_topic}")
        rospy.loginfo(f"Flask server will run on: {flask_host}:{flask_port}")

        # ROS Subscribers
        rospy.Subscriber(license_plate_topic, RosString, license_plate_data_callback)
        rospy.Subscriber(annotated_image_topic, RosImage, annotated_image_callback, queue_size=1, buff_size=2**24) # queue_size and buff_size like in lpr_node

        # Start ROS spinning in a daemon thread
        ros_thread = threading.Thread(target=ros_spin_thread_func)
        ros_thread.daemon = True  # Allows main program to exit even if this thread is still running
        ros_thread.start()
        rospy.loginfo("ROS spin thread initiated.")

        # Start Flask server
        # Setting debug=False as per instructions, which is generally better for threaded apps.
        # use_reloader=False is also often recommended when mixing Flask with background threads if issues arise.
        rospy.loginfo("Starting Flask server...")
        app.run(host=flask_host, port=flask_port, debug=False, use_reloader=False) # Added use_reloader=False

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down LPR Web Backend.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in LPR Web Backend main: {e}")
    finally:
        rospy.loginfo("LPR Web Backend has shut down.")
