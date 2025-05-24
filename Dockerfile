# Start from a ROS Noetic base image
FROM ros:noetic-ros-base-focal

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies 
# python3-flask and python3-requests are used by the webapp/lpr_web_backend.py
# rospy and cv_bridge (for sensor_msgs) are part of ros-base.
RUN apt-get update && apt-get install -y \
    python3-flask \
    python3-requests \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the webapp directory (containing the backend and frontend) into the container
COPY ./webapp ./webapp

# Expose the port the Flask app runs on (default specified in lpr_web_backend.py)
EXPOSE 5000

# Command to run the application
# The script webapp/lpr_web_backend.py will be executed.
CMD ["python3", "webapp/lpr_web_backend.py"]
