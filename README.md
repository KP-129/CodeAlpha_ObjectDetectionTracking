# VisionTrack - Object Detection and Tracking

## Real-Time AI-Powered Object Detection and Tracking Web Application

VisionTrack is a real-time computer vision application that detects objects from a webcam using a pretrained YOLO11 model and assigns tracking IDs to detected objects using a lightweight IoU-based SORT-style tracking approach.

The application provides a browser-based interface built with Flask, allowing users to start and stop the webcam and view annotated detection results in real time.

---

## Features

- Real-time webcam input
- Pretrained YOLO11 object detection
- Bounding boxes around detected objects
- Object class labels
- Confidence scores
- Persistent tracking IDs
- Lightweight IoU-based SORT-style tracking
- Active object counter
- Browser-based interface
- Start and Stop camera controls
- Responsive user interface

---

## Technology Stack

- Python
- Flask
- OpenCV
- YOLO11
- Ultralytics
- NumPy
- HTML
- CSS
- JavaScript

---

## Project Structure

```text
Task - 4 - Object Detection Tracking/
│
├── app.py
├── detector.py
├── object_detection_tracking.py
├── requirements.txt
├── README.md
├── .gitignore
├── yolo11n.pt
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css