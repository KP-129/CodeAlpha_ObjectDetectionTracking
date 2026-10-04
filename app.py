from flask import Flask, Response, render_template, jsonify
import cv2

from detector import ObjectDetector


app = Flask(__name__)

camera = None
detector = None
camera_running = False


def initialize_camera():
    global camera

    if camera is None or not camera.isOpened():

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera = None
            return False

    return True


def generate_frames():

    global camera_running

    while camera_running:

        if camera is None or not camera.isOpened():
            break

        success, frame = camera.read()

        if not success:
            break

        processed_frame = detector.process_frame(frame)

        success, buffer = cv2.imencode(
            ".jpg",
            processed_frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start_camera():

    global camera_running, detector

    if not initialize_camera():

        return jsonify({
            "success": False,
            "message": "Could not access the webcam."
        }), 500

    if detector is None:

        detector = ObjectDetector(
            model_path="yolo11n.pt",
            confidence=0.45
        )

    camera_running = True

    return jsonify({
        "success": True,
        "message": "Camera started successfully."
    })


@app.route("/stop")
def stop_camera():

    global camera, camera_running

    camera_running = False

    if camera is not None:

        camera.release()
        camera = None

    return jsonify({
        "success": True,
        "message": "Camera stopped."
    })


@app.route("/video_feed")
def video_feed():

    if not camera_running:

        return Response(
            "Camera is not running.",
            status=503
        )

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    print("=" * 55)
    print("Object Detection & Tracking Web Application")
    print("=" * 55)
    print("Open http://127.0.0.1:5000 in your browser.")
    print("Press CTRL+C in the terminal to stop the server.")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )