import base64
import io
import json
import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Add safe global for ultralytics model loading
torch.serialization.add_safe_globals([DetectionModel])

# Patch torch.load to support older ultralytics weight formats
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    try:
        return _original_torch_load(f, *args, **kwargs)
    except Exception:
        kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# Flask App Initialization
app = Flask(__name__)

# Model and dataset configuration
MODEL_PATHS = [
    "yoloV8_best.pt",
    "best.pt",
    "detection_model.pt",
]
ACCURACY_JSON = os.path.join("test_results", "test_results.json")
DEFAULT_CONFIDENCE = 0.25
VIDEO_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "setu_video_uploads")

os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_model():
    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "No model file found. Expected one of: " + ", ".join(MODEL_PATHS)
        )
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}. device={model.device}")
    return model


def load_accuracy_data():
    if not os.path.exists(ACCURACY_JSON):
        return {
            "available": False,
            "message": "Accuracy metrics not found. Run test scripts to create test_results/test_results.json.",
        }

    with open(ACCURACY_JSON, "r", encoding="utf-8") as file:
        data = json.load(file)

    total_images = data.get("total_images", 0)
    total_ground_truth = data.get("total_ground_truth", 0)
    total_predictions = data.get("total_predictions", 0)
    detection_rate = 0.0
    if total_ground_truth > 0:
        detection_rate = (total_predictions / total_ground_truth) * 100

    return {
        "available": True,
        "loaded_at": datetime.utcnow().isoformat() + "Z",
        "total_images": total_images,
        "total_ground_truth": total_ground_truth,
        "total_predictions": total_predictions,
        "detection_rate": round(detection_rate, 2),
        "images_with_detections": data.get("images_with_detections", 0),
        "class_stats": data.get("class_stats", {}),
    }


class PredictionEngine:
    def __init__(self, model, confidence=DEFAULT_CONFIDENCE):
        self.model = model
        self.confidence = confidence

    def annotate_frame(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        if not results or len(results) == 0:
            return frame, []

        result = results[0]
        detections = []

        if len(result.boxes) > 0:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box.tolist()
                detections.append(
                    {
                        "class_id": int(cls),
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        annotated = result.plot() if hasattr(result, "plot") else frame
        return annotated, detections

    def predict_image(self, image_bytes):
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image file")

        annotated, detections = self.annotate_frame(frame)
        return annotated, detections

    def predict_video(self, video_path, max_frames=120):
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        frame_count = 0
        total_detections = 0
        preview_image = None

        while frame_count < max_frames:
            success, frame = capture.read()
            if not success:
                break

            annotated, detections = self.annotate_frame(frame)
            total_detections += len(detections)
            frame_count += 1

            if preview_image is None and annotated is not None:
                preview_image = annotated

        capture.release()

        average_detections = 0.0
        if frame_count > 0:
            average_detections = total_detections / frame_count

        return preview_image, {
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "average_detections": round(average_detections, 2),
        }


model = load_model()
predictor = PredictionEngine(model)


def image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode annotated image")
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/accuracy_data")
def accuracy_data():
    return jsonify(load_accuracy_data())


@app.route("/predict_image", methods=["POST"])
def predict_image():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"success": False, "message": "No image uploaded."}), 400

    image_bytes = uploaded.read()
    try:
        annotated, detections = predictor.predict_image(image_bytes)
        preview_b64 = image_to_base64(annotated)
        return jsonify(
            {
                "success": True,
                "detections": detections,
                "preview": f"data:image/jpeg;base64,{preview_b64}",
            }
        )
    except Exception as ex:
        return jsonify({"success": False, "message": str(ex)}), 500


@app.route("/predict_video", methods=["POST"])
def predict_video():
    uploaded = request.files.get("video")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"success": False, "message": "No video uploaded."}), 400

    file_name = os.path.basename(uploaded.filename)
    upload_path = os.path.join(VIDEO_UPLOAD_DIR, f"uploaded_{datetime.utcnow().timestamp()}_{file_name}")
    uploaded.save(upload_path)

    try:
        preview_image, stats = predictor.predict_video(upload_path)
        preview_b64 = None
        if preview_image is not None:
            preview_b64 = image_to_base64(preview_image)
        return jsonify(
            {
                "success": True,
                "stats": stats,
                "preview": f"data:image/jpeg;base64,{preview_b64}" if preview_b64 else None,
            }
        )
    except Exception as ex:
        return jsonify({"success": False, "message": str(ex)}), 500
    finally:
        try:
            os.remove(upload_path)
        except Exception:
            pass


@app.route("/video_feed")
def video_feed():
    sample_video = os.path.join(os.getcwd(), "sample_video.mp4")
    if not os.path.exists(sample_video):
        return jsonify({"success": False, "message": "Sample video not found."}), 404

    def frame_generator():
        capture = cv2.VideoCapture(sample_video)
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break

            annotated, _ = predictor.annotate_frame(frame)
            success, buffer = cv2.imencode(".jpg", annotated)
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        capture.release()

    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
