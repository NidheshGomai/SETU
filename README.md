# Pothole Detection

A computer vision project for pothole and road damage detection using YOLO and a Flask dashboard.

## Project Overview

This repository contains inference scripts, a live dashboard, evaluation utilities, and supporting files for a pothole detection pipeline.

Key features:
- `dashboard.py`: New Flask dashboard for image/video upload prediction and accuracy metrics display.
- `templates/dashboard.html`: Dashboard UI for live stream preview, image/video uploads, and model accuracy.
- `app.py`: Existing Flask web app showing live video inference with detection count.
- `templates/index.html`: Original dashboard UI for video stream and statistics.
- `test.py` / `v8test.py`: YOLO model inference scripts for validation and testing.
- `show_results.py`: Prints summary statistics from saved JSON test results.
- `generate_full_gallery.py`: Builds an HTML gallery from visualization output images.
- `server.js`: Minimal Express backend example.
- Jupyter notebooks for model exploration and severity detection.

## Repository Structure

- `dashboard.py` - New Flask app for image/video prediction plus accuracy metrics.
- `app.py` - Existing Flask application for live video detection.
- `server.js` - Simple Node.js Express server.
- `test.py` - YOLO inference test runner.
- `v8test.py` - Extended YOLOv8 evaluation script.
- `show_results.py` - Summarizes existing test results.
- `generate_full_gallery.py` - Creates HTML gallery from visualization outputs.
- `templates/dashboard.html` - New web dashboard template for predictions and accuracy.
- `templates/index.html` - Original dashboard template.
- `POTHOLEL_OBJECT_DETECTION.ipynb` - Notebook for object detection exploration.
- `pothole_detection_severity.ipynb` - Notebook for severity detection.
- `*.pt`, `*.pth` - Model weight files.

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the new Flask dashboard

1. Make sure your model file exists in the project root. The app checks for:
   `yoloV8_best.pt`, `best.pt`, or `detection_model.pt`.
2. Place an optional sample video at `sample_video.mp4` in the project root if you want the built-in live feed preview.
3. Run the new dashboard app:

```bash
python dashboard.py
```

4. Open the dashboard in your browser:

```text
http://localhost:5000
```

5. Use the dashboard to:
   - upload an image for pothole detection,
   - upload a video for frame-based detection summary,
   - view stored test accuracy statistics from `test_results/test_results.json`.

### Run the existing Flask app

1. Update the video file path inside `app.py` if needed.
2. Run the app:

```bash
python app.py
```

3. Open the dashboard in your browser:

```text
http://localhost:5000
```

### Run model tests

- `test.py`: Loads `yoloV8_best.pt` and evaluates images from `test/images`.
- `v8test.py`: Alternative inference script with JSON output support.

### Generate result summaries

```bash
python show_results.py
python generate_full_gallery.py
```

## Node.js backend

`server.js` is a minimal Express example. To run it:

```bash
npm install express cors
node server.js
```

Then open:

```text
http://localhost:3000
```
