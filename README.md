# PyResearch Pothole Detection

A computer vision project for pothole and road damage detection using YOLO and a Flask dashboard.

## Project Overview

This repository contains inference scripts, a live dashboard, evaluation utilities, and supporting files for a pothole detection pipeline.

Key features:
- `app.py`: Flask web app showing live video inference with detection count.
- `templates/index.html`: Dashboard UI for video stream and statistics.
- `test.py` / `v8test.py`: YOLO model inference scripts for validation and testing.
- `show_results.py`: Prints summary statistics from saved JSON test results.
- `generate_full_gallery.py`: Builds an HTML gallery from visualization output images.
- `server.js`: Minimal Express backend example.
- Jupyter notebooks for model exploration and severity detection.

## Repository Structure

- `app.py` - Flask application for live video detection.
- `server.js` - Simple Node.js Express server.
- `test.py` - YOLO inference test runner.
- `v8test.py` - Extended YOLOv8 evaluation script.
- `show_results.py` - Summarizes existing test results.
- `generate_full_gallery.py` - Creates HTML gallery from visualization outputs.
- `templates/index.html` - Web dashboard template.
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

### Run the Flask dashboard

1. Update the video file path inside `app.py` if needed. The current path is set to:
   `E:\kjsce\Hackathons\Setu\cityRoad_potHoles-side.mp4`.
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

## Notes

- Model weight files such as `yoloV8_best.pt`, `best.pt`, `detection_model.pt`, and `severity_model.pth` are large and may not be suitable for pushing to GitHub.
- Update hard-coded paths in the scripts to match your local file structure.
- If you add dataset folders, keep them outside version control or use Git LFS for large files.

## Suggested Improvements

- Add a `data.yaml` dataset configuration file.
- Convert absolute paths to relative paths.
- Add a license file if you want to publish this project publicly.
