import torch

# Monkeypatch torch.load to disable weights_only for older model files
_original_load = torch.load
def patched_load(f, *args, **kwargs):
    # Force weights_only=False for models that aren't compatible with PyTorch 2.6+
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import cv2
import glob
import os
from pathlib import Path

# Load the model
print("Loading best.pt model...")
model = YOLO("detection_model.pt")

# Define paths
test_images_dir = r"E:\kjsce\Hackathons\Setu\test\images"
output_dir = r"E:\kjsce\Hackathons\Setu\test_results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = sorted(glob.glob(f"{test_images_dir}/*.jpg")) + sorted(glob.glob(f"{test_images_dir}/*.png"))

if not image_files:
    print(f"ERROR: No images found in {test_images_dir}")
    exit(1)

print(f"Found {len(image_files)} images to test\n")

total_detections = 0

# Process each image
for idx, image_path in enumerate(image_files[:10], 1):  # Test first 10 images
    print(f"[{idx}/{min(10, len(image_files))}] Processing: {Path(image_path).name}")
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ERROR: Could not read image\n")
        continue
    
    try:
        # Run inference
        results = model(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            detections = len(result.boxes) if result.boxes is not None else 0
            total_detections += detections
            
            print(f"  Detections: {detections}")
            
            # Save results with annotations
            annotated_frame = result.plot()
            output_path = os.path.join(output_dir, f"result_{Path(image_path).stem}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"  Saved: {output_path}\n")
        else:
            print(f"  No results from model\n")
    except Exception as e:
        print(f"  ERROR during inference: {type(e).__name__}: {str(e)}\n")
        continue

print(f"\n========== TEST SUMMARY ==========")
print(f"Total images tested: {min(10, len(image_files))}")
print(f"Total detections: {total_detections}")
print(f"Avg detections per image: {total_detections / min(10, len(image_files)):.2f}")
print(f"Results saved to: {output_dir}")
print("==================================")

