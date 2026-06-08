#!/usr/bin/env python3
"""
YOLOv8 Pothole Detection Model Testing Script
Tests the trained yoloV8_best.pt model on the test dataset
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

# Patch torch.load to handle PyTorch 2.6+ weights_only issue
_original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that disables weights_only for ultralytics models"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    try:
        return _original_torch_load(f, *args, **kwargs)
    except Exception:
        # Try again with weights_only=False explicitly
        kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# Now import YOLO after patching torch.load
from ultralytics import YOLO

# Configuration
MODEL_PATH = "detection_model.pt"
TEST_IMAGES_DIR = "test/images"
TEST_LABELS_DIR = "test/labels"
OUTPUT_DIR = "test_results"
DATA_YAML = "data.yaml"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_model():
    """Load the YOLOv8 model"""
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully. Device: {model.device}")
    return model

def get_test_images():
    """Get list of test images"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    if os.path.exists(TEST_IMAGES_DIR):
        for file in os.listdir(TEST_IMAGES_DIR):
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(TEST_IMAGES_DIR, file))
    
    print(f"Found {len(images)} test images")
    return sorted(images)

def parse_label_file(label_path):
    """
    Parse YOLO label file format
    Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    """
    bboxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
    
    return bboxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bboxes"""
    # Convert normalized coords to pixel coords
    def norm_to_pixel(bbox, img_w, img_h):
        x_center, y_center, width, height = bbox
        x_center, y_center, width, height = (
            x_center * img_w, y_center * img_h, width * img_w, height * img_h
        )
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if xi2 < xi1 or yi2 < yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def run_inference(model, image_path):
    """Run inference on a single image"""
    results = model(image_path, conf=0.25, verbose=False)
    return results[0]

def test_model():
    """Main testing function"""
    print("=" * 80)
    print("YOLOv8 Pothole Detection Model Testing")
    print("=" * 80)
    
    # Load model
    model = load_model()
    
    # Get test images
    test_images = get_test_images()
    
    if not test_images:
        print("No test images found!")
        return
    
    # Statistics
    stats = {
        'total_images': len(test_images),
        'images_with_detections': 0,
        'total_ground_truth': 0,
        'total_predictions': 0,
        'class_stats': defaultdict(lambda: {'detected': 0, 'missed': 0, 'tp': 0, 'fp': 0}),
        'test_time': datetime.now().isoformat(),
    }
    
    results_data = []
    
    print(f"\nTesting on {len(test_images)} images...")
    print("-" * 80)
    
    for idx, image_path in enumerate(test_images, 1):
        filename = os.path.basename(image_path)
        label_path = os.path.join(TEST_LABELS_DIR, Path(filename).stem + '.txt')
        
        # Get ground truth
        ground_truth = parse_label_file(label_path)
        stats['total_ground_truth'] += len(ground_truth)
        
        # Run inference
        result = run_inference(model, image_path)
        detections = result.boxes.data.cpu().numpy() if len(result.boxes) > 0 else []
        stats['total_predictions'] += len(detections)
        
        if len(detections) > 0:
            stats['images_with_detections'] += 1
        
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = 640, 640
        
        # Process detections
        image_result = {
            'image': filename,
            'ground_truth_count': len(ground_truth),
            'prediction_count': len(detections),
            'detections': []
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            conf = detection[4]
            class_id = int(detection[5])
            
            image_result['detections'].append({
                'class_id': class_id,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
            
            # Count detection by class
            stats['class_stats'][class_id]['detected'] += 1
        
        # Count misses by class
        for gt in ground_truth:
            class_id = gt['class_id']
            stats['class_stats'][class_id]['missed'] += 1
        
        results_data.append(image_result)
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(test_images)} images...")
    
    print(f"Processed {len(test_images)}/{len(test_images)} images.")
    print("-" * 80)
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    class_names = ['crocodile crack', 'longitudinal crack', 'pothole']
    
    print(f"\nTotal Images Tested: {stats['total_images']}")
    print(f"Images with Detections: {stats['images_with_detections']}")
    print(f"Total Ground Truth Objects: {stats['total_ground_truth']}")
    print(f"Total Predictions: {stats['total_predictions']}")
    
    print("\nPer-Class Statistics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Detected':<12} {'Missed':<12}")
    print("-" * 80)
    
    for class_id in sorted(stats['class_stats'].keys()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        detected = stats['class_stats'][class_id]['detected']
        missed = stats['class_stats'][class_id]['missed']
        print(f"{class_name:<20} {detected:<12} {missed:<12}")
    
    # Calculate and display detection rate
    if stats['total_ground_truth'] > 0:
        detection_rate = (stats['total_predictions'] / stats['total_ground_truth']) * 100
        print(f"\nOverall Detection Rate: {detection_rate:.2f}%")
    
    # Save detailed results
    output_json = os.path.join(OUTPUT_DIR, 'test_results.json')
    stats['results'] = results_data
    
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDetailed results saved to {output_json}")
    
    # Save summary to text file
    output_txt = os.path.join(OUTPUT_DIR, 'test_summary.txt')
    with open(output_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv8 Pothole Detection Model Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Time: {stats['test_time']}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test Dataset: {TEST_IMAGES_DIR}\n\n")
        f.write(f"Total Images Tested: {stats['total_images']}\n")
        f.write(f"Images with Detections: {stats['images_with_detections']}\n")
        f.write(f"Total Ground Truth Objects: {stats['total_ground_truth']}\n")
        f.write(f"Total Predictions: {stats['total_predictions']}\n")
        f.write(f"Detection Rate: {detection_rate:.2f}%\n\n")
        f.write("Per-Class Statistics:\n")
        f.write("-" * 80 + "\n")
        for class_id in sorted(stats['class_stats'].keys()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            detected = stats['class_stats'][class_id]['detected']
            missed = stats['class_stats'][class_id]['missed']
            f.write(f"{class_name}: {detected} detected, {missed} missed\n")
    
    print(f"Summary saved to {output_txt}")
    print("=" * 80)

def run_validation():
    """Run YOLOv8 built-in validation if data.yaml exists"""
    print("\n" + "=" * 80)
    print("Running YOLOv8 Validation Metrics")
    print("=" * 80)
    
    if not os.path.exists(DATA_YAML):
        print(f"Warning: {DATA_YAML} not found. Skipping validation metrics.")
        return
    
    try:
        model = YOLO(MODEL_PATH)
        print("Running validation...")
        results = model.val(data=DATA_YAML, split='test', device=0)
        
        # Save validation results
        if hasattr(results, 'save_dir'):
            print(f"Validation results saved to: {results.save_dir}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    try:
        test_model()
        run_validation()
        print("\nTesting completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
