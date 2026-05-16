#!/usr/bin/env python3
"""
Visualize YOLOv8 Model Predictions
Draws bounding boxes on test images to show model detections
"""

import os
import cv2
import json
from pathlib import Path

# Configuration
TEST_IMAGES_DIR = "test/images"
TEST_LABELS_DIR = "test/labels"
RESULTS_JSON = "test_results/test_results.json"
OUTPUT_DIR = "test_results/visualized_images"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Class names and colors
CLASS_NAMES = ['crocodile crack', 'longitudinal crack', 'pothole']
CLASS_COLORS = {
    0: (0, 255, 0),      # Green for crocodile crack
    1: (0, 165, 255),    # Orange for longitudinal crack
    2: (0, 0, 255)       # Red for pothole
}

def load_results():
    """Load test results from JSON"""
    with open(RESULTS_JSON, 'r') as f:
        return json.load(f)

def parse_label_file(label_path):
    """Parse YOLO label file and convert to pixel coordinates"""
    bboxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bboxes.append({
                            'class_id': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
    return bboxes

def norm_to_pixel(bbox, img_w, img_h):
    """Convert normalized YOLO coordinates to pixel coordinates"""
    x_center = bbox['x_center'] * img_w
    y_center = bbox['y_center'] * img_h
    width = bbox['width'] * img_w
    height = bbox['height'] * img_h
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def visualize_predictions(image_path, predictions, ground_truth, output_path):
    """Draw predictions and ground truth on image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return False
    
    img_h, img_w = img.shape[:2]
    
    # Draw ground truth boxes in LIGHT colors
    for gt in ground_truth:
        class_id = gt['class_id']
        x1, y1, x2, y2 = norm_to_pixel(gt, img_w, img_h)
        
        # Light color for ground truth
        color = tuple(int(c * 0.5) for c in CLASS_COLORS.get(class_id, (255, 255, 255)))
        
        # Draw dashed rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
        cv2.putText(img, f"GT: {class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw prediction boxes in BRIGHT colors
    for pred in predictions:
        class_id = pred['class_id']
        conf = pred['confidence']
        x1, y1, x2, y2 = pred['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Draw solid rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
        label = f"{class_name}: {conf:.2f}"
        
        # Put text with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(output_path, img)
    return True

def main():
    """Visualize all test predictions"""
    print("=" * 80)
    print("Visualizing YOLOv8 Predictions")
    print("=" * 80)
    
    if not os.path.exists(RESULTS_JSON):
        print(f"Results file not found: {RESULTS_JSON}")
        print("Please run v8test.py first to generate results.")
        return
    
    results = load_results()
    results_list = results.get('results', [])
    
    print(f"\nFound {len(results_list)} test results")
    print(f"Saving visualized images to: {OUTPUT_DIR}\n")
    
    success_count = 0
    image_with_detections = []
    
    for idx, result in enumerate(results_list, 1):
        image_name = result['image']
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        label_path = os.path.join(TEST_LABELS_DIR, Path(image_name).stem + '.txt')
        
        # Load ground truth
        ground_truth = parse_label_file(label_path)
        
        # Get predictions
        predictions = result['detections']
        
        # Only visualize images with detections or ground truth
        if len(predictions) > 0 or len(ground_truth) > 0:
            output_path = os.path.join(OUTPUT_DIR, image_name)
            
            if visualize_predictions(image_path, predictions, ground_truth, output_path):
                success_count += 1
                image_with_detections.append({
                    'image': image_name,
                    'predictions': len(predictions),
                    'ground_truth': len(ground_truth),
                    'output_path': output_path
                })
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(results_list)} images...")
    
    print(f"\nSuccessfully visualized {success_count} images")
    print(f"Images with detections or ground truth: {len(image_with_detections)}")
    
    # Show summary of images with detections
    print("\n" + "-" * 80)
    print("Sample visualized images (first 10):")
    print("-" * 80)
    
    for i, item in enumerate(image_with_detections[:10], 1):
        print(f"{i}. {item['image']}")
        print(f"   Predictions: {item['predictions']}, Ground Truth: {item['ground_truth']}")
    
    print("\n" + "=" * 80)
    print(f"All visualized images saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
