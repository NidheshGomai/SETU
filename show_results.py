import json

with open('test_results/test_results.json') as f:
    data = json.load(f)

print("\n" + "="*80)
print("DETAILED TEST RESULTS")
print("="*80)
print(f"\nTest Date: {data['test_time']}")
print(f"Total Images Tested: {data['total_images']}")
print(f"Images with Detections: {data['images_with_detections']}")
print(f"Total Ground Truth Objects: {data['total_ground_truth']}")
print(f"Total Predictions: {data['total_predictions']}")

if data['total_ground_truth'] > 0:
    detection_rate = (data['total_predictions'] / data['total_ground_truth']) * 100
    print(f"Overall Detection Rate: {detection_rate:.2f}%")

class_names = ['crocodile crack', 'longitudinal crack', 'pothole']

print("\n" + "-"*80)
print("CLASS-WISE STATISTICS")
print("-"*80)
for class_id, stats in sorted(data['class_stats'].items()):
    class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {class_id}"
    print(f"\n{class_name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

# Show sample results from first few images
print("\n" + "-"*80)
print("SAMPLE PREDICTIONS (First 5 images with detections)")
print("-"*80)

count = 0
for result in data['results'][:30]:
    if result['prediction_count'] > 0 and count < 5:
        print(f"\n{result['image']}:")
        print(f"  Ground Truth Objects: {result['ground_truth_count']}")
        print(f"  Predictions: {result['prediction_count']}")
        for i, det in enumerate(result['detections'][:3], 1):
            class_id = det['class_id']
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            conf = det['confidence']
            print(f"    {i}. {class_name} (confidence: {conf:.3f})")
        count += 1

print("\n" + "="*80)
