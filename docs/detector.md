# ProductDetector Documentation

The `ProductDetector` class is the core computer vision component of ShelfAI, responsible for detecting and classifying products on grocery shelves using YOLOv8 object detection.

## Overview

ProductDetector combines state-of-the-art YOLO object detection with intelligent preprocessing and product classification to provide accurate shelf analysis suitable for retail environments.

## Features

- **YOLOv8 Integration**: Supports all YOLOv8 model sizes (n, s, m, l, x)
- **Advanced Preprocessing**: CLAHE enhancement, sharpening, and noise reduction
- **Smart Product Classification**: Categorizes detected objects into food, beverages, household items
- **Device Optimization**: Automatic MPS/CUDA/CPU selection for optimal performance
- **Shelf Density Analysis**: Comprehensive metrics for inventory management
- **Visualization Support**: Bounding box visualization with confidence scores

## Class Reference

### Constructor

```python
ProductDetector(confidence=0.4, model_size='x')
```

**Parameters:**
- `confidence` (float): Detection confidence threshold (0.0-1.0). Default: 0.4
- `model_size` (str): YOLO model size ('n', 's', 'm', 'l', 'x'). Default: 'x'

**Model Size Comparison:**
| Size | Speed | Accuracy | Model File | Use Case |
|------|-------|----------|------------|----------|
| `n` | Fastest | Basic | `yolov8n.pt` | Real-time applications |
| `s` | Fast | Good | `yolov8s.pt` | Mobile/edge devices |
| `m` | Medium | Better | `yolov8m.pt` | Balanced performance |
| `l` | Slow | High | `yolov8l.pt` | High accuracy needs |
| `x` | Slowest | Highest | `yolov8x.pt` | Maximum accuracy |

### Methods

#### `detect_products(image_path: str) -> Dict`

Main detection method that processes a shelf image and returns comprehensive analysis.

**Parameters:**
- `image_path` (str): Path to the shelf image file

**Returns:**
Dictionary containing:
```python
{
    'total_detections': int,        # Total products detected
    'food_items': int,              # Number of food products
    'beverages': int,               # Number of beverage products  
    'household_items': int,         # Number of household products
    'other_products': int,          # Number of unclassified products
    'food_percentage': float,       # Percentage of food items
    'shelf_density': str,           # Density level (very_low to very_high)
    'detected_objects': List[Dict]  # Detailed detection data
}
```

**Detection Object Structure:**
```python
{
    'class': str,        # YOLO class name (e.g., 'bottle', 'apple')
    'confidence': float, # Detection confidence (0.0-1.0)
    'type': str         # Product category ('food', 'beverage', 'household', 'other')
}
```

#### `visualize(image_path: str, output_path: str = None) -> str`

Creates visualization with bounding boxes and saves to file.

**Parameters:**
- `image_path` (str): Path to the original image
- `output_path` (str, optional): Output path for visualization. Auto-generated if None.

**Returns:**
- `str`: Path to the saved visualization file

## Usage Examples

### Basic Detection

```python
from PlantAI import ProductDetector

# Initialize with default settings
detector = ProductDetector()

# Run detection
results = detector.detect_products('shelf_image.jpg')

print(f"Total products: {results['total_detections']}")
print(f"Food items: {results['food_items']}")
print(f"Shelf density: {results['shelf_density']}")
```

### High-Sensitivity Detection

```python
# Lower confidence for dense shelves
detector = ProductDetector(confidence=0.25, model_size='x')
results = detector.detect_products('crowded_shelf.jpg')

# Create visualization
detector.visualize('crowded_shelf.jpg', 'output_visualization.jpg')
```

### Performance Optimization

```python
# Fast detection for real-time applications
fast_detector = ProductDetector(confidence=0.5, model_size='n')

# High accuracy for detailed analysis
precise_detector = ProductDetector(confidence=0.3, model_size='x')
```

### Analyzing Detection Results

```python
results = detector.detect_products('shelf.jpg')

# Access detailed detections
for obj in results['detected_objects']:
    print(f"Found {obj['class']} ({obj['type']}) with {obj['confidence']:.2f} confidence")

# Check shelf density
if results['shelf_density'] in ['very_high', 'high']:
    print("Shelf is well-stocked")
elif results['shelf_density'] in ['low', 'very_low']:
    print("Shelf needs restocking")
```

## Technical Details

### Image Preprocessing Pipeline

1. **CLAHE Enhancement**: Improves contrast using Contrast Limited Adaptive Histogram Equalization
2. **Sharpening**: Applies convolution kernel for better edge definition
3. **Noise Reduction**: Uses bilateral filtering to reduce noise while preserving edges

```python
def _preprocess_image(self, image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    
    # CLAHE enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Bilateral filtering
    denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    return denoised
```

### YOLO Detection Parameters

The detector uses optimized YOLO parameters for grocery shelf analysis:

- **Confidence**: Configurable threshold for detection filtering
- **IoU**: 0.4 for better handling of overlapping products
- **Image Size**: 1280 pixels for high-resolution detection
- **Augmentation**: Test-time augmentation for improved robustness

### Product Classification System

Products are classified into four categories based on YOLO class names:

#### Food Items
- Fresh produce: banana, apple, orange, carrot, broccoli
- Prepared foods: sandwich, pizza, hot dog, cake, donut
- Packaged foods: bread, cheese, egg, meat, chicken, fish
- Pantry items: pasta, rice, cereal, snacks, cookies, chocolate

#### Beverages
- Containers: bottle, wine glass, cup, bowl, can
- Drink types: soda, water, juice, beer, wine, coffee, tea

#### Household Items
- Personal care: soap, shampoo, toothbrush, toothpaste
- Cleaning: detergent, paper towel, toilet paper, cleaning products

#### Other Products
- Any detected object not matching the above categories

### Shelf Density Levels

Density classification based on total product count:

| Level | Product Count | Description |
|-------|---------------|-------------|
| `very_low` | 0-5 | Nearly empty shelf |
| `low` | 6-10 | Sparse stocking |
| `moderate` | 11-15 | Adequate stocking |
| `high` | 16-20 | Well-stocked shelf |
| `very_high` | 21+ | Densely packed shelf |

## Performance Considerations

### Hardware Acceleration

The detector automatically selects the best available device:

1. **MPS** (Apple Silicon): Optimal for M1/M2 Macs
2. **CUDA** (NVIDIA GPUs): Best performance on compatible systems
3. **CPU**: Fallback for universal compatibility

### Memory Usage

Model memory requirements:
- YOLOv8n: ~6 MB
- YOLOv8s: ~22 MB  
- YOLOv8m: ~52 MB
- YOLOv8l: ~88 MB
- YOLOv8x: ~136 MB

### Processing Time

Typical processing times (varies by hardware):
- YOLOv8n: 50-100ms per image
- YOLOv8x: 200-500ms per image

## Troubleshooting

### Low Detection Rates

1. **Lower confidence threshold**: Try 0.2-0.3 for dense shelves
2. **Use larger model**: Switch to YOLOv8l or YOLOv8x
3. **Check image quality**: Ensure good lighting and resolution

### Memory Issues

1. **Use smaller model**: Switch to YOLOv8n or YOLOv8s
2. **Reduce image size**: Preprocess images to lower resolution
3. **Close other applications**: Free up system memory

### Performance Issues

1. **Use GPU acceleration**: Ensure CUDA/MPS is available
2. **Optimize confidence**: Higher thresholds reduce processing time
3. **Batch processing**: Process multiple images together

## Integration Examples

### With ShelfAnalyzer

```python
from PlantAI import ProductDetector, ShelfAnalyzer

detector = ProductDetector(confidence=0.3, model_size='x')
analyzer = ShelfAnalyzer(provider='claude')

# Detection + Analysis pipeline
detection_results = detector.detect_products('shelf.jpg')
ai_insights = analyzer.analyze_shelf('shelf.jpg')

print(f"Detected {detection_results['total_detections']} products")
print(f"AI Analysis: {ai_insights}")
```

### Custom Classification

```python
class CustomProductDetector(ProductDetector):
    def _classify_product(self, class_name: str) -> str:
        # Custom classification logic
        if 'organic' in class_name.lower():
            return 'organic'
        return super()._classify_product(class_name)
```

### Batch Processing

```python
import os
from pathlib import Path

detector = ProductDetector()
results = {}

# Process all images in directory
for image_file in Path('shelf_images').glob('*.jpg'):
    results[image_file.name] = detector.detect_products(str(image_file))
    
# Aggregate results
total_products = sum(r['total_detections'] for r in results.values())
print(f"Total products across all shelves: {total_products}")
```