import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import Dict


class ProductDetector:
    def __init__(self, confidence=0.4, model_size='x'):
        self.confidence = confidence

        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        if self.device != 'cpu':
            self.model.to(self.device)

    def detect_products(self, image_path: str) -> Dict:
        processed_image = self._preprocess_image(image_path)

        results = self.model(
            processed_image, 
            conf=self.confidence, 
            iou=0.4,
            device=self.device, 
            verbose=False,
            imgsz=1280,
            augment=True
        )

        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'type': self._classify_product(class_name)
                    })

        food_items = [d for d in detections if d['type'] == 'food']
        beverages = [d for d in detections if d['type'] == 'beverage']
        household_items = [d for d in detections if d['type'] == 'household']
        other_products = [d for d in detections if d['type'] == 'other']

        total_products = len(food_items) + len(beverages) + len(household_items) + len(other_products)
        food_percentage = (len(food_items) / total_products * 100) if total_products > 0 else 0

        return {
            'total_detections': len(detections),
            'food_items': len(food_items),
            'beverages': len(beverages),
            'household_items': len(household_items),
            'other_products': len(other_products),
            'food_percentage': round(food_percentage, 1),
            'shelf_density': self._get_shelf_density(total_products),
            'detected_objects': detections
        }

    def _classify_product(self, class_name: str) -> str:
        class_lower = class_name.lower()

        food_items = {
            'banana', 'apple', 'orange', 'sandwich', 'pizza', 'hot dog', 'cake', 
            'carrot', 'broccoli', 'donut', 'bread', 'cheese', 'egg', 'meat', 
            'chicken', 'fish', 'pasta', 'rice', 'cereal', 'milk', 'yogurt',
            'fruit', 'vegetable', 'snack', 'cookie', 'chocolate', 'candy'
        }
        
        beverages = {
            'bottle', 'wine glass', 'cup', 'bowl', 'can', 'soda', 'water', 
            'juice', 'beer', 'wine', 'coffee', 'tea', 'energy drink'
        }
        
        household_items = {
            'soap', 'shampoo', 'toothbrush', 'toothpaste', 'detergent', 
            'paper towel', 'toilet paper', 'cleaning product'
        }

        if any(food in class_lower for food in food_items):
            return 'food'
        elif any(bev in class_lower for bev in beverages):
            return 'beverage'
        elif any(item in class_lower for item in household_items):
            return 'household'
        else:
            return 'other'

    def _get_shelf_density(self, product_count: int) -> str:
        if product_count > 20:
            return 'very_high'
        elif product_count > 15:
            return 'high'
        elif product_count > 10:
            return 'moderate'
        elif product_count > 5:
            return 'low'
        else:
            return 'very_low'

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        return denoised

    def visualize(self, image_path: str, output_path: str = None) -> str:
        results = self.model(image_path, conf=self.confidence, device=self.device, verbose=False)

        annotated = results[0].plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2
        )

        if output_path:
            cv2.imwrite(output_path, annotated)
            return output_path
        else:
            output_path = image_path.replace('.jpg', '_products_detected.jpg')
            cv2.imwrite(output_path, annotated)
            return output_path