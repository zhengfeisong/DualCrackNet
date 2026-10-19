import os
import cv2
import numpy as np
from pathlib import Path

def get_union_bbox(mask):
    """
    Return the smallest axis-aligned bounding box enclosing all foreground pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])
    return (x_min, y_min, x_max - x_min, y_max - y_min)  # x, y, w, h

def process_folder(data_folder='/root/RCD900/val/label', output_folder='/root/RCD900/val/box'):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(data_folder, filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            bbox = get_union_bbox(binary)
            if bbox:
                x, y, w, h = bbox
            else:
                x, y, w, h = 0, 0, 0, 0  # Default bounding box for a mask with no foreground pixels.
            name = Path(filename).stem
            output_path = os.path.join(output_folder, f"{name}.txt")
            with open(output_path, 'w') as f:
                f.write(f"{x} {y} {w} {h}\n")
            print(f"[✓] BBox saved: {output_path}")

if __name__ == "__main__":
    process_folder()
