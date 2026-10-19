import os
import cv2
from pathlib import Path

def visualize_bboxes(data_folder='RCAR/Dataset/train/label', bbox_folder='RCAR/Dataset/train/box', output_folder='RCAR/Dataset/train/box_vis'):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        name = Path(filename).stem
        image_path = os.path.join(data_folder, filename)
        bbox_path = os.path.join(bbox_folder, f"{name}.txt")

        if not os.path.exists(bbox_path):
            print(f"[ ] Skipped: {filename} (no bbox)")
            continue

        # Load image and bbox
        image = cv2.imread(image_path)
        if image is None:
            continue
        with open(bbox_path, 'r') as f:
            line = f.readline()
            x, y, w, h = map(int, line.strip().split())

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Optional: write label text
        cv2.putText(image, "bbox", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save image
        output_path = os.path.join(output_folder, f"{name}_bbox.png")
        cv2.imwrite(output_path, image)
        print(f"[✓] Saved visualization: {output_path}")

if __name__ == "__main__":
    visualize_bboxes()
