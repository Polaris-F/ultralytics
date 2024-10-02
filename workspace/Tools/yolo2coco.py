
import os
import json
from pathlib import Path
from PIL import Image

def yolo_to_coco(yolo_dir, images_dir, output_json):
    # COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create a mapping of category index to category id
    category_mapping = {}
    category_id = 1

    # Read all the yolo label files
    for yolo_file in Path(yolo_dir).glob('*.txt'):
        image_name = yolo_file.stem + '.jpg'  # Assuming images are in jpg format
        image_path = Path(images_dir) / image_name
        if not image_path.is_file():
            continue  # Skip if the image does not exist

        # Get image size
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info to COCO format
        coco_format['images'].append({
            "id": len(coco_format['images']) + 1,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Read the YOLO file to get annotations
        with open(yolo_file, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                category_id = int(class_id) + 1  # YOLO classes start from 0

                # Check if the category is already in the mapping
                if category_id not in category_mapping:
                    category_mapping[category_id] = {
                        "id": category_id,
                        "name": f'class_{category_id}'  # Customize category names if needed
                    }
                    coco_format['categories'].append(category_mapping[category_id])

                # Convert YOLO format (x_center, y_center, width, height) to COCO format (x, y, width, height)
                coco_format['annotations'].append({
                    "id": len(coco_format['annotations']) + 1,
                    "image_id": len(coco_format['images']),
                    "category_id": category_id,
                    "bbox": [
                        (x_center - width / 2) * width,  # x
                        (y_center - height / 2) * height,  # y
                        width * width,  # width
                        height * height  # height
                    ],
                    "area": (width * width) * (height * height),
                    "iscrowd": 0
                })

    # Save COCO format to JSON file
    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# Example usage
yolo_directory = '/path/to/yolo/labels'
images_directory = '/path/to/images'
output_coco_file = '/path/to/output/coco_annotations.json'

yolo_to_coco(yolo_directory, images_directory, output_coco_file)
