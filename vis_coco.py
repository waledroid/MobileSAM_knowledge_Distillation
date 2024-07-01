import os
import json
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def visualize_coco_annotations(coco_json_path, images_dir, output_dir):
    # Load COCO annotations
    coco = COCO(coco_json_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each image in the COCO dataset
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image {img_info['file_name']} not found in {images_dir}.")
            continue
        
        # Convert image to RGB (OpenCV loads images in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Draw annotations on the image
        for ann in anns:
            bbox = ann['box']
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            
            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put category name text
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the image with annotations
        output_img_path = os.path.join(output_dir, img_info['file_name'])
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(img_info['file_name'])
        plt.savefig(output_img_path)
        plt.close()

        print(f"Processed {img_info['file_name']}")

# Define paths
coco_json_path ='/home/wasoria-abdi/Desktop/ML_STUDY/nanosam/data/models/resnet185k/resnet185k_coco_results.json'
images_dir = '/home/wasoria-abdi/Desktop/ML_STUDY/data/wa5k/val'
output_dir = '/home/wasoria-abdi/Desktop/ML_STUDY/results_distillation_nano/resnet185k'

# Call the function to visualize annotations
visualize_coco_annotations(coco_json_path, images_dir, output_dir)
