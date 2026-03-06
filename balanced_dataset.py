import os
import shutil
import random

# 1. SET YOUR PATHS
# Update this to your exact Training folder path
source_path = r'C:/Users/hp/Downloads/Brain_Tumor_Prediction_and_Segmentation/classification/Training'
dest_path = r'C:/Users/hp/Downloads/Brain_Tumor_Prediction_and_Segmentation/Sorted_Dataset'

# Note: I included 'notumor' because a real doctor needs to know when it's healthy!
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']

for cat in categories:
    os.makedirs(os.path.join(dest_path, cat), exist_ok=True)

    # Get all images
    full_src = os.path.join(source_path, cat)
    images = [f for f in os.listdir(full_src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Take 2000 images (or all if less than 2000)
    random.shuffle(images)
    selected = images[:2000]

    print(f"Moving {len(selected)} images for {cat}...")
    for img in selected:
        shutil.copy(os.path.join(full_src, img), os.path.join(dest_path, cat, img))

print("✅ Balanced Dataset Ready!")