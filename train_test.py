import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_path = r'C:\Users\hp\Downloads\Brain_Tumor_Prediction_and_Segmentation'
seg_path = os.path.join(base_path, 'Segmentation')
export_path = os.path.join(base_path, 'Segmentation_Dataset_YOLO')

# Create folder structure
for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    os.makedirs(os.path.join(export_path, folder), exist_ok=True)

folders = ['Glioma', 'Meningioma', 'Pituitary tumor']

for folder in folders:
    current_dir = os.path.join(seg_path, folder)
    # Get only the original images (not the masks)
    images = [f for f in os.listdir(current_dir) if not f.endswith('_mask.png') and f.endswith(('.png', '.jpg'))]

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)


    def move_files(files, split):
        for f in files:
            # Move Image
            shutil.copy(os.path.join(current_dir, f), os.path.join(export_path, split, 'images', f))
            # Move Corresponding Label (.txt)
            label_name = os.path.splitext(f)[0] + '.txt'
            if os.path.exists(os.path.join(current_dir, label_name)):
                shutil.copy(os.path.join(current_dir, label_name),
                            os.path.join(export_path, split, 'labels', label_name))


    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')

print(f"✅ Dataset organized at: {export_path}")