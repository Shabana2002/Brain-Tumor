import cv2
import os


def mask_to_yolo(mask_path, output_txt_path, class_id):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = mask.shape
    with open(output_txt_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < 10: continue
            points = []
            for point in contour:
                x = point[0][0] / width
                y = point[0][1] / height
                points.append(f"{x:.6f} {y:.6f}")  # Precision matters for medical scans

            if points:
                f.write(f"{class_id} " + " ".join(points) + "\n")


# --- PATH CONFIGURATION ---
folders = {
    r'C:\Users\hp\Downloads\Brain_Tumor_Prediction_and_Segmentation\Segmentation\Glioma': 0,
    r'C:\Users\hp\Downloads\Brain_Tumor_Prediction_and_Segmentation\Segmentation\Meningioma': 1,
    r'C:\Users\hp\Downloads\Brain_Tumor_Prediction_and_Segmentation\Segmentation\Pituitary tumor': 2
}

print("🚀 Starting conversion of Masks to YOLO labels...")

for folder_path, class_id in folders.items():
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue

    count = 0
    for file in os.listdir(folder_path):
        # Identify the mask files
        if file.endswith('_mask.png'):
            mask_full_path = os.path.join(folder_path, file)

            # Create the TXT name by removing '_mask.png'
            # Example: enh_989_mask.png -> enh_989.txt
            txt_name = file.replace('_mask.png', '.txt')
            txt_full_path = os.path.join(folder_path, txt_name)

            mask_to_yolo(mask_full_path, txt_full_path, class_id)
            count += 1

    print(f"✅ Processed {count} masks in {os.path.basename(folder_path)}")

print("\n🎉 All folders processed! You now have .txt files next to your images.")