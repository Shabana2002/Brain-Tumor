from ultralytics import YOLO
import cv2
import os

# 1. UPDATE PATH to the NEW Classification results folder
# 1. UPDATED PATH based on your actual training logs
model_path = r'C:/Users/hp/Downloads/Brain_Tumor_Prediction_and_Segmentation/runs/classify/Tumor_Results/Brain_Classify_Final/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print("💡 Tip: Ensure your training has finished and 'best.pt' exists.")
else:
    # Load the classification model
    model = YOLO(model_path)

    # Use your test images folder
    test_folder = r'C:/Users/hp/Downloads/Brain_Tumor_Prediction_and_Segmentation/classification/Testing'

    # Get a list of all images in subfolders
    image_files = []
    for root, dirs, files in os.walk(test_folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    if len(image_files) == 0:
        print(f"⚠️ No images found in: {test_folder}")
    else:
        print(f"🔍 Found {len(image_files)} images. Starting Classification...")
        print("-" * 60)

        for full_path in image_files:
            image_name = os.path.basename(full_path)

            # Run prediction
            # imgsz=224 must match your training size
            results = model.predict(source=full_path, imgsz=224, save=False, verbose=False)

            for r in results:
                # Get the top class (the one with the highest probability)
                top_class_idx = r.probs.top1
                label = model.names[top_class_idx]
                conf = float(r.probs.top1conf) * 100

                # Color coding for the terminal output
                if label == "notumor":
                    icon = "⚪"
                    display_label = "Healthy (No Tumor)"
                else:
                    icon = "🔴"
                    display_label = label.capitalize()

                print(f"{icon} Image: {image_name:30} -> RESULT: {display_label:18} | Confidence: {conf:.2f}%")

                # Show the image with the prediction text
                annotated_frame = r.plot()
                cv2.imshow("AI Brain Classification (Press 'q' to Quit, any key for Next)", annotated_frame)

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit()

        cv2.destroyAllWindows()
        print("-" * 60)
        print("✅ Analysis Complete.")