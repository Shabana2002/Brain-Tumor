from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n-cls.pt')

    model.train(
        data=r'C:/Users/hp/Downloads/Brain_Tumor_Prediction_and_Segmentation/Sorted_Dataset',
        epochs=50,         # Increased for better accuracy
        imgsz=224,
        batch=16,
        patience=10,       # Stops early if no improvement for 10 epochs
        name='Brain_Classify_Final',
        project='Tumor_Results',
        optimizer='AdamW', # AdamW is excellent for classification stability
        lr0=0.001          # Standard starting learning rate
    )