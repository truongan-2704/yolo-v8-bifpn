import os
from ultralytics import YOLO

def train_yolo_with_bifpn():
    # 1. Đường dẫn đến file cấu hình YOLOv8 với BiFPN
    config_path = "ultralytics/models/v8/yolov8n.yaml"

    # 2. Đường dẫn đến file dữ liệu (tùy chỉnh dataset của bạn tại đây)
    data_path = "coco_dataset/data.yaml"

    # 3. Khởi tạo mô hình YOLO (không có pre-trained weights)
    model = YOLO("ultralytics/models/v8/yolov8n.yaml")

    # 4. Train mô hình với cấu hình đã tạo
    model.train(
        data=data_path,          # Dataset YAML file
        epochs=2,               # Số epoch
        batch=8,                # Batch size
        imgsz=640,               # Kích thước ảnh
        device='cpu'                 # GPU ID (đặt -1 nếu dùng CPU)
    )
    print("Model initialized successfully!")

train_yolo_with_bifpn()