from ultralytics import YOLO

# YOLOv8s 모델 로드
model = YOLO('yolov8s.pt')  # 이미 학습된 YOLOv8 모델

# ONNX 형식으로 변환
model.export(format="onnx", imgsz=640)  # 입력 이미지 크기 설정 (예: 640x640)

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --config python3
