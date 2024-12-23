from ultralytics import YOLO
model = YOLO('/home/kkymin/Downloads/best.pt') # 이미 학습된 YOLOv8 모델
model.export(format='onnx', imgsz=640, name='/home/kkymin/Downloads/best.onnx') # 입력 이미지 크기 설정 (예: 640x640)
