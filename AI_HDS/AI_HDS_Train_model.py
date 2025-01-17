from ultralytics import YOLO
import os

# YAML 파일
yaml_file = '/ultralytics/AI_HDS/datasets/AJIN_Data_Final/data.yaml'

# 초기 가중치 모델
model_name = "yolov8m.pt"

# 고정된 결과 폴더 이름
result_folder = "train_results_final"  # 원하는 폴더 이름을 지정

# YOLO 모델 로드 (이전 학습 결과 사용)
model = YOLO(model_name)

# 모델 학습 실행
model.train(
    data=yaml_file,
    epochs=600,
    imgsz=640,
    batch=32,
    augment=True,
    mosaic=True,
    mixup=True,  # MixUp 활성화
    lr0=0.001,  # 초기 학습률
    lrf=0.05,   # 학습률 감소
    optimizer='AdamW',  # 옵티마이저 설정
    project='./runs/detect',  # 결과 저장 경로
    name=result_folder,  # 고정된 이름으로 저장
    exist_ok=True
)


print(f"Training completed. Best model saved at: {model_name}\n")

results = model.val()
print(results)
