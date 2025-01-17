from ultralytics import YOLO
import cv2
import os

# 학습된 모델 파일 경로
model_path = "./runs/detect/train_results/weights/best.pt"
input_folder = "./test_images"
output_folder = "./results"

# YOLO 모델 로드
model = YOLO(model_path)

# 입력 폴더에서 모든 이미지 파일 찾기
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 각 이미지에 대해 탐지 수행
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, f"output_{image_file}")  # 결과 이미지 저장 경로
    
    # 이미지 로드
    image = cv2.imread(image_path)

    # 이미지 감지 실행
    results = model.predict(source=image_path, save=False)  # 감지 수행 (결과 저장은 수동 처리)

    confidence_threshold = 0.5  # 최소 신뢰도 설정

    # 탐지 결과를 시각화하며 저장
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # 클래스 ID
            label = model.names[cls]  # 클래스 이름
            conf = box.conf[0]  # 신뢰도

            if conf >= confidence_threshold:  # 신뢰도가 임계값 이상인 경우
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표 (좌상단, 우하단)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 바운딩 박스

                label_text = f"{label} {conf:.2f}"  # 객체 이름 및 신뢰도
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Detected {label} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    # 결과 이미지 저장
    cv2.imwrite(output_path, image)

print("All images processed successfully.")