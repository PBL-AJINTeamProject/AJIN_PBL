https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/


cd hailo-rpi5-examples

source setup_env.sh

python3 basic_pipelines/detection.py --hef-path resources/Detection_Person.hef --input /dev/video0 --labels-json resources/Detection_Person.json

https://nan-sso-gong.tistory.com/m/6




for detection in detections:
    label = detection.get_label()
    bbox = detection.get_bbox()  # 바운딩 박스 좌표 가져오기
    confidence = detection.get_confidence()
    
    # 바운딩 박스 좌표 고정 로직
    x_min, y_min, x_max, y_max = bbox
    if y_max > 480:
        y_max = 480  # y_max 값 고정
        bbox = (x_min, y_min, x_max, y_max)  # 수정된 바운딩 박스를 다시 저장
    
    if label == "Person":
        string_to_print += f"Detection: {label} {confidence:.2f}, BBox: {bbox}\n"
        detection_count += 1
        person_detected = True
