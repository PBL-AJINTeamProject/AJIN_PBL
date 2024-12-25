https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst


from hailo_platform import HEF, Device, InputVStreams, OutputVStreams
import cv2
import numpy as np

# 1. HEF 파일 및 장치 초기화
hef_path = "object_detection_model.hef"
with Device() as device:
    hef = HEF(hef_path)
    configure_params = hef.create_configure_params()
    network_group = device.configure(hef, configure_params)

    # 2. 입력/출력 스트림 구성
    input_vstreams = InputVStreams(network_group)
    output_vstreams = OutputVStreams(network_group)

    # 3. 실시간 데이터 캡처 및 전처리
    cap = cv2.VideoCapture(0)  # 웹캠 연결
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 입력 데이터 전처리 (모델 크기와 일치)
        resized_frame = cv2.resize(frame, (300, 300))  # 모델에 맞는 입력 크기
        input_data = np.expand_dims(resized_frame, axis=0)

        # 4. 모델 실행 (입력 전송 및 결과 수신)
        input_vstreams[0].send(input_data)
        output_data = output_vstreams[0].receive()

        # 5. 출력 데이터 처리 및 시각화
        detections = parse_detections(output_data)  # 사용자 정의 함수로 데이터 해석
        for x, y, w, h, conf, class_id in detections:
            label = f"Class {class_id}: {conf:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
