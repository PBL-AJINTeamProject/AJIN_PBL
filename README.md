https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst


import hailo_platform

# Hailo 장치 연결
try:
    device = hailo_platform.Device()  # Hailo 장치 초기화
    print("Hailo device connected successfully!")
except Exception as e:
    print(f"Failed to connect to Hailo device: {e}")
    exit(1)

# HEF 파일 경로 설정
hef_path = "path/to/your_model.hef"  # HEF 파일 경로 입력

# HEF 파일 로드
try:
    hef = hailo_platform.Hef(hef_path)
    print(f"HEF file '{hef_path}' loaded successfully!")
except Exception as e:
    print(f"Failed to load HEF file: {e}")
    exit(1)

# 모델 구성
try:
    network_group = device.configure(hef)
    print("Network group configured successfully!")
except Exception as e:
    print(f"Failed to configure the network group: {e}")
    exit(1)

# 입력 크기 확인
input_shapes = network_group.get_input_vstream_shapes()
print(f"Input shapes: {input_shapes}")

# 출력 크기 확인
output_shapes = network_group.get_output_vstream_shapes()
print(f"Output shapes: {output_shapes}")

# 가상 스트림 설정
vstream_manager = hailo_platform.VStreamsManager(network_group)

# 추론 테스트
import numpy as np

# 입력 데이터 생성 (랜덤 값)
input_data = {
    name: np.random.rand(*shape).astype(np.float32) 
    for name, shape in input_shapes.items()
}

# 추론 실행
try:
    output_data = vstream_manager.run(input_data)
    print("Inference completed successfully!")
    print(f"Output data: {output_data}")
except Exception as e:
    print(f"Failed to execute inference: {e}")
