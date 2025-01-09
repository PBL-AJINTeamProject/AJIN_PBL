https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/


cd hailo-rpi5-examples

source setup_env.sh

python3 basic_pipelines/detection.py --hef-path resources/Detection_Person.hef --input /dev/video0 --labels-json resources/Detection_Person.json

import board
import neopixel

# 설정
LED_COUNT = 12          # 네오픽셀 LED 개수
PIN = board.D18         # 데이터 핀 (GPIO18 사용)
BRIGHTNESS = 0.5        # 밝기 (0.0 ~ 1.0)

# 네오픽셀 초기화
pixels = neopixel.NeoPixel(PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False)

# 색상 정의 (RGB 값)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# LED 제어
def set_color(color):
    pixels.fill(color)  # 모든 LED를 해당 색으로 설정
    pixels.show()       # LED에 반영

# 실행
try:
    while True:
        set_color(RED)
        time.sleep(1)
        set_color(GREEN)
        time.sleep(1)
        set_color(BLUE)
        time.sleep(1)

except KeyboardInterrupt:
    # 종료 시 LED 끄기
    set_color((0, 0, 0))
