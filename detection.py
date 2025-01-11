import gi
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from gpiozero import Button, LED
from threading import Thread, Lock
from time import sleep
from signal import pause
from detection_pipeline import GStreamerDetectionApp  # GStreamerDetectionApp을 detection_pipeline에서 가져옵니다.

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# -----------------------------------------------------------------------------------------------
# GPIO 설정
# -----------------------------------------------------------------------------------------------
led_pins = [14, 15, 18, 23]  # 27번 포트를 제외하고 LED 포트 설정
leds = [LED(pin) for pin in led_pins]  # LED 객체 생성 (27번 제외)
led_speaker = LED(21)  # 스피커용 포트 27번은 따로 설정
led_lock = Lock()  # LED 제어용 락 생성
led_active = False  # LED 상태 플래그
person_detected = False  # 사람이 감지되었는지 여부

# 버튼이 눌렸을 때 LED 상태를 제어
def button_pressed():
    global led_active, person_detected
    print("버튼이 눌렸습니다.")
    
    # 버튼을 누르면 모든 LED를 끄도록
    led_active = False
    for led in leds:
        led.off()
    led_speaker.off()  # 스피커도 끄기

# -----------------------------------------------------------------------------------------------
# LED 깜빡임 제어 함수 (별도 쓰레드에서 실행)
# -----------------------------------------------------------------------------------------------
def blink_leds(leds):
    global led_active
    with led_lock:  # LED 상태 보호
        while led_active:  # LED가 켜져 있으면 무한 대기
            for led in leds:
                led.on()  # LED 켜기
            sleep(0.5)  # 0.5초 대기
            for led in leds:
                led.off()  # LED 끄기
            sleep(0.5)  # 0.5초 대기

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# ----------------------------------------------------------------------------------------------- 
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42

    def new_function(self):
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    global person_detected, led_active
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # ROI 영역 정의 (프레임의 왼쪽 절반, 즉 x좌표 0부터 width//2까지)
    roi_points = np.array([(0, 0), (width // 2, 0), (width // 2, height), (0, height)], np.int32)

    # 프레임이 None이 아닌 경우에만 작업 수행
    if frame is not None:
        # 프레임 크기 출력
        height, width, _ = frame.shape
        print(f"Frame size: width={width}, height={height}")

        # ROI 좌표 출력
        print(f"ROI Points: {roi_points}")

        # ROI를 노란색으로 그리기
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    current_person_detected = False  # 사람이 탐지되었는지 여부
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Bounding box 좌표 추출 및 변환 (정규화된 좌표 → 픽셀 좌표)
        xmin = int(bbox.xmin() * width)
        ymin = int(bbox.ymin() * height)
        xmax = int(bbox.xmax() * width)
        ymax = int(bbox.ymax() * height)

        # 사람 감지된 객체가 ROI 영역에 포함되는지 확인
        if label == "Person":
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            in_roi = cv2.pointPolygonTest(roi_points, (center_x, center_y), False)
            
            # 디버깅 출력
            print(f"Person detected: {label}, Confidence: {confidence:.2f}")
            print(f"Bounding Box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            print(f"Center: ({center_x}, {center_y}), In ROI: {in_roi}")

            # ROI 영역에 포함된 경우만 LED 제어
            if in_roi >= 0:
                string_to_print += f"Detection: {label} {confidence:.2f}\n"
                detection_count += 1
                current_person_detected = True

    # LED 제어 (비동기) 
    if current_person_detected and not led_active:  # 사람이 ROI 영역에 있을 때 LED 울리기
        print("사람이 감지되었습니다! LED 켜기")
        led_active = True
        led_thread = Thread(target=blink_leds, args=(leds,))  # LED 깜빡이기 설정
        led_thread.start()  # 쓰레드 실행

    # 스피커는 대기 없이 바로 켜짐
    if current_person_detected:
        led_speaker.on()  # 스피커가 계속 울리게 설정

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    try:
        # 버튼을 눌렀을 때 LED를 끄도록 연결
        button = Button(20)  # GPIO 20번 핀으로 버튼 설정
        button.when_pressed = button_pressed  # 버튼 눌렸을 때 LED 끄기

        # Create an instance of the user app callback class
        user_data = user_app_callback_class()
        app = GStreamerDetectionApp(app_callback, user_data)
        app.run()
    finally:
        for led in leds:
            led.off()  # 프로그램 종료 시 모든 LED 끄기
        led_speaker.off()  # 프로그램 종료 시 스피커 끄기
