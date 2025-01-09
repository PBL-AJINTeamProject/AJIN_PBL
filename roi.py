import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp
from gpiozero import LED
from threading import Thread, Lock
from time import sleep

# -----------------------------------------------------------------------------------------------
# GPIO 설정
# -----------------------------------------------------------------------------------------------
led_pins = [2, 3, 17, 26, 27]  # GPIO 핀 번호
leds = [LED(pin) for pin in led_pins]  # LED 객체 생성
led_lock = Lock()  # LED 제어용 락 생성
led_active = False  # LED 상태 플래그

# -----------------------------------------------------------------------------------------------
# LED 제어 함수 (별도 쓰레드에서 실행)
# -----------------------------------------------------------------------------------------------
def control_leds(leds, duration):
    global led_active
    with led_lock:  # LED 상태 보호
        if led_active:  # LED가 이미 켜져 있는 경우 실행하지 않음
            return
        led_active = True  # LED 상태를 "켜짐"으로 설정
        print("LED 켜짐")
        for led in leds:
            led.on()  # 모든 LED 켜기
        sleep(duration)  # LED를 지정된 시간 동안 유지
        for led in leds:
            led.off()  # 모든 LED 끄기
        print("LED 꺼짐")
        led_active = False  # LED 상태를 "꺼짐"으로 설정

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

    # ROI 영역을 정의합니다
    roi_points = np.array([(50, 50), (400, 50), (400, 300), (50, 300)], np.int32)

    # 프레임이 None이 아닌 경우에만 작업 수행
    if frame is not None:
        # 프레임 크기 출력
        height, width, _ = frame.shape
        print(f"Frame size: width={width}, height={height}")

        # ROI 좌표 출력
        print(f"ROI Points: {roi_points}")

        # ROI를 노란색 두꺼운 선으로 그리기
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=3)  # 두께 3

        # ROI 내부를 반투명하게 채우기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], color=(0, 255, 255))  # 노란색 채우기
        alpha = 0.3  # 투명도 설정 (0.0 ~ 1.0, 낮을수록 투명)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    person_detected = False  # 사람이 탐지되었는지 여부
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
                person_detected = True

    # LED 제어 (비동기)
    if person_detected and not led_active:  # LED가 꺼져 있는 경우에만 실행
        print("사람이 감지되었습니다! LED 켜기")
        led_thread = Thread(target=control_leds, args=(leds, 5))  # LED를 5초 동안 켜기
        led_thread.start()  # 쓰레드 실행

    if user_data.use_frame:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    try:
        # Create an instance of the user app callback class
        user_data = user_app_callback_class()
        app = GStreamerDetectionApp(app_callback, user_data)
        app.run()
    finally:
        for led in leds:
            led.off()  # 프로그램 종료 시 모든 LED 끄기
