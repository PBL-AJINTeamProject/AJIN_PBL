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
led_pins = [2, 3, 4, 17, 27, 22, 10, 14, 15, 18, 23, 24, 25, 8]  # GPIO 핀 번호
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
        if label == "Person":
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
