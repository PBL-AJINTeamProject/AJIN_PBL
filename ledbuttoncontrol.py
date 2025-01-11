from gpiozero import Button, LED
from signal import pause

button = Button(3)  # 버튼 연결 (BCM GPIO 3번 핀)
led = LED(14)       # LED 연결 (BCM GPIO 17번 핀)

# LED 상태를 토글하는 함수
def toggle_led():
    if led.is_lit:  # LED가 켜져 있다면
        led.off()    # LED를 끔
        print("LED OFF")
    else:           # LED가 꺼져 있다면
        led.on()     # LED를 켬
        print("LED ON")

# 버튼을 누를 때마다 LED 상태를 토글
button.when_pressed = toggle_led

# 프로그램 유지
pause()
