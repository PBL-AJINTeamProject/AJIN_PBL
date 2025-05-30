실행 코드

cd hailo-rpi5-examples

source setup_env.sh

python3 basic_pipelines/detection.py --hef-path resources/Detection_Person.hef --input /dev/video0 --labels-json resources/Detection_Person.json


sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps


from picamera2 import Picamera2
from time import sleep

picam2 = Picamera2()
picam2.start()
sleep(5)  # 5초간 실행
picam2.stop()
