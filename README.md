실행 코드

cd hailo-rpi5-examples
source setup_env.sh
python3 basic_pipelines/detection.py --hef-path resources/Detection_Person.hef --input /dev/video0 --labels-json resources/Detection_Person.json
