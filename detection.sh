
#!/bin/bash
cd /home/kkymin/hailo-rpi5-examples  # 작업 디렉토리 이동
source setup_env.sh             # 환경 설정
python3 basic_pipelines/detection.py --hef-path resources/Detection_Person.hef --input /dev/video0 --labels-json resources/Detection_Person.json
