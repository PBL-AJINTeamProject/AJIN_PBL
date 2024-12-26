https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/


cd hailo-rpi5-examples

source setup_env.sh


python3 basic_pipelines/detection.py --hef-path resources/best2.hef --input rpi --labels-json resources/barcode-labels.json

