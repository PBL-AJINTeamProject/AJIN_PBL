https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/

from hailo_platform import HEF, Device, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams
from picamera2 import Picamera2
import cv2
import numpy as np

hef_path = "/home/kkymin/Downloads/yolov8s.hef"

device = Device()
hef = HEF(hef_path)

# Configure device and get the network group
network_groups = device.configure(hef)
network_group = network_groups[0]  # Assuming the first item is the correct network group

# Create input and output vstream parameters
input_vstream_params = InputVStreamParams()
output_vstream_params = OutputVStreamParams()

# Initialize input and output vstreams
input_vstreams = InputVStreams(network_group, input_vstream_params)
output_vstreams = OutputVStreams(network_group, output_vstream_params)

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 640)})
picam2.configure(config)
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocess the input data
        resized_frame = cv2.resize(frame_bgr, (640, 640))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

        # Send input data to the input vstream (use the .get_stream() method)
        input_stream = input_vstreams.get_stream(0)  # or .get_stream() if there are multiple
        input_stream.send(input_data)

        # Receive output data from the output vstream (use the .get_stream() method)
        output_stream = output_vstreams.get_stream(0)
        output_data = output_stream.receive()

        # Post-process and display results
        for box in output_data[0]:  # Modify according to the output format
            x1, y1, x2, y2, conf, class_id = box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"{class_id}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Object Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
