https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst

https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/

from hailo_platform import HEF, Device, InputVStreams, OutputVStreams
import cv2
import numpy as np

# Path to the HEF file
hef_path = "/path/to/yolov8.hef"

# Initialize the Hailo device and load the HEF file
device = Device()
hef = HEF(hef_path)

# Configure the network group
network_groups = device.configure(hef)
network_group = network_groups[0]

# Set up input and output vstreams
input_vstreams = InputVStreams(network_group)
output_vstreams = OutputVStreams(network_group)

# Initialize camera for real-time video capture
cap = cv2.VideoCapture(0)  # Use the default camera
if not cap.isOpened():
    raise Exception("Could not open the camera!")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the input frame
        resized_frame = cv2.resize(frame, (640, 640))  # Match the input size of the model
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0  # Normalize

        # Send the input data to the Hailo model
        input_stream = input_vstreams[0]
        input_stream.send(input_data)

        # Receive the output data from the Hailo model
        output_stream = output_vstreams[0]
        output_data = output_stream.receive()

        # Post-process and visualize the results
        for box in output_data[0]:  # Adjust based on the model's output format
            x1, y1, x2, y2, conf, class_id = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Class {class_id}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    device.release()
