https://blog.naver.com/roboholic84/223485800255

https://github.com/hailo-ai/hailo_model_zoo?tab=readme-ov-file

https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst


from hailo_platform.tools.hef_profile_converter import HefProfileConverter

input_hef_path = "model_hailo8.hef"
output_hef_path = "model_hailo8l.hef"

converter = HefProfileConverter()
converter.convert(input_hef_path, output_hef_path, target_device="hailo8l")

print(f"Converted HEF saved to: {output_hef_path}")

