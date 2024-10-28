from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3xvzQvj0i7ihj4wR7zTE"
)
your_image = "D:/WorkSpace/PycharmProjects/datasets/images/train/0141_04768_b_bright.jpg"
result = CLIENT.infer(your_image, model_id="license-plates-recognition-iuk6u/1")

print(result)