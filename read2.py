from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3xvzQvj0i7ihj4wR7zTE"
)

result = CLIENT.infer("D:/WorkSpace/PycharmProjects/datasets/images/val/0000_06886_b.jpg", model_id="license-plates-recognition-iuk6u/1")

print(result['predictions'])


