import requests
import sys

API_URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path):
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    r = requests.post(API_URL, files=payload).json()

    return r

img_path = sys.argv[1]
print("Checking results for {}".format(img_path))
result = predict_result(img_path)
print(result)
