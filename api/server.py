import flask
from PIL import Image
import io
from prediction import inference
from flask import Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        result = inference(image)

        data["response"] = result
        data["success"] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
