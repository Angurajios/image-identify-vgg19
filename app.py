
import base64
import io
import os
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)
CORS(app)

CLASS_NAMES = ["Apple", "Elephant", "Pen"]

interpreter = tflite.Interpreter(model_path="vgg19_CustomImage.tflite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def preprocess_image(image_data_url):
    """
    Converts a base64 image (from file upload or camera capture) into the
    exact format VGG16 expects: 224x224, BGR order, ImageNet mean-subtracted.
    This replicates keras.applications.vgg16.preprocess_input manually,
    so we don't need to install full TensorFlow just for this one function.
    """
    base64_text = image_data_url.split(",")[1]
    image_bytes = base64.b64decode(base64_text)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    array = np.array(image).astype(np.float32)

    # RGB -> BGR (VGG16 was trained on BGR images)
    array = array[..., ::-1]

    # Subtract ImageNet mean per channel (this is what preprocess_input does)
    mean = [103.939, 116.779, 123.68]
    array[..., 0] -= mean[0]
    array[..., 1] -= mean[1]
    array[..., 2] -= mean[2]

    array = np.expand_dims(array, axis=0)
    return array


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        processed = preprocess_image(data["image"])

        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        all_scores = {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "label": predicted_label,
            "confidence": round(confidence, 2),
            "all_scores": all_scores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)