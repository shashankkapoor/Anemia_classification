import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from tensorflow import keras

# from tensorflow.keras.utils import img_to_array
import numpy as np
import flask
from PIL import Image
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    # img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


# preprocess image
def process_img(img) -> np.array:
    # img = cv2.imread(filepath)
    # img = img_to_array(img)
    img = cv2.resize(img, (227, 227))
    return img


# get model
def get_convnexttiny():
    cnn = keras.applications.ConvNeXtTiny(
        model_name="convnext_tiny",
        include_top=False,
        include_preprocessing=True,
        weights="weights/weights_palm/convnext_tiny_notop.h5",
        input_shape=(227, 227, 3),
    )
    cnn.trainable = False
    classifier = keras.models.Sequential(
        [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ],
        name="classifier",
    )
    model = keras.models.Sequential(
        [
            cnn,
            classifier,
        ],
        name=cnn.name,
    )
    classifier.load_weights("weights/weights_palm/anemia-classifier-ConvNeXtTiny.h5")

    return model


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = create_opencv_image_from_stringio(image, cv2.IMREAD_COLOR)
            # image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = process_img(image)
            image = image[None, :, :, :3]
            model.trainable = False
            pred = model.predict(image, verbose=0)
            data["prob"] = [float(pred[0, 0]), float(pred[0, 1])]

            pred = np.argmax(pred, axis=1)  # bad practice, will change later
            if pred[0] == 0:
                data["anemia"] = True
            if pred[0] == 1:
                data["anemia"] = False

            data["success"] = True
    return flask.jsonify(data)


if __name__ == "__main__":
    print(
        (
            "* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"
        )
    )
    model = get_convnexttiny()
    print("Model loaded with weights. ")
    app.run(host='0.0.0.0', debug=False, port=5000)
