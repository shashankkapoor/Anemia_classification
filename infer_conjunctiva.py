import cv2
from tensorflow import keras
import numpy as np
import argparse


# read images and resize
# Make Sure the image inputed, should be the segmented image of conjunctiva otherwise the model result is not
# As to the trained model. 

def read_img(filepath) -> np.array:
    img = cv2.imread(filepath)
    img = cv2.resize(img, (64, 64))
    return img


def get_efficientnetv2b0():
    cnn = keras.applications.EfficientNetV2B0(
        include_top=False,
        # weights="weights/efficientnetv2-b0_notop.h5",
        input_tensor=None,
        input_shape=(64, 64, 3),
        include_preprocessing=True,
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
    model.load_weights("weights/weights_conjunctiva/efficientnetV2B0_model.h5")
    return model


def get_efficientnetv2b1():
    cnn = keras.applications.EfficientNetV2B1(
        include_top=False,
        #weights="weights/efficientnetv2-b1_notop.h5",
        input_tensor=None,
        input_shape=(64, 64, 3),
        include_preprocessing=True,
    )
    cnn.trainable = False
    classifier = keras.models.Sequential(
        [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model = keras.models.Sequential(
        [
            cnn,
            classifier,
        ],
        name=cnn.name,
    )
    model.load_weights("weights/weights_conjunctiva/efficientnetV2B1_model.h5")
    return model


def get_efficientnetv2b2():
    cnn = keras.applications.EfficientNetV2B2(
        include_top=False,
        #weights="weights/efficientnetv2-b2_notop.h5",
        input_tensor=None,
        input_shape=(64, 64, 3),
        include_preprocessing=True,
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
        ]
    )
    model.load_weights("weights/weights_conjunctiva/efficientnetV2B2_model.h5")
    return model


def get_model(model_name):
    if model_name == "efficientnetv2b0":
        return get_efficientnetv2b0()
    if model_name == "efficientnetv2b1":
        return get_efficientnetv2b1()
    if model_name == "efficientnetv2b2":
        return get_efficientnetv2b2()


# parse arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to infer from")
    parser.add_argument("img_path", help="Path to the image to predict class")
    args = parser.parse_args()
    # print(args.model_name)
    # print(args.img_path)

    return args


def main():
    args = get_args()
    img = read_img(args.img_path)
    model = get_model(args.model_name)

    # incase of single channel image, model is trained on BGR images
    if len(img.shape) == 2 or (img.shape[0] == 1 and len(img.shape) == 3):
        img = np.stack([img, img, img], axis=0)

    img = img[None, :, :, :]  # adding batch dim to 3 ch img, ch first
    model.trainable = False  # just a precaussion
    pred = model.predict(img, verbose=0)
    pred = np.argmax(pred, axis=1)

    if pred[0] == 0:
        print("Anemia")
    if pred[0] == 1:
        print("Non-anemia")


if __name__ == "__main__":
    main()
