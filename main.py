import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
import numpy as np
import json

def load_and_preprocess_data(data_dir, subset, target_size=(224, 224)):

    print('[INFO] loading dataset...')
    images_dir = os.path.join(data_dir, subset, 'image')
    json_dir = os.path.join(data_dir, subset, 'json')

    # main variable
    data = []
    target = []
    filenames = []

    # helping variable
    coordinates = None
    startX = None
    startY = None
    endX = None
    endY = None

    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):

            image_path = os.path.join(images_dir, filename)
            json_path = os.path.join(json_dir, filename.replace(".png", ".json"))
            # print(image_path)

            if os.path.exists(json_path):
                # Load JSON annotation
                with open(json_path, 'r') as file:
                    annotation = json.load(file)

                # Extract total and coordinates
                w = annotation["meta"]["image_size"]["width"]
                h = annotation["meta"]["image_size"]["height"]

                for line in annotation.get("valid_line", []):
                    if line.get("category") == "total.total_price" and line.get("words"):
                        for word in line["words"]:
                            if word.get("is_key") == 0:
                                coordinates = word["quad"]
                                startX = coordinates['x1']
                                startY = coordinates['y1']
                                endX = coordinates['x4']
                                endY = coordinates['y4']
                                break  # Stop iterating once the key word is found
                        if coordinates is not None:
                            break

                image = load_img(image_path, target_size=target_size)
                image = img_to_array(image)

                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h
                # Append to the lists
                target.append((startX, startY, endX, endY))
                data.append(image)
                filenames.append(filename)
                # print(filename)

    return np.array(data, dtype='float32')/255.0, np.array(target, dtype='float32'), filenames

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Check accuracy
        if (logs.get('accuracy') > 0.7):
            # Stop if threshold is met
            print("\nAccuracy is bigger than 0.7 so cancelling training!")
            self.model.stop_training = True
def main():
    data_dir = 'data'
    trainImages, trainTargets, trainFilenames = load_and_preprocess_data(data_dir, 'train')
    devImages, devTarget, devFilenames = load_and_preprocess_data(data_dir, 'dev')
    testImages, testTargets, testFilenames = load_and_preprocess_data(data_dir, 'test')
    print(len(trainImages))
    print(len(trainTargets))

    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False

    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)

    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)

    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)

    callback = myCallback()
    model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    model.fit(trainImages, trainTargets,
              validation_data=(devImages, devTarget),
              batch_size=32,
              epochs=30,
              verbose=1,
              callbacks=[callback])
    return model

if __name__ == '__main__':
    model = main()
    model.save("capstone.h5")