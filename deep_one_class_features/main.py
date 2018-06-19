import keras
from keras.models import Model
from keras import layers
from keras import applications
import os
import cv2
import numpy as np

import deep_one_class_features.custom_loss as my_loss


def main():
    # construct reference model...note~ use test_on_batch...categorical_crossentropy?
    ref_model = applications.VGG16()
    ref_model.compile(optimizer='sgd', loss='mean_squared_error')

    # construct secondary model, todo: choice for optimizer
    secondary_model = Model(inputs=ref_model.inputs, outputs=ref_model.get_layer("fc1").output)
    secondary_model.compile(optimizer='sgd', loss=my_loss.compute_total_loss)

    # manually train on batches

    x, y = read_images("/Users/harpreetsingh/Downloads/airfield/pos_small")
    y = keras.utils.to_categorical(y)

    for i in range(0, 3):
        loss = secondary_model.train_on_batch(x, y)
        print(f"loss: {loss}")

    return


def read_images(training_images_dir):
    images_list = []
    classifications = []
    for root, sub_dirs, files in os.walk(training_images_dir):
        for image_file in files:
            if not image_file.endswith(".jpg"):
                continue

            image_file_path = os.path.join(root, image_file)
            cv_image = read_image(image_file_path, (224, 224))

            images_list.append(cv_image)
            classifications.append(999)

    return np.array(images_list), np.array(classifications)


def read_image(image_file, resize_image=()):
    """
    Read an image and resize it, if necessary

    :param image_file: absolute image path
    :param resize_image: (x, y) tuple for new image dimensions
    :return: cv2 image
    """

    cv_image = cv2.imread(image_file)

    if cv_image is None:
        raise RuntimeError(f"Unable to open {image_file}")

    if len(resize_image) > 0:
        cv_image = cv2.resize(cv_image, resize_image)

    return cv_image

if __name__ == '__main__':
    main()
