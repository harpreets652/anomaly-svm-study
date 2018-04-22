import os
from tensorflow.contrib.keras import applications
import tensorflow.contrib.keras as keras
import numpy as np

import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderVGG16(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir):
        """
        This data provider utilizes VGG 16 deep network to extract features for each image

        :param training_images_dir: (string) directory containing training data
        """

        base_model = applications.VGG16()
        self._model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

        training_x_list = []
        for image_file in os.listdir(training_images_dir):
            if not image_file.endswith(".jpg"):
                continue

            image_file_path = training_images_dir + "/" + image_file
            cv_image = DataProviderVGG16.read_image(image_file_path, (224, 224))
            cv_image = np.expand_dims(cv_image, axis=0)
            cv_image = cv_image.astype("float32")
            cv_image = applications.vgg16.preprocess_input(cv_image)

            features = self._model.predict(cv_image)
            training_x_list.append(features)

        self._X = np.vstack(training_x_list)
        return

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last fully connected layer from VGG 16

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderVGG16.read_image(image_path, (224, 224))
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = cv_image.astype("float32")
        cv_image = applications.vgg16.preprocess_input(cv_image)

        return self._model.predict(cv_image)

    def get_training_data(self):
        return self._X
