import os
from tensorflow.contrib.keras import applications
import tensorflow.contrib.keras as keras
import numpy as np

import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderVGG16(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir):
        """
        This data provider utilizes VGG 16 deep network to extract features for each image
        - feature vector size: 4096

        :param training_images_dir: (string) directory containing training data
        """

        self._model = applications.VGG16(include_top=False)

        training_x_list = []
        training_counter = 0

        for root, sub_dirs, files in os.walk(training_images_dir):
            for image_file in files:
                if not image_file.endswith(".jpg"):
                    continue

                training_counter += 1
                if training_counter % 1000 == 0:
                    print(f"{training_counter} images completed")

                image_file_path = os.path.join(root, image_file)
                cv_image = DataProviderVGG16.read_image(image_file_path, (224, 224))
                cv_image = np.expand_dims(cv_image, axis=0)
                cv_image = cv_image.astype("float32")
                cv_image = applications.vgg16.preprocess_input(cv_image)

                features = self._model.predict(cv_image)
                training_x_list.append(features)

        print(f"{training_counter} total number of images in training.")

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
