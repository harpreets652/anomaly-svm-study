import os
from tensorflow.contrib.keras import applications
import numpy as np

import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderResNet50(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir):
        """
        This data provider utilizes ResNet 50 deep network to extract features for each image
        - feature vector size: 2048

        :param training_images_dir: (string) directory containing training data
        """

        self._model = applications.resnet50.ResNet50(include_top=False, pooling="avg", input_shape=(224, 224, 3))

        training_x_list = []
        for image_file in os.listdir(training_images_dir):
            if not image_file.endswith(".jpg"):
                continue

            image_file_path = training_images_dir + "/" + image_file
            cv_image = DataProviderResNet50.read_image(image_file_path, (224, 224))
            cv_image = np.expand_dims(cv_image, axis=0)
            cv_image = cv_image.astype("float32")
            cv_image = applications.resnet50.preprocess_input(cv_image)

            features = self._model.predict(cv_image)
            training_x_list.append(features)

        self._X = np.vstack(training_x_list)

        # normalize
        # self._mean = np.mean(self._X, axis=0)
        # self._std_dev = np.std(self._X, axis=0)
        #
        # self._X = np.divide(self._X - self._mean, self._std_dev)
        #
        # for i in range(self._X.shape[0]):
        #     for j in range(self._X.shape[1]):
        #         if np.isnan(self._X[i][j]) or np.isinf(self._X[i][j]):
        #             self._X[i][j] = 0.0

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last average pooling layer of ResNet50

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderResNet50.read_image(image_path, (224, 224))
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = cv_image.astype("float32")
        cv_image = applications.resnet50.preprocess_input(cv_image)

        x = self._model.predict(cv_image)
        # x = np.divide(x - self._mean, self._std_dev)
        #
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         if np.isnan(x[i][j]) or np.isinf(x[i][j]):
        #             x[i][j] = 0.0
        return x
