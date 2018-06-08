import os
import anomaly_img_features.network_models.alex_net as alex_net
import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider
import numpy as np


class DataProviderAlexNet(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir):
        """
        This data provider utilizes AlexNet deep network to extract features for each image
        - feature vector size: 4096

        :param training_images_dir: (string) directory containing training data
        """
        model_weights = "/Users/harpreetsingh/github/anomaly-svm-study/anomaly_img_features/network_models/bvlc_alexnet.npy"

        self._model = alex_net.AlexNet(model_weights)

        training_x_list = []
        for image_file in os.listdir(training_images_dir):
            if not image_file.endswith(".jpg"):
                continue

            image_file_path = training_images_dir + "/" + image_file
            cv_image = DataProviderAlexNet.read_image(image_file_path, (227, 227))
            cv_image = cv_image.astype("float32")

            features = self._model.predict(cv_image)
            training_x_list.append(features)

        self._X = np.vstack(training_x_list)

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last average pooling layer of InceptionV3

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderAlexNet.read_image(image_path, (227, 227))
        cv_image = cv_image.astype("float32")

        return self._model.predict(cv_image)
