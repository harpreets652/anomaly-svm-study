import os
from tensorflow.contrib.keras import applications
import numpy as np

import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderInception(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir):
        """
        This data provider utilizes inception V3 deep network to extract features for each image
        - feature vector size: 2048

        :param training_images_dir: (string) directory containing training data
        """
        self._model = applications.inception_v3.InceptionV3(include_top=False,
                                                            pooling="avg",
                                                            input_shape=(299, 299, 3))

        training_x_list = []
        for root, sub_dirs, files in os.walk(training_images_dir):
            for image_file in files:
                if not image_file.endswith(".jpg"):
                    continue

                image_file_path = os.path.join(root, image_file)
                cv_image = DataProviderInception.read_image(image_file_path, (299, 299))
                cv_image = np.expand_dims(cv_image, axis=0)
                cv_image = cv_image.astype("float32")
                cv_image = applications.inception_v3.preprocess_input(cv_image)

                features = self._model.predict(cv_image)
                training_x_list.append(features)

        self._X = np.vstack(training_x_list)

        # self._mean, self._std_dev = DataProviderInception.compute_normalization_params(self._X)
        # self._X = DataProviderInception.normalize(self._X, self._mean, self._std_dev)

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last average pooling layer of InceptionV3

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderInception.read_image(image_path, (299, 299))
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = cv_image.astype("float32")
        cv_image = applications.inception_v3.preprocess_input(cv_image)

        x = self._model.predict(cv_image)

        return x
