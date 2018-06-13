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
        training_counter = 0

        for root, sub_dirs, files in os.walk(training_images_dir):
            for image_file in files:
                if not image_file.endswith(".jpg"):
                    continue

                training_counter += 1
                if training_counter % 1000 == 0:
                    print(f"{training_counter} images completed")

                image_file_path = os.path.join(root, image_file)
                cv_image = DataProviderAlexNet.read_image(image_file_path, (227, 227))
                cv_image = cv_image.astype("float32")

                features = self._model.predict(cv_image)
                training_x_list.append(features)

        print(f"{training_counter} total number of images in training.")

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
