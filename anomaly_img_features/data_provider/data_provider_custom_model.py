import os
from keras import applications
import keras
import numpy as np

import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider
import deep_one_class_features.custom_loss as custom_loss


class DataProviderCustomModel(abstract_provider.AbstractDataProvider):
    MODEL_IMAGE_SHAPE = (229, 229)

    def __init__(self, training_images_dir, model_file_path):
        """
        This data provider utilizes a custom model loaded from an .h5 file

        :param training_images_dir: (string) directory containing training data
        """

        self._model = keras.models.load_model(model_file_path,
                                              custom_objects={'doc_total_loss': custom_loss.doc_total_loss})

        training_x_list = []
        training_counter = 0

        for root, sub_dirs, files in os.walk(training_images_dir):
            for image_file in files:
                if not image_file.endswith((".jpg", ".JPEG")):
                    continue

                training_counter += 1
                if training_counter % 1000 == 0:
                    print(f"{training_counter} images completed")

                image_file_path = os.path.join(root, image_file)
                cv_image = DataProviderCustomModel.read_image(image_file_path,
                                                              DataProviderCustomModel.MODEL_IMAGE_SHAPE)

                cv_image = np.expand_dims(cv_image, axis=0)
                cv_image = cv_image.astype("float32")
                cv_image = applications.inception_v3.preprocess_input(cv_image)

                features = self._model.predict(cv_image)
                training_x_list.append(features)

        print(f"{training_counter} total number of images in training.")

        self._X = np.vstack(training_x_list)

        # self._mean, self._std_dev = DataProviderCustomModel.compute_normalization_params(self._X)
        # self._X = DataProviderCustomModel.normalize(self._X, self._mean, self._std_dev)

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last fully connected layer from VGG 16

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderCustomModel.read_image(image_path, DataProviderCustomModel.MODEL_IMAGE_SHAPE)
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = cv_image.astype("float32")
        cv_image = applications.inception_v3.preprocess_input(cv_image)

        x = self._model.predict(cv_image)

        return x
