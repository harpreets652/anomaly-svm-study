import os
import cv2
from cv2 import xfeatures2d as non_free
import math
import numpy as np
import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderSURF(abstract_provider.AbstractDataProvider):
    def __init__(self, training_images_dir, **kwargs):
        """
        This data provider utilizes the visual bag of words algorithm to map image_file
        to feature vectors for the one-class SVM classifier.
        Process:
            - Partition each image into a grid and generate SURF descriptor from each patch
            - Compute K clusters from all of the features from all of the image_file (visual bag of words)
            - Construct normalized histogram for each image
            - Feature vector is then the values of the normalized histogram (vector quantization)

        :param training_images_dir: (string)
        :param kwargs:
            - num_clusters: (Integer) Size of the visual bag of words
            - resize_image: (tuple(x, y)) resize input image
            - patch_size: (Integer) size of patch to compute a descriptor
        """

        # note~ not much arg validation here...

        self._resize_image = kwargs.pop("resize_image", ())
        self._patch_size = kwargs.pop("patch_size", 16)

        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        bow_model = cv2.BOWKMeansTrainer(kwargs.pop("num_clusters", 500), termination_criteria)

        key_point_tensor = {}
        training_counter = 0

        for root, sub_dirs, files in os.walk(training_images_dir):
            for image_file in files:
                if not image_file.endswith(".jpg"):
                    continue

                training_counter += 1
                if training_counter % 1000 == 0:
                    print(f"{training_counter} images completed")

                image_path = os.path.join(root, image_file)

                cv_image = DataProviderSURF.read_image(image_path, self._resize_image)
                descriptors, key_points = DataProviderSURF.extract_features_descriptors(cv_image, self._patch_size)

                key_point_tensor[image_file] = [cv_image, key_points]
                bow_model.add(descriptors[1])

        print(f"{training_counter} total number of images in training.")

        self._clusters = bow_model.cluster()

        self._img_descriptor_mapper = cv2.BOWImgDescriptorExtractor(non_free.SURF_create(extended=True),
                                                                    cv2.FlannBasedMatcher_create())
        self._img_descriptor_mapper.setVocabulary(self._clusters)

        training_x_list = []
        for img, img_data in key_point_tensor.items():
            image_descriptor = self._img_descriptor_mapper.compute(img_data[0], img_data[1])
            training_x_list.append(image_descriptor)

        self._X = np.vstack(training_x_list)

        return

    def get_image_descriptor(self, image_path):
        """
        Compute quantized image descriptor based on bag of features of the training data

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        cv_image = DataProviderSURF.read_image(image_path, self._resize_image)
        key_points = DataProviderSURF.extract_features(cv_image, self._patch_size)

        return self._img_descriptor_mapper.compute(cv_image, key_points)

    def get_training_data(self):
        return self._X

    def __getstate__(self):
        """
        construct state that should be saved

        :return: picklable state
        """

        state = self.__dict__.copy()

        del state["_img_descriptor_mapper"]
        del state["_X"]

        return state

    def __setstate__(self, state):
        """
        Restores the state of the object; sets the image descriptor mapper. Note that the training set is not restored

        :param state: saved state dictionary
        """
        self.__dict__.update(state)
        self._img_descriptor_mapper = cv2.BOWImgDescriptorExtractor(non_free.SURF_create(extended=True),
                                                                    cv2.FlannBasedMatcher_create())
        self._img_descriptor_mapper.setVocabulary(self._clusters)
        self._X = None

        return

    @staticmethod
    def extract_features_descriptors(image, patch_size=16):
        """
        Computes features based on the patch size

        :param image: input cv2 image
        :param patch_size: size of the patches in the grid
        :return: list of SURF descriptors
        """

        key_points = DataProviderSURF.extract_features(image, patch_size)

        surf = non_free.SURF_create(extended=True)
        descriptors = surf.compute(image, key_points)

        return descriptors, key_points

    @staticmethod
    def extract_features(image, patch_size=16):
        key_points = []
        blob_size = int(math.floor(patch_size / 2))

        start_loc = blob_size
        for x_loc in range(start_loc, image.shape[1], patch_size):
            for y_loc in range(start_loc, image.shape[0], patch_size):
                key_points.append(cv2.KeyPoint(x_loc, y_loc, blob_size))

        # note~ DEBUG CODE
        # key_point_image = cv2.drawKeypoints(image, key_points, None)
        # cv2.imshow("none", key_point_image)
        # cv2.waitKey(0)
        # kp, desc = surf.detectAndCompute(image, None)

        return key_points
