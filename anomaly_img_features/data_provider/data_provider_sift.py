import os
import cv2
from cv2 import xfeatures2d as nonfree
import math


class DataProviderSURF(object):
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

        for image_file in os.listdir(training_images_dir):
            image_path = training_images_dir + "/" + image_file
            print(f"Image {image_path}")
            cv_image = DataProviderSURF.read_image(image_path, self._resize_image)
            descriptors = DataProviderSURF.extract_features(cv_image, self._patch_size)
            bow_model.add(descriptors[1])

        # use sklearn k-means clustering class to find x number of clusters
        clusters = bow_model.cluster()

        # iterate over each image, and perform vector quantization

        # save the k means object and training set

        return

    @staticmethod
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

    @staticmethod
    def extract_features(image, patch_size=16):
        """
        Computes features based on the patch size

        :param image: input cv2 image
        :param patch_size: size of the patches in the grid
        :return: list of SURF descriptors
        """

        # compute key points, center of the patch; rows and cols are reversed
        key_points = []
        blob_size = int(math.floor(patch_size / 2))
        start_loc = blob_size
        for x_loc in range(start_loc, image.shape[1], patch_size):
            for y_loc in range(start_loc, image.shape[0], patch_size):
                key_points.append(cv2.KeyPoint(x_loc, y_loc, blob_size))

        surf = nonfree.SURF_create(extended=True)
        descriptors = surf.compute(image, key_points)

        # note~ DEBUG CODE
        # key_point_image = cv2.drawKeypoints(image, key_points, None)
        # cv2.imshow("none", key_point_image)
        # cv2.waitKey(0)
        # kp, desc = surf.detectAndCompute(image, None)

        return descriptors
