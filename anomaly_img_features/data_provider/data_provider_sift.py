import os
import cv2 as cv


class DataProviderSURF(object):
    def __init__(self, training_images_dir, **kwargs):
        """
        This data provider utilizes the visual bag of words algorithm to map images to feature vectors for the one-class
        SVM classifier.
        Process:
            - Partition each image into a grid and generate SURF descriptor from each patch
            - Compute K clusters from all of the features from all of the images (visual bag of words)
            - Construct normalized histogram for each image
            - Feature vector is then the values of the normalized histogram (vector quantization)

        :param training_images_dir: (string)
        :param kwargs:
            - num_clusters: (Integer) Size of the visual bag of words
            - resize_image: (tuple(x, y)) If input images need to be resized
            - grid_size: (tuple(x, y)) number of patches (i.e. number of features per image)
        """

        # note~ not much arg validation here...

        self._num_clusters = kwargs.pop("num_clusters", 200)
        self._resize_image = kwargs.pop("resize_image", ())
        self._grid_size = kwargs.pop("grid_size", ())

        features_tensor = []
        # extract features for all the images num imgs x size of grid x 128 (Sift features) tensor
        for images in os.listdir(training_images_dir):
            DataProviderSURF.extract_features(training_images_dir + "/" + images)

        # use sklearn k-means clustering class to find x number of clusters

        # iterate over each image, and perform vector quantization

        # save the k means object and training set

        return

    @staticmethod
    def extract_features(image_file, image_resize=(), grid_size=()):
        image = cv.imread(image_file)

        if not image:
            raise RuntimeError(f"Unable to open {image_file}")

        if len(image_resize) > 0:
            image = cv.resize(image, image_resize)

        # compute key points...just the center of the patch?

        # compute descriptors for each key point

        # return descriptors

        return
