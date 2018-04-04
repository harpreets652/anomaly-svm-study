import os
import cv2 as cv


class DataProviderSIFT(object):
    def __init__(self, p_training_images_dir):
        features_tensor = []
        # extract features for all the images num imgs x size of grid x 128 (Sift features) tensor
        for image_file in os.listdir(p_training_images_dir):
            # todo: does python 3.6 return file names as bytes?
            DataProviderSIFT.extract_features(image_file)

        # use sklearn k-means clustering class to find x number of clusters

        # iterate over each image, and perform vector quantization

        # save the k means object and training set

        return

    @staticmethod
    def extract_features(p_image_file):
        image = cv.imread(p_image_file)

        return
