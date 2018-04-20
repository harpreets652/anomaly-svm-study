import cv2


class AbstractDataProvider(object):

    def get_training_data(self):
        raise NotImplementedError("Cannot use abstract data provider")
        pass

    def get_image_descriptor(self, image_path):
        raise NotImplementedError("Cannot use abstract data provider")
        pass

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
