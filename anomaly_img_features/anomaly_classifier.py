from sklearn import svm
import numpy as np


class AnomalyClassifier(object):
    def __init__(self, provider, **kwargs):
        """
        Anomaly classifier using a one class SVM model


        :param provider: class implementing an abstract data provider
        :param kwargs: parameters to the svm classifier
            - nu: (float)
            - kernel: (string) svm kernel
            - gamma: (float) complexity of the decision boundary (rbf kernel)
        :return:
        """
        if provider is None:
            raise AttributeError("A data provider must be provided")

        self._provider = provider

        nu = kwargs.pop("nu", 0.5)
        kernel = kwargs.pop("kernel", "rbf")
        gamma = kwargs.pop("gamma", 0.4)

        self._svm_classifier = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self._svm_classifier.fit(self._provider.get_training_data())

        return

    def predict(self, image_path):
        """
        predict classification using trained one class svm

        :param image_path: path to the image
        :return: ndarray of predictions (-1 for outliers and +1 for inliers
        """

        new_data = self._provider.get_image_descriptor(image_path)

        if new_data.size < 1:
            raise RuntimeError(f"Empty image data descriptor for {image_path}")

        return self._svm_classifier.predict(new_data)

    def get_support_vectors(self):
        return np.copy(self._svm_classifier.support_vectors_)

    def __getstate__(self):
        """
        construct state that should be saved

        :return: picklable state
        """

        state = self.__dict__.copy()
        del state["_provider"]

        return state

    def __setstate__(self, state):
        """
        Restores the state of the object; sets the image descriptor mapper. Note that the training set is not restored

        :param state: saved state dictionary
        """
        self.__dict__.update(state)
        self._provider = None

        return
