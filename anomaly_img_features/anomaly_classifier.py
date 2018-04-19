from sklearn import svm


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
        # todo: training data needs to be a 2d array
        self._svm_classifier.fit(self._provider.get_training_data())

        return

    def predict(self, new_data):
        """
        predict classification using trained one class svm

        :param new_data: ndarray
        :return: ndarray of predictions (-1 for outliers and +1 for inliers
        """

        return self._svm_classifier.predict(new_data)

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
