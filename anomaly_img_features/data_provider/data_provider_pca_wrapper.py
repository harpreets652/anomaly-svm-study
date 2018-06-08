import operator
import numpy as np
import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider
import sklearn.decomposition.pca as pca


class DataProviderPCAWrapper(abstract_provider.AbstractDataProvider):

    def __init__(self, provider, num_principle_components):
        """
        This class wraps an data provider and performs dimensionality reduction using PCA
        """
        self._provider = provider
        self._pca_transform = pca.PCA(n_components=num_principle_components, svd_solver="full")
        self._pca_transform.fit(provider.get_training_data())

        self._X = self._pca_transform.transform(provider.get_training_data())

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from provider

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """
        x = self._provider.get_image_descriptor(image_path)
        return self._pca_transform.transform(x)

    @staticmethod
    def run_pca(x_mat, num_principle_components):
        # find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = x_mat.mean(axis=0)

        # find the covariance; row var=False because each column represents a variable, with rows as observations
        cov_mat = np.cov(x_mat, rowvar=False)

        # find the eigen vectors/values
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen = list(zip(eigen_vals, eigen_vecs))
        n_principle_components = sorted(eigen, key=operator.itemgetter(0), reverse=True)[:num_principle_components]

        eigen_vec_mat = np.array([vec[1] for vec in n_principle_components])

        training_data_centered = x_mat - mean_vec

        projected_data = training_data_centered.dot(eigen_vec_mat.T)

        return projected_data, mean_vec, eigen_vec_mat
