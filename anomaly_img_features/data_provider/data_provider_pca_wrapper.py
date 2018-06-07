import operator
import numpy as np
import anomaly_img_features.data_provider.abstract_data_provider as abstract_provider


class DataProviderPCAWrapper(abstract_provider.AbstractDataProvider):

    def __init__(self, provider, num_principle_components):
        """
        This class wraps an data provider and performs dimensionality reduction using PCA
        """
        self._provider = provider
        proj_data, mean_vec, eigen_vec = DataProviderPCAWrapper.run_pca(provider.get_training_data(),
                                                                        num_principle_components)

        self._X = proj_data.real
        self._mean_vec = mean_vec.real
        self._eigenvalue_vec = eigen_vec.real

        return

    def get_training_data(self):
        return self._X

    def get_image_descriptor(self, image_path):
        """
        Compute image descriptor from last average pooling layer of InceptionV3

        :param image_path: (string) path to the image
        :return: (ndarray) numpy array
        """

        x = self._provider.get_image_descriptor(image_path)
        projected_x = np.dot(x - self._mean_vec, self._eigenvalue_vec.T)

        return projected_x.real

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

        # num_components x 256
        eigen_vec_mat = np.array([vec[1] for vec in n_principle_components])

        # n x 256
        training_data_centered = x_mat - mean_vec

        # [n x 256] x [256 x num_components] = [n x num_components]
        projected_data = training_data_centered.dot(eigen_vec_mat.T)

        return projected_data, mean_vec, eigen_vec_mat
