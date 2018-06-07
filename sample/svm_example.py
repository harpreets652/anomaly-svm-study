import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

"""
some code borrowed from:
http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py 
"""


# note~ this is not the slack variable
# upper bound on the fraction of training errors and lower bound on fraction of support vectors
nu = 0.1

# note~ complexity of the decision boundary: high gamma leads to over fitting
# for rbf kernel
gamma = 0.1

kernel = "rbf"

# generate training data
X = np.random.randn(80, 2)
X_train = X + 1

# generate test data
X = np.random.randn(20, 2)
X_test = X + 1

# Note: since outliers are normally distributed, not all are outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# train one class svm
classifier = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
classifier.fit(X_train)

# classify data
training_predictions = classifier.predict(X_train)
testing_predictions = classifier.predict(X_test)
outlier_predictions = classifier.predict(X_outliers)

# compute errors
training_error = training_predictions[training_predictions == -1].size
test_error = testing_predictions[testing_predictions == -1].size
outlier_error = outlier_predictions[outlier_predictions == 1].size

print(f"Training error: {training_error}/{X_train.shape[0]}, "
      f"" f"regular test error: {test_error}/{X_test.shape[0]}, "
      f"outliers error: {outlier_error}/{X_outliers.shape[0]}")

print(f"number of support vectors: {classifier.support_vectors_.shape}")

# visualize results
xx, yy = np.meshgrid(np.linspace(-8, 8, 200), np.linspace(-8, 8, 200))

Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')

plt.axis('tight')
plt.xlim((-8, 8))
plt.ylim((-8, 8))

plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training data", "regular test data", "outliers"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=8))
plt.xlabel(f"nu: {nu}, kernel: {kernel}, gamma: {gamma}")

plt.show()

# todo: understand the parameters to the one class svm
# note~ decision function returns the distance of point to separating plane
# note~ to save trained model: http://scikit-learn.org/stable/modules/model_persistence.html
