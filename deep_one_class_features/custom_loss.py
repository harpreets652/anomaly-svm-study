from keras import backend as k
import tensorflow as tf

discriminative_loss = 1.0
lambda_coefficient = 0.1


def compute_total_loss(y_true, y_pred):
    if type(y_pred.shape[0]) is not type(int):
        return tf.constant(0.0)

    mean = k.mean(y_pred, axis=0)
    compactness_loss = 0.0

    for i in range(y_pred.shape[0]):
        x_i = y_pred[i]
        mean_minus_i = (y_pred.shape[0] * mean - x_i) / (y_pred.shape[0] - 1)
        z_i = x_i - mean_minus_i
        compactness_loss += k.dot(z_i.T, z_i)

    compactness_loss *= 1 / (y_pred.shape[0] * y_pred.shape[1])

    total_loss = discriminative_loss + (lambda_coefficient * compactness_loss)

    return tf.constant(total_loss)
