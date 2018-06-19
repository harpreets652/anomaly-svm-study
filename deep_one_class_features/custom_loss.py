from keras import backend as k

discriminative_loss = 0.0
scalar_lambda = 0.1


def compute_total_loss(y_true, y_pred):
    mean = k.mean(y_pred, axis=0)

    compactness_loss = 0.0
    for i in range(y_pred.shape[0]):
        x_i = y_pred[i]
        mean_minus_i = (y_pred.shape[0] * mean - x_i) / (y_pred.shape[0] - 1)
        z_i = x_i - mean_minus_i
        compactness_loss += k.dot(z_i.T, z_i)

    compactness_loss *= 1 / (y_pred.shape[0] * y_pred.shape[1])

    total_loss = discriminative_loss + (scalar_lambda * compactness_loss)

    return total_loss
