import keras

loss_discriminative = 1000.0


# todo: this is able to access the loss_discriminative value
def compute_total_loss(y_true, y_pred):
    return loss_discriminative * keras.losses.mean_squared_error(y_true, y_pred)
