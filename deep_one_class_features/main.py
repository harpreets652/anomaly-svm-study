from keras.models import Model
from keras import applications
from keras import utils
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import deep_one_class_features.custom_loss as my_loss


def main():
    # construct reference model
    ref_model = applications.VGG16()
    ref_model.compile(optimizer='sgd', loss='categorical_crossentropy')

    # construct secondary model with shared layers
    secondary_model = Model(inputs=ref_model.inputs, outputs=ref_model.get_layer("fc1").output)
    secondary_model.compile(optimizer='Nadam', loss=my_loss.doc_total_loss)

    # manually train on batches
    ref_train_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_small"
    ref_train_label_file = "/Users/harpreetsingh/Downloads/ILSVRC2012_validation_ground_truth.txt"
    target_train_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_small"

    tot_loss, ref_loss = train(ref_model, secondary_model, 5, 2,
                               ref_train_data_dir, ref_train_label_file, target_train_data_dir)

    # save secondary model after training
    my_file_name = "path/to/results/trained_model.h5"
    secondary_model.save(my_file_name)

    visualize_data(tot_loss, "total loss history")
    visualize_data(ref_loss, "ref loss history")

    return


def visualize_data(data, title):
    plt.scatter(data[:, 0], data[:, 1], c='blueviolet', s=40, edgecolors='k')
    plt.title(title)
    plt.show()
    return


def train(ref_model, secondary_model, batch_size, num_epochs,
          ref_train_data_dir, ref_train_label_file, target_train_data_dir):
    ref_images_list = get_images_list(ref_train_data_dir)
    ref_data_labels = read_ref_data_labels(ref_train_label_file)
    target_images_list = get_images_list(target_train_data_dir)

    num_iterations = max(len(target_images_list) / batch_size, 1) * num_epochs

    total_loss_history = []
    ref_loss_history = []
    for i in range(int(num_iterations)):
        ref_batch_x, ref_batch_y = read_image_batch(ref_images_list, batch_size, ref_data_labels)
        target_batch_x, target_batch_y = read_image_batch(ref_images_list, batch_size)

        my_loss.discriminative_loss = ref_model.test_on_batch(ref_batch_x, ref_batch_y)
        ref_loss_history.append((i, my_loss.discriminative_loss))

        loss = secondary_model.train_on_batch(target_batch_x, target_batch_y)
        total_loss_history.append((i, loss))

    return np.array(total_loss_history), np.array(ref_loss_history)


def read_image_batch(image_list, batch_size, class_labels=None):
    batch_images = []
    classification = []
    for k in range(batch_size):
        rand_loc = random.randrange(0, len(image_list))
        cv_image = read_image(image_list[rand_loc], (224, 224))

        batch_images.append(cv_image)

        if not class_labels:
            classification.append(4095)
        else:
            # todo: verify the labels correspond to the correct image
            classification.append(class_labels[rand_loc])

    return np.array(batch_images), utils.to_categorical(np.array(classification))


def read_ref_data_labels(data_file):
    labels = []
    with open(data_file, 'r') as label_file:
        lines = label_file.readlines()
        labels = [int(line) for line in lines]

    return labels


def get_images_list(list_dir):
    images_list = []
    for root, sub_dirs, files in os.walk(list_dir):
        images_list += [os.path.join(root, file) for file in files if file.endswith(".jpg")]

    return images_list


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


if __name__ == '__main__':
    main()
