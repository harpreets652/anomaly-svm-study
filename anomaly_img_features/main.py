import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import anomaly_img_features.data_provider.data_provider_surf as surf_provider
import anomaly_img_features.data_provider.data_provider_vgg16 as vgg_provider
import anomaly_img_features.data_provider.data_provider_resnet50 as resnet_provider
import anomaly_img_features.data_provider.data_provider_inceptv3 as inception
import anomaly_img_features.data_provider.data_provider_pca_wrapper as pca_wrapper
import anomaly_img_features.anomaly_classifier as anomaly_classifier


def save_model(output_file, model):
    with open(output_file, 'w') as model_file:
        pickle.dump(model, model_file)
    return


def load_model(model_file):
    with open(model_file, 'r') as file:
        model = pickle.load(file)
    return model


def run_test_set(classifier, test_set_dir):
    total_num_images = 0
    pred_map = {-1: 0, 1: 1}
    pred_count = np.zeros(2)

    print(f"testing {test_set_dir}")
    for test_image in os.listdir(test_set_dir):
        if not test_image.endswith(".jpg"):
            continue

        test_image_path = test_set_dir + "/" + test_image
        prediction = classifier.predict(test_image_path)

        pred_count[pred_map[prediction[0]]] += 1
        total_num_images += 1

        # print(f"image: {test_image_path}, prediction: {prediction}")

    print(f"Results:\n{pred_count}\n total number of image: {total_num_images}")

    return


def main():
    training_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_big"
    # data_provider = surf_provider.DataProviderSURF(training_data_dir,
    #                                                num_clusters=200,
    #                                                resize_image=(400, 225),
    #                                                patch_size=16)
    data_provider = inception.DataProviderInception(training_data_dir)
    print(f"Using {type(data_provider)}")

    # note~ debug code...==============================
    print("performing pca")
    # provider_pca_wrapper = pca_wrapper.DataProviderPCAWrapper(provider=data_provider,
    #                                                           num_principle_components=10)

    # train_data_plt = provider_pca_wrapper.get_training_data()
    # plt.scatter(train_data_plt[:, 0], train_data_plt[:, 1], c='blueviolet', s=40, edgecolors='k')
    #
    # neg_list = []
    # test_data_dir = "/Users/harpreetsingh/Downloads/airfield/neg"
    # for test_image in os.listdir(test_data_dir):
    #     if not test_image.endswith(".jpg"):
    #         continue
    #
    #     test_image_path = test_data_dir + "/" + test_image
    #     x_i = provider_pca_wrapper.get_image_descriptor(test_image_path)
    #     neg_list.append(x_i)
    #
    # test_x = np.vstack(neg_list)
    # plt.scatter(test_x[:, 0], test_x[:, 1], c='gold', s=40, edgecolors='k')
    # plt.show()

    # print("pca done")
    # Note~ t===============================

    classifier = anomaly_classifier.AnomalyClassifier(data_provider, nu=0.1, gamma=0.1)

    support_vectors = classifier.get_support_vectors()
    print("number of support vectors: \n", support_vectors.shape)

    # test training data
    run_test_set(classifier, training_data_dir)

    # test outlier data
    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/neg"
    run_test_set(classifier, test_data_dir)

    # test inlier data
    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_val"
    run_test_set(classifier, test_data_dir)

    return


if __name__ == '__main__':
    main()
