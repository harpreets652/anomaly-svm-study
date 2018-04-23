import os
import pickle
import numpy as np

import anomaly_img_features.data_provider.data_provider_surf as surf_provider
import anomaly_img_features.data_provider.data_provider_vgg16 as vgg_provider
import anomaly_img_features.data_provider.data_provider_resnet50 as resnet_provider
import anomaly_img_features.data_provider.data_provider_inceptv3 as inception
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

        print(f"image: {test_image_path}, prediction: {prediction}")

    print(f"Results:\n{pred_count}\n total number of image: {total_num_images}")

    return


def main():
    training_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_big"
    data_provider = resnet_provider.DataProviderResNet50(training_data_dir)

    classifier = anomaly_classifier.AnomalyClassifier(data_provider, nu=0.1, gamma=0.8)

    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_big"
    run_test_set(classifier, test_data_dir)

    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/neg"
    run_test_set(classifier, test_data_dir)

    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos_val"
    run_test_set(classifier, test_data_dir)

    return


if __name__ == '__main__':
    main()
