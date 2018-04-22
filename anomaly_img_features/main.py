import os
import pickle
import anomaly_img_features.data_provider.data_provider_surf as surf_provider
import anomaly_img_features.data_provider.data_provider_vgg16 as vgg_provider
import anomaly_img_features.anomaly_classifier as anomaly_classifier


def save_model(output_file, model):
    with open(output_file, 'w') as model_file:
        pickle.dump(model, model_file)
    return


def load_model(model_file):
    with open(model_file, 'r') as file:
        model = pickle.load(file)
    return model


def main():
    training_data_dir = "/Users/harpreetsingh/Downloads/airfield/pos"
    data_provider = vgg_provider.DataProviderVGG16(training_data_dir)

    classifier = anomaly_classifier.AnomalyClassifier(data_provider)

    test_data_dir = "/Users/harpreetsingh/Downloads/airfield/neg"

    correct_prediction_counter = 0
    total_num_images = 0
    for test_image in os.listdir(test_data_dir):
        if not test_image.endswith(".jpg"):
            continue

        test_image_path = test_data_dir + "/" + test_image
        prediction = classifier.predict(test_image_path)

        if prediction[0] == -1:
            correct_prediction_counter += 1
        total_num_images += 1

        print(f"image: {test_image_path}, prediction: {prediction}")

    print(f"Accuracy: {correct_prediction_counter/total_num_images}")

    return


if __name__ == '__main__':
    main()
