import pickle
import anomaly_img_features.data_provider.data_provider_sift as sift_provider


def save_model(output_file, model):
    with open(output_file, 'w') as model_file:
        pickle.dump(model, model_file)
    return


def load_model(model_file):
    with open(model_file, 'r') as file:
        model = pickle.load(file)
    return model


def main():
    sift_provider.DataProviderSURF("/Users/harpreetsingh/Downloads/airfield/neg_debug",
                                   num_clusters=3,
                                   resize_image=(400, 225),
                                   patch_size=32)

    return


if __name__ == '__main__':
    main()
