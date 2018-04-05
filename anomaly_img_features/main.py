import numpy as np
import anomaly_img_features.data_provider.data_provider_sift as sift_provider

sift_provider.DataProviderSURF("/Users/harpreetsingh/Downloads/airfield/neg", num_clusters=1000, resize_image=(800, 450))
# sift_provider.DataProviderSURF.extract_features(np.zeros((56, 56)))
