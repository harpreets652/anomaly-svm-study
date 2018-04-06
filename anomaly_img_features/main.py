import numpy as np
import anomaly_img_features.data_provider.data_provider_sift as sift_provider

sift_provider.DataProviderSURF("/Users/harpreetsingh/Downloads/airfield/neg_debug",
                               num_clusters=3,
                               resize_image=(400, 225),
                               patch_size=32)

# sift_provider.DataProviderSURF.extract_features(np.zeros((56, 56)))
