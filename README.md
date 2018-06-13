# anomaly-svm-study
Study of one-class SVM to detect visual anomalies in images

### sample
* Contains an example python script that demonstrates the usage of a one-class svm using Sklearn

### anomaly_img_features
* Performs anomaly detection on images by learning from positive datasets and training
a one-class svm. The features are extracted using SURF features and visual bag-of-words,
and deep network features.
* Libraries
    * Tensorflow
    * numpy
    * opencv and opencv-contrib
    * matplotlib
    * scipy
    * sklearn
    * h5py
