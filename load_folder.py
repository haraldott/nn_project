import os
import numpy as np
import pathlib

def load_all_folds(data_dir):
    subsequent_fold = False
    label_names = sorted(item.name for item in pathlib.Path(data_dir).glob('*/') if item.is_file())
    print(label_names)
    for k in range(0, len(label_names), 2):

        feature_file = os.path.join(data_dir, label_names[k])
        labels_file = os.path.join(data_dir, label_names[k+1])
        print("Loading ", feature_file, " ", labels_file)
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print("New Features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True


    return features, labels



# this is used to load the folds incrementally
def load_folds(folds, data_dir):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print(fold_name, "features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True

    return features, labels