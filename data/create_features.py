import os

import librosa
import numpy as np


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for dir_path, subdir_list, file_list in os.walk("."):
        for fname in file_list:
            full_file_name_with_path = os.path.join(dir_path, fname)
            file_name_without_ext = os.path.splitext(fname)[0]
            if os.path.splitext(fname)[1] == ".wav":
                sound_clip, s = librosa.load(full_file_name_with_path)
                for (start, end) in windows(sound_clip, window_size):
                    if (len(sound_clip[start:end]) == int(window_size)):
                        signal = sound_clip[start:end]
                        melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                        logspec = librosa.logamplitude(melspec)
                        logspec = logspec.T.flatten()[:, np.newaxis].T
                        log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



# use this to process the audio files into numpy arrays
def save_folds(data_dir):
    for k in range(1, 11):
        fold_name = 'fold' + str(k)
        print("\nSaving " + fold_name)
        features, labels = extract_features(parent_dir, [fold_name])
        labels = one_hot_encode(labels)

        print("Features of", fold_name, " = ", features.shape)
        print("Labels of", fold_name, " = ", labels.shape)

        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        np.save(feature_file, features)
        print("Saved " + feature_file)
        np.save(labels_file, labels)
        print("Saved " + labels_file)


def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)


# uncomment this to recreate and save the feature vectors
parent_dir = "raw/us8k"  # Where you have saved the UrbanSound8K data set"
save_dir = "data/us8k-np-cnn"
assure_path_exists(save_dir)
save_folds(save_dir)
