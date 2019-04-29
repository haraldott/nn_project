import os
import pathlib

import librosa
import numpy as np


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(dir, label, bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    print(dir)
    for fn in dir.glob("*"):
        print(fn.name)
        if fn.name.endswith(".wav"):
            print("fn", fn)
            sound_clip, s = librosa.load(os.path.join(dir, fn.name))
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[int(start):int(end)]) == int(window_size)):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.power_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels)


def save_features(data_dir):
    label_names = sorted(item.name for item in pathlib.Path("data").glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_names)
    print(label_to_index)
    for k in range(1, 7):
        class_name = str(label_names[k - 1])
        print("\nSaving " + class_name)
        print(pathlib.Path(os.path.join(parent_dir, label_names[k - 1])))
        print(label_to_index[label_names[k - 1]])
        features, labels = extract_features(pathlib.Path(os.path.join(parent_dir, label_names[k - 1])),
                                            label_to_index[label_names[k - 1]])

        feature_file = os.path.join(data_dir, class_name + '_x.npy')
        labels_file = os.path.join(data_dir, class_name + '_y.npy')
        np.save(feature_file, features)
        print("Saved " + feature_file)
        np.save(labels_file, labels)
        print("Saved " + labels_file)


def create_folder(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)


parent_dir = "data"
save_dir = "data/processed_data/"
create_folder(save_dir)
save_features(save_dir)
