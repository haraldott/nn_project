import librosa.display
import pylab
import numpy as np
import os

for dir_path, subdir_list, file_list in os.walk("."):
    for fname in file_list:
        full_file_name_with_path = os.path.join(dir_path, fname)
        file_name_without_ext = os.path.splitext(fname)[0]
        if os.path.splitext(fname)[1] == ".wav":
            sig, fs = librosa.load(full_file_name_with_path)
            spectrogram_dir_path = dir_path.replace("audio", "spectrogram")
            save_path = spectrogram_dir_path + "/" + file_name_without_ext + ".jpg"
            pylab.figure(figsize=[0.6, 0.41])
            pylab.axis('off')
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            S = librosa.feature.melspectrogram(y=sig, sr=fs)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            pylab.close()