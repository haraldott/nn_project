from pydub import AudioSegment
import os

for dir_path, subdir_list, file_list in os.walk("."):
    for fname in file_list:
        full_file_name = os.path.join(dir_path, fname)
        file_name_without_ext = os.path.splitext(fname)[0]
        if os.path.splitext(fname)[1] == ".wav":
            song = AudioSegment.from_wav(full_file_name)
            length = int(song.duration_seconds)
            for i in range(1, length):
                song[i*1000:(i+1)*1000].export(os.path.join(dir_path, file_name_without_ext)+str(i)+".wav", format="wav")
