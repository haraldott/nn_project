import os
for dir_path, subdir_list, file_list in os.walk("."):
    for fname in file_list:
        full_file_name = os.path.join(dir_path, fname)
        file_name_without_ext = os.path.splitext(fname)[0]
        if os.path.splitext(fname)[1] == ".wav":

            dir_path = dir_path.replace("audio", "spectogram")
            print(dir_path + "/" + file_name_without_ext + ".jpg")
            #print(os.path.join(dir_path, '/spectogram/'))