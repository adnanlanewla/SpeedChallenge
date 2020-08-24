import os
import numpy as np


def get_filenames_labels(image_directory):
    subdirs, dirs, files = os.walk(image_directory).__next__()
    m = len(files)
    print(m)
    filenames = []
    labels = np.zeros((m, 1), dtype=np.float)
    filenames_counter = 0

    for subdir, dirs, files in os.walk(image_directory):
        # print(files)
        for file in files:
            if(file.endswith('.jpg')):
                image_file_name = os.path.splitext(file)[0]
                split_filename = image_file_name.split('_')
                speed = split_filename[len(split_filename)-1]
                filenames.append(file)
                labels[filenames_counter, 0] = speed
                filenames_counter = filenames_counter + 1

    print(len(filenames))
    print(labels.shape)

    return filenames, labels

if __name__ == '__main__':
    None