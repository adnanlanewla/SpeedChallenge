import os
import numpy as np


def get_filenames_labels(image_directory):
    filenames = []
    labels = []

    for file in os.listdir(image_directory):
        if(file.endswith('.jpg')):
            image_file_name = os.path.splitext(file)[0]
            split_filename = image_file_name.split('_')
            speed = split_filename[len(split_filename)-1]
            filenames.append(os.path.join(image_directory,file))
            labels.append(speed)

    print(len(filenames))
    print(len(labels))

    return filenames, labels


if __name__ == '__main__':
    None