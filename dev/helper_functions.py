import cv2
import os
import numpy as np
from tempfile import TemporaryFile
import pandas as pd

def frame_extractor():
    #Create Image Folder
    os.mkdir('../data/Images')

    filename = '../data/train.mp4'
    #Get video object
    captured_video = cv2.VideoCapture(filename)
    # get the first frame from the video file. Each time the command below
    # is run, a subsequent image is read.
    success, image = captured_video.read()
    frame_number = 1
    fps = 20

    while success:
        second = (frame_number - 1) // fps
        minute = second // 60
        frame_number_within_one_second = frame_number % fps + fps * int(not bool(frame_number % fps))
        # name of each image
        image_name = f'../data/Images/frame_{minute}_{second%60}_{frame_number_within_one_second}.jpg'
        # save the image at 100% quality
        cv2.imwrite(image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # Keep running the command below until there are no more frames left
        # which will be indicated by the 'success' boolean
        success, image = captured_video.read()
        if not bool(frame_number % (fps * 60)):
            print(f'{frame_number//(fps * 60)} minute(s) of video processed')

        frame_number += 1


def file_renamer():
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Images'

    # Change the current python directory to where the files are located
    os.chdir(directory)

    # Get absolute path to the files
    directory = os.getcwd()

    # Get all the filenames
    filenames = os.listdir(directory)

    # sorted(mylist, key=WhatToSortBy)
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    # open the notepad file
    f = open('../train.txt', 'r')

    for file in filenames:
        # read the speed in each line and append it to the filename after an '_'
        speed = f.readline()
        new_name = file.rsplit('.')[0] + '_' + speed.rsplit('\n')[0] + '.jpg'
        os.rename(file, new_name)

    f.close()
    # Change the current python directory to where this script file is
    os.chdir(os.path.dirname(os.path.realpath(__file__)))


def image_ndim_reshaper():
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Local Data/Images'

    # Change the current python directory to where the files are located
    os.chdir(directory)

    # Get absolute path to the files
    directory = os.getcwd()

    # Get all the filenames
    filenames = os.listdir(directory)

    # sorted(mylist, key=WhatToSortBy)
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    array_col_size = cv2.imread(filenames[0])
    array_col_size = array_col_size.shape[0] * array_col_size.shape[1] * array_col_size.shape[2]
    training_set = np.empty((len(filenames)-1, array_col_size), dtype='uint8')
    fps = 20
    for i in range(len(filenames)-1):
        A = cv2.imread(filenames[i])
        A = np.ravel(A)
        B = cv2.imread(filenames[i+1])
        B = np.ravel(B)
        C = A - B
        training_set[i] = C
        if not bool(i % (fps * 60)):
            print(f'{i //(fps * 60)} minute(s) of video processed')

    training_set = training_set.T
    np.savetxt('../Frame_Differences.csv', training_set, delimiter=',', fmt='%u')
    # pd.DataFrame(training_set.T).to_csv('../Frame_Differences_0-2.csv', header=False, index=False)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

def save_images_as_np_array(image_directory, array_filename):
    # Change the current python directory to where the files are located
    os.chdir(image_directory)
    # Get absolute path to the files
    image_directory = os.getcwd()
    # Get all the filenames
    filenames = os.listdir(image_directory)
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))
    array_col_size = cv2.imread(filenames[0])
    fps = 20
    # Append the numpy array to a list in a loop
    all_images = []
    for i in range(len(filenames)-1):
        A = cv2.imread(filenames[i])
        all_images.append(A)
        A = np.ravel(A)
        if not bool(i % (fps * 60)):
            print(f'{i //(fps * 60)} minute(s) of video processed')

    array_of_images = np.array(all_images, dtype=np.uint8)
    np.savez(array_filename, array_of_images)

def load_numpy_array(array_filename):
    container = np.load(array_filename)
    data = [container[key] for key in container]
    array_of_images = data[0]
    return array_of_images

def speed_difference():
    path = '../data/train.txt'
    data = np.loadtxt(path)
    data = np.diff(data)
    np.savetxt('../data/Speed_Differences.txt', data, fmt='%f')

#if __name__ == '__main__':
    # filename1 = '../data/Local Data/Images/frame_0_0_1_28.105569.jpg'
    # filename2 = '../data/Local Data/Images/frame_0_2_15_27.266277.jpg'
    # A = cv2.imread(filename1)
    # B = cv2.imread(filename2)
    # C = A + B
    # cv2.imwrite('../data/Local Data/test2.jpg', C)