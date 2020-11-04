import os
import numpy as np
import cv2

def file_renamer():
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Local_Data/Images'
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
    f = open('../../train.txt', 'r')

    for file in filenames:
        # read the speed in each line and append it to the filename after an '_'
        speed = f.readline()
        new_name = file.rsplit('.')[0] + '_' + speed.rsplit('\n')[0] + '.jpg'
        os.rename(file, new_name)

    f.close()
    # Change the current python directory to where this script file is
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

def image_to_RGB(start_image_number = 1, num_of_images_per_file = 3600):
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Local_Data/Cropped_Images'
    # Change the current python directory to where the files are located
    os.chdir(directory)
    # Get absolute path to the files
    directory = os.getcwd()
    # Get all the filenames
    filenames = os.listdir(directory)
    # sort the array based on the files' creation date
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    # read one file to extract its height, width and channels using the shape attribute
    array_col_size = cv2.imread(filenames[0])
    #examples is variable used to determine the number of images to be stored in one file. Its
    #provided as a function argument
    examples = num_of_images_per_file
    #initialize numpy array which will contain rgb data for one output file. One
    #output file contains data for a number of images specified by the 'examples' variable above
    training_set = np.empty((examples, array_col_size.shape[0], array_col_size.shape[1], array_col_size.shape[2]), dtype=np.uint8)
    # frame per second variable for housekeeping purposes in the for-loop below
    fps = 20
    #Loop end calculator
    loop_end = min(start_image_number + examples - 1, len(filenames))

    for i in range(start_image_number-1, loop_end):
        training_set[i] = cv2.imread(filenames[i])
        if not bool(i % (fps * 60)):
            print(f'{i //(fps * 60)} minute(s) of video processed')

    np.savez_compressed(f'../../Image_RGB_Values_{start_image_number}_{loop_end}', a = training_set)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))


def save_images_as_np_array(image_directory, array_filename):
    # Change the current python directory to where the files are located
    os.chdir(image_directory)
    # Get absolute path to the files
    image_directory = os.getcwd()
    # Get all the filenames
    filenames = os.listdir(image_directory)
    # sort the array based on the files' creation date
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))
    fps = 20
    # Append the numpy array to a list in a loop
    all_images = []
    for i in range(len(filenames)-1):
        A = cv2.imread(filenames[i])
        all_images.append(A)
        if not bool(i % (fps * 60)):
            print(f'{i //(fps * 60)} minute(s) of video processed')

    array_of_images = np.array(all_images, dtype=np.uint8)
    np.savez(array_filename, array_of_images)

# load numpy array from a file
def load_numpy_array(array_filename):
    container = np.load(array_filename)
    data = [container[key] for key in container]
    array_of_images = data[0]
    return array_of_images