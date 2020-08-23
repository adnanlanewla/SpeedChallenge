import cv2
import os
import numpy as np


def frame_extractor():
    #Create Image Folder
    os.mkdir('../data/Local_Data/Images')
    filename = '../data/train.mp4'
    #Get video object
    captured_video = cv2.VideoCapture(filename)
    # get the first frame from the video file. Each time the command below
    # is run, a subsequent image is read.
    success, image = captured_video.read()
    frame_number = 1
    fps = 20
    f = open('../data/train.txt', 'r')

    while success:
        second = (frame_number - 1) // fps
        minute = second // 60
        frame_number_within_one_second = frame_number % fps + fps * int(not bool(frame_number % fps))
        # read the speed of each image from train.txt and add it to the filename after an '_'
        speed = f.readline()
        speed = speed.rsplit('\n')[0]
        # name of each image
        image_name = f'../data/Local_Data/Images/frame_{minute}_{second%60}_{frame_number_within_one_second}_{speed}.jpg'
        # save the image at 100% quality
        cv2.imwrite(image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # Keep running the command below until there are no more frames left
        # which will be indicated by the 'success' boolean
        success, image = captured_video.read()
        if not bool(frame_number % (fps * 60)):
            print(f'{frame_number//(fps * 60)} minute(s) of video processed')

        frame_number += 1

    f.close()

def image_cropper():
    # Create Image Folder
    os.mkdir('../data/Local_Data/Cropped_Images')
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Local_Data/Images'
    # Change the current python directory to where the files are located
    os.chdir(directory)
    # Get absolute path to the files
    directory = os.getcwd()
    # Get all the filenames
    filenames = os.listdir(directory)
    # sort the array based on the files' creation date
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    array_col_size = cv2.imread(filenames[0])

    for i in range(len(filenames)):
        A = cv2.imread(filenames[i])
        cropped_A = A[20:360, :]
        image_name = f'../Cropped_Images/' + filenames[i]
        cv2.imwrite(image_name, cropped_A, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

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

if __name__ == '__main__':
    None