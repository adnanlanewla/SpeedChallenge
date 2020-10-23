import cv2
import os
import numpy as np
import glob


def frame_extractor(video_filename='../data/train.mp4',dest_folder='../data/Images/train'):
    #Create Image Folder
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = video_filename
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

def speed_difference():
    path = '../data/train.txt'
    data = np.loadtxt(path)
    data = np.diff(data)
    np.savetxt('../data/Speed_Differences.txt', data, fmt='%f')

def video_writer(image_directory, video_filename):
    img_array = []
    filenames = os.listdir(image_directory)
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))
    for filename in filenames:
        img = cv2.imread(os.path.join(image_directory,filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()