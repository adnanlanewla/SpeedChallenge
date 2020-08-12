import cv2
import os


def frame_extractor():
    #Create Image Folder
    os.mkdir('../data/Images')

    filename = '../data/train.mp4'
    #Get video file
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

if __name__ == '__main__':
    # leave this empty before closing the project
