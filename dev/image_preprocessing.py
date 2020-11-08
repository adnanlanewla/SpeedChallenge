import cv2
import os
import numpy as np

def makedirectory(directory):
    '''
    Create the given directory if it doesn't already exist
    '''

    try:
        os.mkdir(directory)
    except FileExistsError:
        pass

def get_sorted_filenames_list(directory):
    # Get absolute path to the files
    directory = os.path.abspath(directory)
    # Get all the filenames
    filenames = os.listdir(directory)
    # sort the array based on the files' creation date
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    return  filenames

def frame_extractor(save_directory_name, video_file,
                    rescale=False,
                    rescale_factor = 0.25,
                    frame_naming=None):
    '''
    Extracts images from a given video

    Inputs:
    save_directory_name -- relative path to the directory where the image files will be saved
    video_file -- relative path to the video file that is to be parsed
    rescale -- indicate whether rescaling is required or not
    rescale_factor -- rescaling factor. Default is 0.25, i.e. rescale to 25% of original
    frame_naming -- Full path of the text file to be used for naming of saved files (optional).
    '''

    #Create directory if it doesn't exist
    makedirectory(save_directory_name)

    #Get full video path + filename from the video file name
    filename = os.path.abspath(video_file)
    #Get video object
    captured_video = cv2.VideoCapture(filename)
    # Get frames per second of the video
    fps = int(captured_video.get(cv2.CAP_PROP_FPS))
    # get the first frame from the video file. Each time the command below
    # is run, a subsequent image is read.
    success, image = captured_video.read()
    frame_number = 1
    #Image naming based on an external text file
    if frame_naming:
        f = open(os.path.abspath(frame_naming), 'r')

    while success:
        second = (frame_number - 1) // fps
        minute = second // 60
        frame_number_within_one_second = frame_number % fps + fps * int(not bool(frame_number % fps))

        if frame_naming:
            # read the speed of each image from the text file specified in frame_naming and add it to the filename
            # after an '_'
            speed = f.readline()
            speed = speed.rsplit('\n')[0]
            # name of each image
            image_name = f'{save_directory_name}/frame_{minute}_{second%60}_{frame_number_within_one_second}_{speed}.jpg'
        else:
            # name of each image
            image_name = f'{save_directory_name}/frame_{minute}_{second%60}_{frame_number_within_one_second}.jpg'

        # rescale the image if required
        if rescale:
            image = image_resize(image=image, scaling_factor=rescale_factor)

        # save the image at 100% quality
        cv2.imwrite(image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # Keep running the command below until there are no more frames left
        # which will be indicated by the 'success' boolean
        success, image = captured_video.read()
        if not bool(frame_number % (fps * 60)):
            print(f'{frame_number//(fps * 60)} minute(s) of video processed')

        frame_number += 1

def image_cropper(uncropped_img_dir, cropped_img_dir,
                  crop_locations=(230,360,100,540),
                  angle_crop=None):
    '''
    Crops images in the uncropped_img_dir folder according to the crop_locations argument

    Inputs:
    uncropped_img_dir -- relative path to the directory where the image files are located
    cropped_img_dir -- relative path to the directory where the cropped image files are to be saved.
    crop_locations -- (x_left, x_right, y_top, y_bottom)
    angle_crop (Optional) -- set to True if images are required to be cropped at an angle.
    '''
    #Get list of sorted filenames that are to be cropped
    filenames = get_sorted_filenames_list(uncropped_img_dir)
    #Make the directory where the cropped images will be stored if it doesn't already exist
    makedirectory(cropped_img_dir)

    #Unpack crop coordinates as scalars
    x_l, x_r, y_t, y_b = crop_locations

    for i in range(len(filenames)):
        A = cv2.imread(f'{uncropped_img_dir}/{filenames[i]}')
        cropped_A = A[x_l:x_r, y_t:y_b]
        if angle_crop:
            cropped_A = angled_crop(cropped_A)
        image_name = f'{cropped_img_dir}/{filenames[i]}'
        cv2.imwrite(image_name, cropped_A, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def angled_crop(image):
    '''
    Crops the image at an angle. All cropping parameters are hardcoded and should be variableized.
    Implementation should also be vectorized to avoid the ugly for-loop.
    '''
    for i in range(image.shape[0]):
        a = int(np.ceil(170-1.5*i))
        b = int(np.ceil(270+1.5*i))
        image[i, 0: max(0, a)] = 255
        image[i, min(439, b):] = 255
    return image

def speed_difference():
    '''
    Get a difference of speed by subtracting two consecutive speeds from the train file
    '''

    path = '../data/train.txt'
    data = np.loadtxt(path)
    data = np.diff(data)
    np.savetxt('../data/Speed_Differences.txt', data, fmt='%f')

def videocreator(img_directory_name, saved_video_name, fps=10):
    '''
    Creates a video from all the images in the img_directory_name folder

    Inputs:
    img_directory_name -- directory name in Local_Data where the image files are located
    saved_video_file -- name of the video file to be saved. File will be saved in the same folder where the images
                        are located unless the full path is provided
    fps -- frames per second of the video. Default is 10
    '''
    # Get list of sorted filenames that are to be merged in to a video
    filenames = get_sorted_filenames_list(img_directory_name)

    img = cv2.imread(f'{img_directory_name}/{filenames[0]}')
    height, width, layers = img.shape
    img = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{saved_video_name}.avi', fourcc, fps, (width, height))
    for i in range(len(filenames)):
        video.write(cv2.imread(f'{img_directory_name}/{filenames[i]}'))

    cv2.destroyAllWindows()
    video.release()

def image_resize(image, scaling_factor=0.5):
    return (cv2.resize(image, (0,0), fx=scaling_factor, fy=scaling_factor))

if __name__ == '__main__':
    pass