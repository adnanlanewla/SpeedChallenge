from models.keras_models import *
from helper_functions import *
from image_preprocessing import *
import models.FlowNet_S as flownet_s
from pipeline import *
from helper_functions import  *
from imageio import imread, imwrite
import generator


if __name__ == '__main__':

    extract_frames = False
    optical_flow = False
    VGG_16 = False
    ConvLSTM = True
    image_directory = "..\data\Images"
    image_dir_train = '../data/Images/train'
    image_dir_test = '../data/Images/test'
    if extract_frames:
        # renaming of files is done during extraction of frames
        frame_extractor()
        print("Frame extraction from Video finished")
    if optical_flow:
        # Optical flow pipeline
        # 1) Create the optical flow images by passing in the image directory
        optical_flow_pipeline(image_directory)
        # 2) Create a video from the optical flow output images
        videocreator('../data/predicted_images', '../data/video_from_optiflow.avi')
        # 3) Use the images from the Optical flow output to pass it to VGG 16 network to predict acceleration and to extract features
        VGG16_pipeline('../data/predicted_images', batch_size=128)
        print("Optical flow pipeline finished")
    if VGG_16:
        # VGG 16 pipeline.
        VGG16_pipeline(image_directory, batch_size=32, normalize_y=True)
        print("VGG_16 pipeline finished")
    if ConvLSTM:
        # ConvLSTM pipeline
        ConvLSTM_pipeline(image_directory, batch_size=4, time_steps=20)
        print("ConvLSTM pipeline finished")

    print('Done')
