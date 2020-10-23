from models.keras_models import *
from helper_functions import *
from image_preprocessing import *
import models.FlowNet_S as flownet_s
from optical_flow_pipeline import *
from helper_functions import  *
from imageio import imread, imwrite
import generator

#The flow output pattern matches the Pytorch output exactly. The background color of the output will depend on
#which image is the first in image concantenation.
if __name__ == '__main__':

    # extract_frames = False
    optical_flow = False
    VGG_16 = True
    # generate_pipeline = True
    image_directory = "..\data\Images"
    image_dir_train = '../data/Images/train'
    image_dir_test = '../data/Images/test'
    # if extract_frames:
    #     # renaming of files is done during extraction of frames
    #     frame_extractor()
    #
    # # this will generate the pipeline, we can use for training the keras model
    # if generate_pipeline:
    #     my_training_batch_generator, my_validation_batch_generator = generator.My_Custom_Generator(image_directory)
    #
    if optical_flow:
        optical_flow_pipeline(image_directory)
        #video_writer('../data/predicted_images', '../data/video_from_optiflow.avi')
        VGG16_pipeline('../data/predicted_images', batch_size=128)
    if VGG_16:
        VGG16_pipeline(image_directory, batch_size=32, normalize_y=True)

    print('Done')
