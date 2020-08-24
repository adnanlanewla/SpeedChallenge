from keras_models import *
from helper_functions import *
from image_preprocessing import *
import generator



if __name__ == '__main__':
    #model = VGG_model(150,150)
    #model.fit(x_train, y_train)

    # 1-100
    # 75-175
    # 150-250
    # 225-325
    #
    # {20400,480,640,3}

    extract_frames = False
    generate_pipeline = True
    image_directory = '../data/Images'
    if extract_frames:
        # renaming of files is done during extraction of frames
        frame_extractor()

    # this will generate the pipeline, we can use for training the keras model
    if generate_pipeline:
        my_training_batch_generator, my_validation_batch_generator = generator.My_Custom_Generator(image_directory)

    print('Done')