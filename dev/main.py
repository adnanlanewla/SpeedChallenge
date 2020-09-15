from dev.models.keras_models import *
from dev.helper_functions import *
from dev.image_preprocessing import *
import dev.models.FlowNet_S as FlowNet_S
# import dev.generator



if __name__ == '__main__':
    model = FlowNet_S.FlowNet_S()

    # extract_frames = False
    # generate_pipeline = True
    # image_directory = '../data/Images'
    # if extract_frames:
    #     # renaming of files is done during extraction of frames
    #     frame_extractor()
    #
    # # this will generate the pipeline, we can use for training the keras model
    # if generate_pipeline:
    #     my_training_batch_generator, my_validation_batch_generator = generator.My_Custom_Generator(image_directory)
    #
    # print('Done')